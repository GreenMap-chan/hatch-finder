from .hatch_finder import HatchFinder
from .logger import Logger

from collections import Counter
from fractions import Fraction
import json
import torch
import math
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Any, Literal

from .config import Config, ModelSettings, config_from_pt_data, model_config_from_pt_data, save_config, merge_model_settings
from .architecture_transfer import transfer_weights
from .checkpoint import restore_training_checkpoint
from .hatch_dataset import HatchDataset
from .image_geometry import canvas_size
from . import losses
from torch.utils.data import DataLoader, RandomSampler


class _Unset:
    pass


UNSET = _Unset()


class _EpochSampler(RandomSampler):
    """Capture the shuffle RNG after DataLoader has seeded its workers."""

    epoch_start_state = None
    resume_state = None

    def __iter__(self):
        if self.resume_state is not None:
            self.generator.set_state(self.resume_state.cpu())
            self.resume_state = None
        self.epoch_start_state = self.generator.get_state()
        yield from super().__iter__()


def _intermediate_validation_batches(batch_count: int, accumulation: int, interval: float) -> set[int]:
    if interval == 1 or batch_count == 0:
        return set()
    # Compare exact multiples of the configured fraction, even for very small intervals.
    spacing = batch_count * Fraction(str(interval))
    return {
        end for end in range(accumulation, batch_count, accumulation)
        if end // spacing > (end - accumulation) // spacing
    }


class Train:
    def __init__(
        self,
        config: Config | None = None,
        *,
        output_path: Path | str | _Unset = UNSET,
        dataset_path: Path | str | _Unset = UNSET,
        device: Literal["auto", "cpu", "cuda"] | _Unset = UNSET,
        **training_overrides: Any,
    ) -> None:
        config, dataset_changed = self._prepare_config(
            config, output_path, dataset_path, device, training_overrides
        )
        if config.data.dataset is None:
            raise ValueError("dataset_path must be specified for training")
        if config.output.directory is None:
            raise ValueError("output_path must be specified for training")

        if config.model.pixel_correlation.enabled and config.training.batch_size != 1:
            raise ValueError("Pixel correlation requires batch_size=1; use gradient accumulation")
        self.config = config
        self.start_epoch = 0
        self.resume_progress = None
        self.consecutive_skipped_updates = 0
        transfer = config.training.load_new_model_config
        self.checkpoint_path = None if transfer else config.training.checkpoint_path
        self.dataset_changed = dataset_changed

        torch.manual_seed(config.training.seed)
        self.data_loader_generator = torch.Generator().manual_seed(config.training.seed)

        self.model = HatchFinder(config)
        self.bf16_enabled = config.training.bf16 and self.model.device.type == "cuda"
        if self.bf16_enabled and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is enabled, but the current CUDA device does not support it")
        self.optimizer = self.create_optimizer(
            config.training.learning_rate, config.training.weight_decay
        )
        transfer_report = None
        if transfer:
            source = config.training.load_model_path or config.training.checkpoint_path
            transfer_report = transfer_weights(self.model, source)
        elif config.training.load_model_path is not None and self.checkpoint_path is None:
            self.model.load_model(config.training.load_model_path)

        self.output_path = self._create_output_directory(
            config.output.directory,
            reuse=not transfer and (self.checkpoint_path is not None or not config.output.unique_run_directory),
        )
        config.output.directory = self.output_path
        self.model.config.output.directory = self.output_path
        save_config(config, self.output_path / "config.yaml")
        self.logger = Logger(self.output_path, config.output.log_file_name)
        if transfer_report:
            for line in transfer_report:
                self.logger.log(line)

    @staticmethod
    def _prepare_config(
        config: Config | None,
        output_path: Path | str | _Unset,
        dataset_path: Path | str | _Unset,
        device: Literal["auto", "cpu", "cuda"] | _Unset,
        training_overrides: dict[str, Any],
    ) -> tuple[Config, bool]:
        if config is None:
            config = Config()
        else:
            config = config.model_copy(deep=True)

        resumable_settings = {"epochs", "max_consecutive_skipped_updates", "validation_interval"}
        resume_overrides = resumable_settings & (config.training.model_fields_set | training_overrides.keys())
        output_specified = "directory" in config.output.model_fields_set or not isinstance(output_path, _Unset)
        dataset_specified = "dataset" in config.data.model_fields_set or not isinstance(dataset_path, _Unset)
        device_specified = "device" in config.runtime.model_fields_set or not isinstance(device, _Unset)
        window_overrides = config.data.search_mask_generation.model_dump(exclude_unset=True)
        size_overrides = config.data.model_dump(include={"imgsz", "rect"}, exclude_unset=True)

        if training_overrides:
            config.training = type(config.training).model_validate({
                **config.training.model_dump(),
                **training_overrides,
            })
        if not isinstance(output_path, _Unset):
            config.output = type(config.output).model_validate({
                **config.output.model_dump(),
                "directory": output_path,
            })
        if not isinstance(dataset_path, _Unset):
            config.data = type(config.data).model_validate({
                **config.data.model_dump(),
                "dataset": dataset_path,
            })
        if not isinstance(device, _Unset):
            config.runtime = type(config.runtime).model_validate({
                **config.runtime.model_dump(),
                "device": device,
            })

        checkpoint_path = config.training.checkpoint_path
        dataset_changed = False
        if config.training.load_new_model_config:
            source = config.training.load_model_path or checkpoint_path
            if source is None:
                raise ValueError("load_new_model_config requires load_model_path or checkpoint_path")
            saved = model_config_from_pt_data(torch.load(source, map_location="cpu", weights_only=True))
            if saved is None:
                raise ValueError("Architecture transfer requires a .pt file with model configuration")
            config.model = ModelSettings.model_validate(merge_model_settings(
                saved, config.model.model_dump(exclude_unset=True),
            ))
            return config, False
        if checkpoint_path is not None:
            allowed_overrides = resumable_settings | {"checkpoint_path", "load_new_model_config"}
            unsupported_overrides = training_overrides.keys() - allowed_overrides
            if unsupported_overrides:
                raise ValueError(
                    "Training settings cannot be overridden when resuming: "
                    f"{sorted(unsupported_overrides)}"
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_config = config_from_pt_data(checkpoint)
            if checkpoint_config is None:
                checkpoint_config = Config()
            # This invocation is an ordinary resume, even if its source was an experiment.
            checkpoint_config.training.load_new_model_config = False
            if resume_overrides:
                checkpoint_config.training = type(checkpoint_config.training).model_validate({
                    **checkpoint_config.training.model_dump(),
                    **config.training.model_dump(include=resume_overrides),
                })
            if output_specified:
                checkpoint_config.output.directory = config.output.directory
            if dataset_specified:
                previous_dataset = checkpoint_config.data.dataset
                new_dataset = config.data.dataset
                dataset_changed = (
                    previous_dataset is None
                    or new_dataset is None
                    or previous_dataset.resolve() != new_dataset.resolve()
                )
                checkpoint_config.data.dataset = config.data.dataset
            if size_overrides:
                previous = checkpoint_config.data
                updated = type(previous).model_validate({**previous.model_dump(), **size_overrides})
                old_geometry = (canvas_size(previous.imgsz), previous.rect) if previous.imgsz is not None else None
                new_geometry = (canvas_size(updated.imgsz), updated.rect) if updated.imgsz is not None else None
                dataset_changed |= old_geometry != new_geometry
                checkpoint_config.data = updated
            if window_overrides:
                previous = checkpoint_config.data.search_mask_generation
                updated = type(previous).model_validate({**previous.model_dump(), **window_overrides})
                training_only = {"negative_fraction", "outside_window_share_of_negatives"}
                validation_windows_changed = (
                    (updated.enabled or previous.enabled)
                    and updated.model_dump(exclude=training_only) != previous.model_dump(exclude=training_only)
                )
                dataset = checkpoint_config.data.dataset
                if validation_windows_changed and not dataset_changed and dataset is not None:
                    with (dataset / "valid" / "manifest.jsonl").open(encoding="utf-8") as manifest:
                        dataset_changed = any(not json.loads(row).get("search_mask") for row in manifest)
                checkpoint_config.data.search_mask_generation = updated
            if device_specified:
                checkpoint_config.runtime.device = config.runtime.device
            checkpoint_config.training.checkpoint_path = checkpoint_path
            config = checkpoint_config

        if (
            config.training.load_model_path is not None
            and config.training.checkpoint_path is None
        ):
            weights = torch.load(
                config.training.load_model_path,
                map_location="cpu",
                weights_only=True,
            )
            weights_model_config = model_config_from_pt_data(weights)
            if weights_model_config is not None:
                for field in ("transformers", "hatch_transformers"):
                    saved_by_level = {
                        settings.level: settings
                        for settings in getattr(weights_model_config, field)
                    }
                    for settings in getattr(config.model, field):
                        if "dropout" not in settings.model_fields_set:
                            continue
                        if settings.level not in saved_by_level:
                            raise ValueError(
                                f"Cannot override {field} dropout at level {settings.level}: "
                                "the loaded model has no transformer at this level"
                            )
                        saved_by_level[settings.level].dropout = settings.dropout
                config.model = weights_model_config

        return config, dataset_changed

    @staticmethod
    def _create_output_directory(directory: Path, *, reuse: bool) -> Path:
        if reuse:
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        candidate = directory
        suffix = 0
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                if not candidate.exists():
                    raise
                candidate = directory.with_name(f"{directory.name}_{suffix}")
                suffix += 1

    def autocast_context(self):
        return torch.autocast(
            device_type=self.model.device.type,
            dtype=torch.bfloat16,
            enabled=self.bf16_enabled,
        )

    def create_optimizer(
        self,
        lr: float,
        weight_decay: float = 0.01,
    ):
        branch = getattr(self.model, "pixel_correlation", None)
        branch_ids = {id(p) for p in branch.parameters()} if branch is not None else set()
        multiplier = self.model.config.training.pixel_correlation_lr_multiplier if branch is not None else 1.0
        groups = [
            {"params": [], "weight_decay": weight_decay, "lr": lr},
            {"params": [], "weight_decay": 0.0, "lr": lr},
        ]
        if branch is not None:
            groups.extend([
                {"params": [], "weight_decay": weight_decay, "lr": lr * multiplier},
                {"params": [], "weight_decay": 0.0, "lr": lr * multiplier},
            ])
        for param in self.model.parameters():
            if param.requires_grad:
                index = (2 if id(param) in branch_ids else 0) + (0 if param.ndim >= 2 else 1)
                groups[index]["params"].append(param)
        return torch.optim.AdamW(groups, lr=lr)

    def create_scheduler(self, total_steps: int, warmup_steps):
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.training.eta_min,
        )
        self.cosine_scheduler = cosine

        if not warmup_steps:
            self.scheduler = cosine
            return self.scheduler

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[
                warmup_steps,
            ],
        )

        return self.scheduler

    def extend_scheduler(self, total_steps: int, warmup_steps: int) -> None:
        cosine = self.cosine_scheduler
        new_t_max = total_steps - warmup_steps
        if new_t_max <= cosine.T_max:
            return
        cosine.T_max = new_t_max
        if cosine.last_epoch >= 0:
            rates = [
                cosine.eta_min + (base_lr - cosine.eta_min)
                * (1 + math.cos(math.pi * cosine.last_epoch / new_t_max)) / 2
                for base_lr in cosine.base_lrs
            ]
            for group, rate in zip(cosine.optimizer.param_groups, rates):
                group["lr"] = rate

    @staticmethod
    def get_accumulation_examples_count(
        batch_num: int,
        train_batches_count: int,
        batch_size: int,
        dataset_size: int,
        gradient_accum_steps: int,
    ) -> int:
        end_batch = min(
            batch_num + gradient_accum_steps,
            train_batches_count,
        )

        start_example = batch_num * batch_size
        end_example = min(
            end_batch * batch_size,
            dataset_size,
        )

        return end_example - start_example

    def _create_loaders(self):
        num_workers = self.config.training.num_workers
        dataset_train = HatchDataset(
            self.config.data.dataset,
            "train",
            self.config.data,
            self.config.augmentation,
            augment=True,
            include_sample_ids=True,
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=self.config.training.batch_size,
            sampler=_EpochSampler(dataset_train, generator=self.data_loader_generator),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.model.device.type == "cuda",
            collate_fn=dataset_train.collate_fn,
            generator=self.data_loader_generator,
        )
        dataset_valid = HatchDataset(
            self.config.data.dataset,
            "valid",
            self.config.data,
            self.config.augmentation,
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.model.device.type == "cuda",
            collate_fn=dataset_valid.collate_fn
        )

        return train_loader, valid_loader

    def _restore_training(self, scheduler, total_steps: int, warmup_steps: int):
        if self.checkpoint_path is None:
            return None, 0
        state = restore_training_checkpoint(self.model, self.optimizer, scheduler, self.checkpoint_path,
                                            include_progress=True)
        self.start_epoch, best_metric, patience_counter, self.consecutive_skipped_updates, self.resume_progress = state
        self.extend_scheduler(total_steps, warmup_steps)
        if self.dataset_changed:
            self.resume_progress = None
            return None, 0
        return best_metric, patience_counter

    def _save_checkpoint(self, scheduler, epoch, best_metric, patience_counter, path, epoch_progress=None):
        self.model.save_checkpoint(
            self.optimizer, scheduler, epoch, best_metric, patience_counter, path,
            consecutive_skipped_updates=self.consecutive_skipped_updates, epoch_progress=epoch_progress,
        )

    def _ensure_best_checkpoint(self, scheduler, best_metric, val_loss: float) -> float:
        if best_metric is None:
            best_metric = val_loss
        best_path = self.output_path / "best.pt"
        if self.checkpoint_path is None or self.dataset_changed:
            self._save_checkpoint(scheduler, self.start_epoch - 1, best_metric, 0, best_path)
        elif not best_path.exists():
            previous_best = self.checkpoint_path.parent / "best.pt"
            if previous_best.exists():
                shutil.copy2(previous_best, best_path)
            else:
                best_metric = val_loss
                epoch = self.start_epoch if self.resume_progress else self.start_epoch - 1
                self._save_checkpoint(scheduler, epoch, best_metric, 0, best_path, self.resume_progress)
        return best_metric

    def _run_training_epoch(self, train_loader: DataLoader, scheduler, epoch: int, epochs: int, *,
                            start_batch=0, validation_batches=(), validate=None):
        train_dataset_length = len(train_loader.dataset)
        train_batches_count = len(train_loader)
        gradient_accum_steps = self.config.training.gradient_accumulation_steps
        batch_size = self.config.training.batch_size
        skip_limit = self.config.training.max_consecutive_skipped_updates

        gradient_norms = []
        sample_counts = Counter()
        epoch_loss = 0.0
        trained_examples = 0
        skipped_updates = 0
        self.optimizer.zero_grad()

        def remaining_batches():
            for index, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", unit="batch")):
                if index >= start_batch:
                    yield index, batch
                    if index + 1 in validation_batches:
                        validate(epoch, index + 1)

        for batch_num, example in remaining_batches():
            current_batch_size = example["drawing"].shape[0]
            sample_counts.update(example.get("sample_kind", ()))

            if batch_num % gradient_accum_steps == 0:
                group_start = batch_num + 1
                group_end = min(batch_num + gradient_accum_steps, train_batches_count)
                group_loss = 0.0
                group_ids = []
                skip_group = False
                accumulation_examples_count = self.get_accumulation_examples_count(
                    batch_num=batch_num,
                    train_batches_count=train_batches_count,
                    batch_size=batch_size,
                    dataset_size=train_dataset_length,
                    gradient_accum_steps=gradient_accum_steps,
                )
            if skip_group:
                continue

            batch_ids = example["sample_ids"]
            group_ids.extend(batch_ids)
            with self.autocast_context():
                example_loss = self.train_one_example(example)[0]

            loss_value = example_loss.detach().item()
            if not math.isfinite(loss_value):
                del example_loss
                problem = f"non-finite loss={loss_value}, batch sample IDs={batch_ids}"
            else:
                group_loss += loss_value * current_batch_size
                loss_for_backward = example_loss * current_batch_size / accumulation_examples_count
                loss_for_backward.backward()
                if batch_num + 1 < group_end:
                    continue

                grad_norm = self.model.clip_grad_norm().item()
                if math.isfinite(grad_norm):
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.consecutive_skipped_updates = 0
                    gradient_norms.append(grad_norm)
                    epoch_loss += group_loss
                    trained_examples += accumulation_examples_count
                    continue
                problem = f"non-finite gradient norm={grad_norm}, group sample IDs={group_ids}"

            self.optimizer.zero_grad()
            skip_group = True
            skipped_updates += 1
            self.consecutive_skipped_updates += 1
            message = (
                f"Epoch {epoch + 1}/{epochs}, batch {batch_num + 1}/{train_batches_count}: "
                f"skipped update (batches {group_start}-{group_end}): {problem} | "
                f"skipped this epoch={skipped_updates}, consecutive={self.consecutive_skipped_updates}, limit={skip_limit}"
            )
            self.logger.log(message)
            if self.consecutive_skipped_updates > skip_limit:
                raise FloatingPointError(message)

        if gradient_norms:
            clipped_steps = sum(norm > self.config.training.max_grad_norm for norm in gradient_norms)
            p95_grad_norm = torch.tensor(gradient_norms).quantile(0.95).item()
            metrics = (
                f"loss: {epoch_loss / trained_examples:.3f} | max_grad_norm: {max(gradient_norms):.3f} | "
                f"p95_grad_norm: {p95_grad_norm:.3f} | clipped_steps: {clipped_steps}/{len(gradient_norms)} "
                f"({clipped_steps / len(gradient_norms):.1%})"
            )
        else:
            metrics = "loss: n/a (no successful updates)"
        part = f" (batches {start_batch + 1}-{train_batches_count})" if start_batch else ""
        self.logger.log(f"epoch {epoch + 1}/{epochs}{part}: {metrics} | skipped_updates: {skipped_updates}")

        if sample_counts:
            counts = ", ".join(
                f"{kind}={sample_counts[kind]} ({sample_counts[kind] / sample_counts.total():.1%})"
                for kind in ("positive", "absent", "outside")
            )
            self.logger.log(f"Epoch {epoch + 1} search windows: {counts}")

    def train(self):
        epochs = self.config.training.epochs
        gradient_accum_steps = self.config.training.gradient_accumulation_steps
        patience = self.config.training.patience
        metric = self.config.training.metric
        warmup_epochs = self.config.training.warmup_epochs
        self.logger.log(f"Model size: {self.model.get_model_size() / 1_000_000:.2f}M")
        self.logger.log(f"BF16 autocast: {'enabled' if self.bf16_enabled else 'disabled'}")

        train_loader, valid_loader = self._create_loaders()

        batch_count = len(train_loader)
        validation_batches = _intermediate_validation_batches(
            batch_count, gradient_accum_steps, self.config.training.validation_interval
        )
        epoch_steps = math.ceil(batch_count / gradient_accum_steps)
        total_steps = epoch_steps * epochs
        scheduler = self.create_scheduler(total_steps, warmup_epochs * epoch_steps)

        best_metric, patience_counter = self._restore_training(
            scheduler, total_steps, warmup_epochs * epoch_steps
        )
        if self.resume_progress and self.resume_progress["batch_count"] != batch_count:
            self.logger.log("Training loader length changed. Restarting the partial epoch at its first batch.")
            self.resume_progress = None
        # Workers inherit this before the first training batch, also on resume.
        train_loader.dataset.window_start_epoch = self.start_epoch
        val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)
        best_metric = self._ensure_best_checkpoint(scheduler, best_metric, val_loss)

        self.logger.log(f"Initials metrics - valid_loss: {val_loss:.5f} | BCE: {bce_loss:.5f} | Dice: {dice_loss:.5f}")

        start_batch = 0
        if self.resume_progress:
            progress = self.resume_progress
            start_batch = progress["batches_completed"]
            train_loader.sampler.resume_state = progress["shuffle_state"]
            train_loader.dataset.window_start_epoch = progress["window_start_epoch"]
            torch.set_rng_state(progress["torch_rng_state"].cpu())
            if self.model.device.type == "cuda" and "cuda_rng_state" in progress:
                torch.cuda.set_rng_state(progress["cuda_rng_state"].cpu(), self.model.device)
            self.logger.log(f"Resuming epoch {self.start_epoch + 1}/{epochs} at batch {start_batch + 1}/{batch_count}")

        def validate(epoch, completed_batches):
            nonlocal best_metric, patience_counter
            intermediate = completed_batches < batch_count
            # An extra validation iterator consumes a CPU RNG draw for worker seeds.
            with torch.random.fork_rng(devices=[], enabled=intermediate):
                val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)
            position = ""
            if validation_batches:
                position = f"epoch {epoch + 1}/{epochs}, batch {completed_batches}/{batch_count} | "
            self.logger.log(f"{position}valid_loss: {val_loss:.5f} | BCE: {bce_loss:.5f} | Dice: {dice_loss:.5f}")

            if val_loss < best_metric:
                patience_counter = 0
                best_metric = val_loss
                progress = None
                if intermediate:
                    progress = {
                        "batches_completed": completed_batches,
                        "batch_count": batch_count,
                        "shuffle_state": train_loader.sampler.epoch_start_state,
                        "window_start_epoch": train_loader.dataset.window_start_epoch,
                        "torch_rng_state": torch.get_rng_state(),
                    }
                    if self.model.device.type == "cuda":
                        progress["cuda_rng_state"] = torch.cuda.get_rng_state(self.model.device)
                self._save_checkpoint(scheduler, epoch, best_metric, patience_counter,
                                      self.output_path / "best.pt", progress)
                self.logger.log(f"Saved best: {best_metric:.3f} ({metric})")

        for i in range(self.start_epoch, epochs):
            if not start_batch:
                patience_counter += 1
            self._run_training_epoch(train_loader, scheduler, i, epochs, start_batch=start_batch,
                                     validation_batches=validation_batches, validate=validate)
            validate(i, batch_count)
            start_batch = 0
            self._save_checkpoint(
                scheduler,
                i,
                best_metric,
                patience_counter,
                self.output_path / "last.pt",
            )

            if patience_counter >= patience:
                break

        self.logger.log(f"=== Saved best: {best_metric:.3f} ===")

    def get_valid_loss(self, dataset: DataLoader):
        self.model.eval()

        with torch.no_grad():
            epoch_loss = torch.zeros((), device=self.model.device)
            epoch_bce_loss = torch.zeros((), device=self.model.device)
            epoch_dice_loss = torch.zeros((), device=self.model.device)
            epoch_dice_count = 0
            valid_examples_count = 0
            for example in tqdm(dataset, desc="valid", unit="batch"):
                with self.autocast_context():
                    example_loss, bce_loss, dice_loss = self.model.get_example_loss(
                        example["drawing"],
                        example["search_mask"],
                        example["hatch"],
                        example["target"],
                    )

                batch_size = example["drawing"].shape[0]
                valid_examples_count += batch_size
                epoch_loss += example_loss.detach() * batch_size
                epoch_bce_loss += bce_loss.detach() * batch_size
                target_pixels = (
                    example["target"] * example["search_mask"]
                ).sum(dim=(1, 2, 3))
                positive_examples = int((target_pixels > 0).sum().item())
                epoch_dice_loss += dice_loss.detach() * positive_examples
                epoch_dice_count += positive_examples

            average_loss = (epoch_loss / valid_examples_count).item()
            average_bce_loss = (epoch_bce_loss / valid_examples_count).item()
            average_dice_loss = (
                (epoch_dice_loss / epoch_dice_count).item()
                if epoch_dice_count > 0
                else 0.0
            )

        self.model.train()

        return average_loss, average_bce_loss, average_dice_loss

    def train_one_example(self, example: dict[str, torch.Tensor]):
        if self.config.training.loss == "bce_dice":
            return self.model.get_example_loss(
                example["drawing"], example["search_mask"],
                example["hatch"], example["target"],
            )

        logits = self.model(example["drawing"], example["search_mask"], example["hatch"])
        return losses.get_loss(
            logits,
            example["target"],
            example["search_mask"],
            dice_fn=lambda outputs, targets, mask: losses.get_tversky(
                outputs, targets, mask,
                fp_weight=self.config.training.tversky_fp_weight,
                fn_weight=self.config.training.tversky_fn_weight,
            ),
        )
