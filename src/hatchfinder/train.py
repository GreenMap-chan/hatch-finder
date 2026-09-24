from .hatch_finder import HatchFinder
from .logger import Logger

import torch
import math
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Any, Literal

from .config import Config, config_from_pt_data, model_config_from_pt_data, save_config
from .hatch_dataset import HatchDataset
from torch.utils.data import DataLoader


class _Unset:
    pass


UNSET = _Unset()


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

        self.config = config
        self.start_epoch = 0
        self.checkpoint_path = config.training.checkpoint_path
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
        if config.training.load_model_path is not None and self.checkpoint_path is None:
            self.model.load_model(config.training.load_model_path)

        self.output_path = self._create_output_directory(
            config.output.directory,
            reuse=self.checkpoint_path is not None or not config.output.unique_run_directory,
        )
        config.output.directory = self.output_path
        self.model.config.output.directory = self.output_path
        save_config(config, self.output_path / "config.yaml")
        self.logger = Logger(self.output_path, config.output.log_file_name)

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

        epochs_specified = "epochs" in config.training.model_fields_set or "epochs" in training_overrides
        output_specified = "directory" in config.output.model_fields_set or not isinstance(output_path, _Unset)
        dataset_specified = "dataset" in config.data.model_fields_set or not isinstance(dataset_path, _Unset)
        device_specified = "device" in config.runtime.model_fields_set or not isinstance(device, _Unset)

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
        if checkpoint_path is not None:
            unsupported_overrides = set(training_overrides) - {"checkpoint_path", "epochs"}
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

            if epochs_specified:
                checkpoint_config.training = type(checkpoint_config.training).model_validate({
                    **checkpoint_config.training.model_dump(),
                    "epochs": config.training.epochs,
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
        decay_parameters = []
        no_decay_parameters = []
        named_parametrs = list(self.model.named_parameters())

        for decay_parametr in self.config.training.decay_parameters:
            removed = True
            added = False
            while removed:
                removed = False
                for item in named_parametrs:
                    name, param = item
                    if decay_parametr in name:
                        decay_parameters.append(param)
                        named_parametrs.remove(item)
                        removed = True
                        added = True
                        break
            if not added:
                raise(ValueError(f"Матрица {decay_parametr} не найдена"))

        no_decay_parameters = [m for _, m in named_parametrs]

        return torch.optim.AdamW(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": weight_decay,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                },
            ],
            lr=lr,
        )

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
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=self.config.training.batch_size,
            shuffle=True,
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
        self.start_epoch, best_metric, patience_counter = self.model.load_checkpoint(
            self.optimizer, scheduler, self.checkpoint_path
        )
        self.extend_scheduler(total_steps, warmup_steps)
        if self.dataset_changed:
            return None, 0
        return best_metric, patience_counter

    def _ensure_best_checkpoint(self, scheduler, best_metric, val_loss: float) -> float:
        if best_metric is None:
            best_metric = val_loss
        best_path = self.output_path / "best.pt"
        if self.checkpoint_path is None or self.dataset_changed:
            self.model.save_checkpoint(
                self.optimizer, scheduler, self.start_epoch - 1, best_metric, 0, best_path
            )
        elif not best_path.exists():
            previous_best = self.checkpoint_path.parent / "best.pt"
            if previous_best.exists():
                shutil.copy2(previous_best, best_path)
            else:
                best_metric = val_loss
                self.model.save_checkpoint(
                    self.optimizer, scheduler, self.start_epoch - 1, best_metric, 0, best_path
                )
        return best_metric

    def _run_training_epoch(self, train_loader: DataLoader, scheduler, epoch: int, epochs: int):
        train_dataset_length = len(train_loader.dataset)
        train_batches_count = len(train_loader)
        gradient_accum_steps = self.config.training.gradient_accumulation_steps
        batch_size = self.config.training.batch_size

        gradient_norms = []
        epoch_loss = torch.zeros((), device=self.model.device)
        self.optimizer.zero_grad()
        for batch_num, example in enumerate(tqdm(train_loader, total=train_batches_count, desc=f"epoch {epoch+1}/{epochs}",unit="batch",)):
            current_batch_size = example["drawing"].shape[0]

            # В начале каждой группы определяем,
            # сколько batch реально будет накоплено
            if batch_num % gradient_accum_steps == 0:
                accumulation_examples_count = self.get_accumulation_examples_count(
                    batch_num=batch_num,
                    train_batches_count=train_batches_count,
                    batch_size=batch_size,
                    dataset_size=train_dataset_length,
                    gradient_accum_steps=gradient_accum_steps,
                )

            with self.autocast_context():
                example_loss, _, _ = self.train_one_example(example)

            epoch_loss += example_loss.detach() * current_batch_size

            loss_for_backward = (example_loss * current_batch_size / accumulation_examples_count)
            loss_for_backward.backward()


            if not (batch_num + 1) % gradient_accum_steps or train_batches_count == batch_num + 1:
                grad_norm = self.model.clip_grad_norm()
                gradient_norms.append(grad_norm.detach())

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        average_loss = (epoch_loss / train_dataset_length).item()
        gradient_norms = torch.stack(gradient_norms).cpu().tolist()
        clipped_steps = sum(
            grad_norm > self.config.training.max_grad_norm
            for grad_norm in gradient_norms
        )
        clipped_steps_percent = clipped_steps / len(gradient_norms) * 100
        p95_grad_norm = torch.tensor(gradient_norms).quantile(0.95).item()

        return average_loss, gradient_norms, clipped_steps, clipped_steps_percent, p95_grad_norm

    def train(self):
        epochs = self.config.training.epochs
        gradient_accum_steps = self.config.training.gradient_accumulation_steps
        patience = self.config.training.patience
        metric = self.config.training.metric
        warmup_epochs = self.config.training.warmup_epochs
        self.logger.log(f"Размер модели: {self.model.get_model_size() / 1_000_000:.2f}M")
        self.logger.log(f"BF16 autocast: {'enabled' if self.bf16_enabled else 'disabled'}")

        train_loader, valid_loader = self._create_loaders()

        epoch_steps = math.ceil(len(train_loader) / gradient_accum_steps)
        total_steps = epoch_steps * epochs
        scheduler = self.create_scheduler(total_steps, warmup_epochs * epoch_steps)

        best_metric, patience_counter = self._restore_training(
            scheduler, total_steps, warmup_epochs * epoch_steps
        )
        val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)
        best_metric = self._ensure_best_checkpoint(scheduler, best_metric, val_loss)

        self.logger.log(f"Initials metrics - valid_loss: {val_loss:.5f} | BCE: {bce_loss:.5f} | Dice: {dice_loss:.5f}")

        for i in range(self.start_epoch, epochs):
            patience_counter += 1

            average_loss, gradient_norms, clipped_steps, clipped_steps_percent, p95_grad_norm = (
                self._run_training_epoch(train_loader, scheduler, i, epochs)
            )

            val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)

            self.logger.log(
                f"epoch {i + 1}/{epochs}: loss: {average_loss:.3f} | "
                f"max_grad_norm: {max(gradient_norms):.3f} | "
                f"p95_grad_norm: {p95_grad_norm:.3f} | "
                f"clipped_steps: {clipped_steps}/{len(gradient_norms)} "
                f"({clipped_steps_percent:.1f}%)"
            )
            self.logger.log(f"valid_loss: {val_loss:.5f} | BCE: {bce_loss:.5f} | Dice: {dice_loss:.5f}")

            if val_loss < best_metric:
                patience_counter = 0
                best_metric = val_loss
                self.model.save_checkpoint(
                    self.optimizer,
                    scheduler,
                    i,
                    best_metric,
                    patience_counter,
                    self.output_path / "best.pt",
                )

                self.logger.log(f"Saved best: {best_metric:.3f} ({metric})")

            self.model.save_checkpoint(
                self.optimizer,
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
        return self.model.get_example_loss(example["drawing"], example["search_mask"], example["hatch"], example["target"])
