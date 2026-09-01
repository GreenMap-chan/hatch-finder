from .hatch_finder import HatchFinder
from .logger import Logger

import torch
import math
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
        if config is None:
            config = Config()
        else:
            config = config.model_copy(deep=True)

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
        if checkpoint_path is not None:
            unsupported_overrides = set(training_overrides) - {"checkpoint_path"}
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

            if not isinstance(output_path, _Unset):
                checkpoint_config.output = config.output
            if not isinstance(dataset_path, _Unset):
                checkpoint_config.data = config.data
            if not isinstance(device, _Unset):
                checkpoint_config.runtime = config.runtime
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

        if config.data.dataset is None:
            raise ValueError("dataset_path must be specified for training")
        if config.output.directory is None:
            raise ValueError("output_path must be specified for training")

        self.config = config
        self.lr = config.training.learning_rate
        self.output_path = config.output.directory
        self.transformer_blocks = []
        self.start_epoch = 0
        self.checkpoint_path = config.training.checkpoint_path
        self.seed = config.training.seed

        torch.manual_seed(self.seed)
        self.data_loader_generator = torch.Generator()
        self.data_loader_generator.manual_seed(self.seed)

        self.output_path.mkdir(exist_ok=True, parents=True)
        save_config(config, self.output_path / "config.yaml")

        self.model = HatchFinder(config)
        self.logger = Logger(self.output_path, config.output.log_file_name)
        self.bf16_enabled = (
            config.training.bf16
            and self.model.device.type == "cuda"
        )
        if self.bf16_enabled and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is enabled, but the current CUDA device does not support it")
        self.optimizer = self.create_optimizer(
            config.training.learning_rate,
            config.training.weight_decay,
        )

        if (
            config.training.load_model_path is not None
            and self.checkpoint_path is None
        ):
            self.model.load_model(config.training.load_model_path)

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

    def train(self):
        dataset_path = self.config.data.dataset
        epochs = self.config.training.epochs
        gradient_accum_steps = self.config.training.gradient_accumulation_steps
        patience = self.config.training.patience
        metric = self.config.training.metric
        warmup_epochs = self.config.training.warmup_epochs
        num_workers = self.config.training.num_workers
        batch_size = self.config.training.batch_size

        self.logger.log(f"Размер модели: {self.model.get_model_size() / 1_000_000:.2f}M")
        self.logger.log(f"BF16 autocast: {'enabled' if self.bf16_enabled else 'disabled'}")

        dataset_train = HatchDataset(
            dataset_path,
            "train",
            self.config.data,
            self.config.augmentation,
            augment=True,
        )
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.model.device.type == "cuda",
            collate_fn=dataset_train.collate_fn,
            generator=self.data_loader_generator,
        )
        dataset_valid = HatchDataset(
            dataset_path,
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

        epoch_steps = math.ceil(len(train_loader) / gradient_accum_steps)
        total_steps = epoch_steps * epochs
        scheduler = self.create_scheduler(total_steps, warmup_epochs * epoch_steps)

        best_metric = None
        patience_counter = 0
        if self.checkpoint_path is not None:
            self.start_epoch, best_metric, patience_counter = self.model.load_checkpoint(
                self.optimizer,
                scheduler,
                self.checkpoint_path,
            )

        possible_metrics_names = ["val_loss"]
        if not metric in possible_metrics_names:
            raise ValueError(f"Некрректное значение параметра metric. Допустимые: {possible_metrics_names}")

        val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)
        if best_metric is None:
            best_metric = val_loss

        self.logger.log(f"Initials metrics - valid_loss: {val_loss:.5f} | BCE: {bce_loss:.5f} | Dice: {dice_loss:.5f}")

        train_dataset_length = len(dataset_train)
        train_batches_count = len(train_loader)

        for i in range(self.start_epoch, epochs):
            patience_counter += 1

            gradient_norms = []
            epoch_loss = torch.zeros((), device=self.model.device)
            self.optimizer.zero_grad()
            for batch_num, example in enumerate(tqdm(train_loader, total=train_batches_count, desc=f"epoch {i+1}/{epochs}",unit="batch",)):
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
                    gradient_norms.append(grad_norm.item())

                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

            average_loss = (epoch_loss / train_dataset_length).item()

            val_loss, bce_loss, dice_loss = self.get_valid_loss(valid_loader)

            self.logger.log(f"epoch {i + 1}/{epochs}: loss: {average_loss:.3f} | max_grad_norm: {max(gradient_norms):.3f}")
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
                if not dice_loss is None:
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
