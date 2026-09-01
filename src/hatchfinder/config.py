from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


PT_FORMAT_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AugmentationSettings(StrictModel):
    enabled: bool = True
    seed: int = 42
    max_retries: int = Field(default=5, ge=1)
    horizontal_flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    vertical_flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    drawing_rotate_90_probability: float = Field(default=0.75, ge=0.0, le=1.0)
    hatch_rotate_90_probability: float = Field(default=0.75, ge=0.0, le=1.0)
    affine_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    affine_angle_degrees: float = Field(default=2.0, ge=0.0)
    affine_translate_percent: float = Field(default=0.02, ge=0.0)
    affine_scale_min: float = Field(default=0.95, gt=0.0)
    affine_scale_max: float = Field(default=1.05, gt=0.0)
    brightness_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    brightness_min: float = Field(default=0.9, ge=0.0)
    brightness_max: float = Field(default=1.1, ge=0.0)
    contrast_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    contrast_min: float = Field(default=0.85, ge=0.0)
    contrast_max: float = Field(default=1.15, ge=0.0)
    gamma_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    gamma_min: float = Field(default=0.9, gt=0.0)
    gamma_max: float = Field(default=1.1, gt=0.0)
    blur_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    blur_sigma_min: float = Field(default=0.1, gt=0.0)
    blur_sigma_max: float = Field(default=0.8, gt=0.0)
    noise_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    noise_std_min: float = Field(default=0.002, ge=0.0)
    noise_std_max: float = Field(default=0.015, ge=0.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "AugmentationSettings":
        ranges = (
            ("affine_scale", self.affine_scale_min, self.affine_scale_max),
            ("brightness", self.brightness_min, self.brightness_max),
            ("contrast", self.contrast_min, self.contrast_max),
            ("gamma", self.gamma_min, self.gamma_max),
            ("blur_sigma", self.blur_sigma_min, self.blur_sigma_max),
            ("noise_std", self.noise_std_min, self.noise_std_max),
        )
        for name, minimum, maximum in ranges:
            if minimum > maximum:
                raise ValueError(f"{name}_min must not exceed {name}_max")
        return self


class ModelSettings(StrictModel):
    drawing_channels: list[int] = Field(default_factory=lambda: [32, 64, 128])
    hatch_channels: list[int] = Field(default_factory=lambda: [32, 64, 128])
    hatch_pool_sizes: list[int] = Field(default_factory=lambda: [2, 4, 6])
    match_dims: list[int] = Field(default_factory=lambda: [64, 128, 256])
    match_feature_dims: list[int] = Field(default_factory=lambda: [8, 16, 32])
    matcher_channels: list[int] = Field(default_factory=lambda: [256, 64])
    drawing_residual_blocks: list[int] = Field(default_factory=lambda: [1, 1, 1])
    hatch_residual_blocks: list[int] = Field(default_factory=lambda: [1, 1, 1])
    group_norm_groups_drawings: int = Field(default=8, gt=0)
    group_norm_groups_hatchings: int = Field(default=8, gt=0)
    kernel_size: int = Field(default=3, gt=0)
    downsample_stride: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_architecture(self) -> "ModelSettings":
        levels = len(self.match_dims)
        if levels == 0:
            raise ValueError("model must contain at least one matching level")
        fields = {
            "drawing_channels": self.drawing_channels,
            "hatch_channels": self.hatch_channels,
            "hatch_pool_sizes": self.hatch_pool_sizes,
            "match_feature_dims": self.match_feature_dims,
            "drawing_residual_blocks": self.drawing_residual_blocks,
            "hatch_residual_blocks": self.hatch_residual_blocks,
        }
        for name, values in fields.items():
            if len(values) != levels:
                raise ValueError(f"{name} must contain {levels} levels")
        if any(value <= 0 for value in self.drawing_channels + self.hatch_channels):
            raise ValueError("encoder channel counts must be positive")
        if any(value <= 0 for value in self.hatch_pool_sizes):
            raise ValueError("hatch_pool_sizes values must be positive")
        if any(value < 0 for value in self.match_dims + self.match_feature_dims):
            raise ValueError("matching dimensions must be non-negative")
        if any(value <= 0 for value in self.matcher_channels):
            raise ValueError("matcher_channels values must be positive")
        if any(value < 0 for value in self.drawing_residual_blocks + self.hatch_residual_blocks):
            raise ValueError("residual block counts must be non-negative")
        if any(m > 0 and f == 0 for m, f in zip(self.match_dims, self.match_feature_dims)):
            raise ValueError("enabled matching levels must have positive match_feature_dims")
        if self.match_dims[-1] == 0:
            raise ValueError("the deepest matching level must be enabled")
        if any(c % self.group_norm_groups_drawings for c in self.drawing_channels):
            raise ValueError("drawing channels must be divisible by their GroupNorm groups")
        if any(c % self.group_norm_groups_hatchings for c in self.hatch_channels):
            raise ValueError("hatch channels must be divisible by their GroupNorm groups")
        return self


class TrainingSettings(StrictModel):
    learning_rate: float = Field(default=0.00003, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    epochs: int = Field(default=150, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=16, gt=0)
    patience: int = Field(default=8, gt=0)
    num_workers: int = Field(default=4, ge=0)
    metric: Literal["val_loss"] = "val_loss"
    warmup_epochs: int = Field(default=0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    seed: int = 42
    bf16: bool = True
    eta_min: float = Field(default=1e-6, ge=0.0)
    decay_parameters: list[str] = Field(default_factory=lambda: [".weight"])
    load_model_path: Path | None = None
    checkpoint_path: Path | None = None

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingSettings":
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if self.eta_min > self.learning_rate:
            raise ValueError("eta_min must not exceed learning_rate")
        if not self.decay_parameters:
            raise ValueError("decay_parameters must not be empty")
        return self


class DataSettings(StrictModel):
    dataset: Path | None = None
    drawing_pad_multiple: int = Field(default=64, gt=0)
    load_dataset_examples: int = Field(default=16, gt=0)


class OutputSettings(StrictModel):
    directory: Path | None = None
    log_file_name: str = Field(default="log.txt", min_length=1)


class RuntimeSettings(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested in config, but it is not available")
        return torch.device(self.device)


class Config(StrictModel):
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    augmentation: AugmentationSettings = Field(default_factory=AugmentationSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


def config_from_pt_data(data: object) -> Config | None:
    if not isinstance(data, dict) or "config" not in data:
        return None

    format_version = data.get("format_version", 1)
    if format_version != PT_FORMAT_VERSION:
        raise ValueError(f"Unsupported .pt format version: {format_version}")

    return Config.model_validate(data["config"])


def model_config_from_pt_data(data: object) -> ModelSettings | None:
    if not isinstance(data, dict):
        return None
    if "model_config" in data:
        format_version = data.get("format_version", 1)
        if format_version != PT_FORMAT_VERSION:
            raise ValueError(f"Unsupported .pt format version: {format_version}")
        return ModelSettings.model_validate(data["model_config"])

    config = config_from_pt_data(data)
    return config.model if config is not None else None


def load_config(path: Path) -> Config:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    config = Config.model_validate(data)
    if config.data.dataset is not None:
        config.data.dataset = _resolve_path(config.data.dataset, path.parent)
    if config.output.directory is not None:
        config.output.directory = _resolve_path(config.output.directory, path.parent)
    if config.training.load_model_path is not None:
        config.training.load_model_path = _resolve_path(config.training.load_model_path, path.parent)
    if config.training.checkpoint_path is not None:
        config.training.checkpoint_path = _resolve_path(config.training.checkpoint_path, path.parent)
    return config


def save_config(config: Config, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
