
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from .hatch_encoder import HatchEncoder
from .drawing_encoder import DrawingEncoder
from .output_comparison import OutputComparison
from .heatmap_decoder import HeatmapDecoder
from .config import Config, model_config_from_pt_data
from . import checkpoint
from . import inference, losses

class HatchFinder(nn.Module):
    def __init__(
        self,
        config: Config | None = None,
        *,
        device: Literal["auto", "cpu", "cuda"] | None = None,
        load_model_path: Path | None = None,
        **model_overrides: Any,
    ) -> None:
        super().__init__()

        pt_data = None
        saved_model_config = None
        if load_model_path is not None:
            pt_data = torch.load(
                load_model_path,
                map_location="cpu",
                weights_only=True,
            )
            saved_model_config = model_config_from_pt_data(pt_data)

        if config is None:
            config = Config()
        else:
            config = config.model_copy(deep=True)

        if saved_model_config is not None:
            config.model = saved_model_config

        if model_overrides:
            config.model = type(config.model).model_validate({
                **config.model.model_dump(),
                **model_overrides,
            })
        if device is not None:
            config.runtime = type(config.runtime).model_validate({
                **config.runtime.model_dump(),
                "device": device,
            })

        self.config = config
        model_config = config.model

        self.hatch_encoder = HatchEncoder(model_config)
        self.drawing_encoder = DrawingEncoder(model_config)
        self.output_comparison = OutputComparison(model_config)
        self.heatmap_decoder = HeatmapDecoder(model_config)

        self.hatch_projections = nn.ModuleDict({
            str(level): nn.Linear(
                hatch_channels
                * hatch_pool_size
                * hatch_pool_size,
                match_dim,
            )
            for level, (hatch_channels, hatch_pool_size, match_dim) in enumerate(zip(
                model_config.hatch_channels,
                model_config.hatch_pool_sizes,
                model_config.match_dims,
            ))
            if match_dim > 0
        })

        self.drawing_projections = nn.ModuleDict({
            str(level): nn.Conv2d(
                channels,
                match_dim,
                kernel_size=1,
            )
            for level, (channels, match_dim) in enumerate(zip(
                model_config.drawing_channels,
                model_config.match_dims,
            ))
            if match_dim > 0
        })

        self.to(config.runtime.resolve_device())

        if pt_data is not None:
            self.load_state_dict(self._get_model_state_dict(pt_data))

    def get_model_size(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_inputs(
        self,
        drawing: torch.Tensor,
        mask: torch.Tensor,
        hatch: torch.Tensor,
    ) -> None:
        expected_channels = {"drawing": 3, "mask": 1, "hatch": 3}

        for name, tensor in (("drawing", drawing), ("mask", mask), ("hatch", hatch)):
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [batch, channels, height, width], "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != expected_channels[name]:
                raise ValueError(
                    f"{name} must have {expected_channels[name]} channels, "
                    f"got {tensor.shape[1]}"
                )

        if drawing.shape[0] != mask.shape[0] or drawing.shape[0] != hatch.shape[0]:
            raise ValueError(
                "drawing, mask and hatch must have the same batch size, got "
                f"{drawing.shape[0]}, {mask.shape[0]} and {hatch.shape[0]}"
            )

        if drawing.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "drawing and mask must have the same spatial size, got "
                f"{tuple(drawing.shape[-2:])} and {tuple(mask.shape[-2:])}"
            )

    def _image_to_tensor(self, image: Image.Image, mode: str) -> torch.Tensor:
        return inference.image_to_tensor(self, image, mode)

    def _get_vectors(
        self,
        drawing_features: list[torch.Tensor],
        hatch: torch.Tensor,
    ):
        hatch_features = self.hatch_encoder(hatch)

        drawing_vectors = [None] * len(drawing_features)
        hatch_vectors = [None] * len(drawing_features)

        for level, feature in enumerate(hatch_features):
            key = str(level)
            if key not in self.hatch_projections:
                continue

            pool_size = self.config.model.hatch_pool_sizes[level]
            pooled_feature = F.adaptive_avg_pool2d(
                feature,
                output_size=(pool_size, pool_size),
            )
            hatch_vectors[level] = self.hatch_projections[key](
                pooled_feature.flatten(1)
            )

        for level, features in enumerate(drawing_features):
            key = str(level)
            if key not in self.drawing_projections:
                continue

            drawing_vectors[level] = self.drawing_projections[key](features)

        return drawing_vectors, hatch_vectors

    def convert_images_to_tensors(self, drawing: Image.Image, mask: Image.Image, hatch: Image.Image):
        return inference.convert_images_to_tensors(self, drawing, mask, hatch)

    def forward(self, drawing: torch.Tensor, mask: torch.Tensor, hatch: torch.Tensor,):
        self._validate_inputs(drawing, mask, hatch)

        device = self.device
        drawing = drawing.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        hatch = hatch.to(device, non_blocking=True)

        drawing_features = self.drawing_encoder(drawing, mask)
        drawing_vectors, hatch_vectors = self._get_vectors(drawing_features, hatch)

        matching_map = self.output_comparison(
            drawing_vectors,
            hatch_vectors,
        )

        logits = self.heatmap_decoder(
            matching_map,
            drawing_features,
            hatch_vectors
        )

        return logits

    def infer(
        self,
        drawing: Image.Image | Path,
        mask: Image.Image | Path,
        hatch: Image.Image | Path,
        debug_path: Path | None = None,
        confidence: float | None = None,
    ):
        return inference.infer(self, drawing, mask, hatch, debug_path, confidence)

    @staticmethod
    def _save_inference_debug(
        drawing: torch.Tensor,
        mask: torch.Tensor,
        heatmap: torch.Tensor,
        debug_path: Path,
        drawing_name: str,
        confidence: float,
    ) -> None:
        inference.save_inference_debug(
            drawing, mask, heatmap, debug_path, drawing_name, confidence
        )

    def get_target_image_tensor(self, target: Image.Image):
        target_tensor = self._image_to_tensor(target, "L")
        target_tensor = (target_tensor > 0.5).float()
        return target_tensor

    def get_example_loss(self, drawing: torch.Tensor, mask: torch.Tensor, hatch: torch.Tensor, target_tensor: torch.Tensor):
        logits = self(drawing, mask, hatch)
        return self._get_loss(logits, target_tensor, mask)

    def _get_loss(
        self,
        logits: torch.Tensor,
        target_tensor: torch.Tensor,
        mask: torch.Tensor,
    ):
        return losses.get_loss(logits, target_tensor, mask, self._get_dice)

    def _get_dice(self, logits: torch.Tensor, target_tensor: torch.Tensor, mask: torch.Tensor):
        return losses.get_dice(logits, target_tensor, mask)

    def save_checkpoint(
        self,
        optimizer,
        scheduler,
        epoch: int,
        best_metric: float,
        patience_counter: int,
        path: Path = Path("runs/checkpoint.pt"),
    ):
        checkpoint.save_checkpoint(
            self, optimizer, scheduler, epoch, best_metric, patience_counter, path
        )

    def load_checkpoint(self, optimizer, scheduler, path: Path) -> tuple[int, float, int]:
        return checkpoint.load_checkpoint(self, optimizer, scheduler, path)

    def save_weights(self, path: Path):
        checkpoint.save_weights(self, path)

    def load_model(self, path: Path):
        checkpoint.load_model(self, path)

    @staticmethod
    def _get_model_state_dict(pt_data: object) -> dict[str, torch.Tensor]:
        return checkpoint.model_state_dict(pt_data)

    def clip_grad_norm(self):
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.config.training.max_grad_norm
        )

        return gradient_norm
