
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from pathlib import Path

from .hatch_encoder import HatchEncoder
from .drawing_encoder import DrawingEncoder
from .output_comparison import OutputComparison
from .heatmap_decoder import HeatmapDecoder
from .config import Config, PT_FORMAT_VERSION, model_config_from_pt_data

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

    def _image_to_tensor(
        self,
        image: Image.Image,
        mode: str,
    ) -> torch.Tensor:
        image = image.convert(mode)

        tensor = to_tensor(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

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
        drawing_tensor = self._image_to_tensor(drawing, "RGB")
        mask_tensor = self._image_to_tensor(mask, "L")
        mask_tensor = (mask_tensor > 0.5).float()
        hatch_tensor = self._image_to_tensor(hatch, "RGB")

        return drawing_tensor, mask_tensor, hatch_tensor

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
        self.eval()

        drawing_name = drawing.stem if isinstance(drawing, Path) else "inference"

        if isinstance(drawing, Path):
            with Image.open(drawing) as image:
                drawing = image.convert("RGB")
        if isinstance(mask, Path):
            with Image.open(mask) as image:
                mask = image.convert("L")
        if isinstance(hatch, Path):
            with Image.open(hatch) as image:
                hatch = image.convert("RGB")

        drawing_tensor, mask_tensor, hatch_tensor = self.convert_images_to_tensors(drawing, mask, hatch)

        with torch.no_grad():
            logits_matrix = self(drawing_tensor, mask_tensor, hatch_tensor)
            
            heatmap = torch.sigmoid(logits_matrix)
            heatmap = heatmap * mask_tensor

        if debug_path is not None:
            confidence = 0.5 if confidence is None else confidence
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            self._save_inference_debug(
                drawing_tensor,
                mask_tensor,
                heatmap,
                debug_path,
                drawing_name,
                confidence,
            )

        return heatmap

    @staticmethod
    def _save_inference_debug(
        drawing: torch.Tensor,
        mask: torch.Tensor,
        heatmap: torch.Tensor,
        debug_path: Path,
        drawing_name: str,
        confidence: float,
    ) -> None:
        drawing_image = to_pil_image(drawing[0].detach().cpu()).convert("RGBA")
        mask_cpu = mask[0].detach().cpu().clamp(0.0, 1.0)
        heatmap_cpu = heatmap[0].detach().cpu().clamp(0.0, 1.0)

        outside_mask = to_pil_image(1.0 - mask_cpu).convert("L")
        outside_overlay = Image.new("RGBA", drawing_image.size, (0, 0, 0, 140))
        outside_overlay.putalpha(
            outside_mask.point(lambda value: round(value * 140 / 255))
        )
        debug_image = Image.alpha_composite(drawing_image, outside_overlay)

        if confidence < 1.0:
            confidence_alpha = (
                (heatmap_cpu - confidence) / (1.0 - confidence)
            ).clamp(0.0, 1.0)
        else:
            confidence_alpha = (heatmap_cpu >= 1.0).float()
        confidence_alpha = confidence_alpha * mask_cpu
        confidence_alpha = confidence_alpha * 175 + (confidence_alpha > 0).float() * 80

        prediction_overlay = Image.new("RGBA", drawing_image.size, (255, 0, 0, 0))
        prediction_overlay.putalpha(
            to_pil_image((confidence_alpha / 255.0).clamp(0.0, 1.0)).convert("L")
        )
        debug_image = Image.alpha_composite(debug_image, prediction_overlay)

        debug_path.mkdir(parents=True, exist_ok=True)
        result_image = debug_image.convert("RGB")
        result_image.save(debug_path / f"{drawing_name}_debug.png")

        drawing_image.close()
        outside_mask.close()
        outside_overlay.close()
        prediction_overlay.close()
        debug_image.close()
        result_image.close()

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
        if logits.shape != target_tensor.shape or logits.shape != mask.shape:
            raise ValueError(
                "logits, target_tensor and mask must have identical shapes, got "
                f"{tuple(logits.shape)}, {tuple(target_tensor.shape)} and "
                f"{tuple(mask.shape)}"
            )

        target_tensor = target_tensor.to(logits.device, non_blocking=True)
        mask = mask.to(logits.device, non_blocking=True)

        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            target_tensor,
            reduction="none",
        )

        bce_loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)

        dice_loss = self._get_dice(logits, target_tensor, mask)

        if dice_loss is None:
            result_loss = bce_loss
        else:
            result_loss = bce_loss + dice_loss

        return result_loss, bce_loss, dice_loss

    def _get_dice(self, logits: torch.Tensor, target_tensor: torch.Tensor, mask: torch.Tensor):
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities * mask
        target = target_tensor * mask

        target_sum = target.sum(dim=(1, 2, 3))

        # В batch оставляем только примеры,
        # где target не пустой
        non_empty = target_sum > 0

        if not non_empty.any():
            return None

        probabilities = probabilities[non_empty]
        target = target[non_empty]

        intersection = (probabilities * target).sum(dim=(1, 2, 3))

        smooth = 1e-6

        dice = (
            2 * intersection + smooth
        ) / (
            probabilities.sum(dim=(1, 2, 3))
            + target.sum(dim=(1, 2, 3))
            + smooth
        )

        dice_loss = 1 - dice.mean()

        return dice_loss

    def save_checkpoint(
        self,
        optimizer,
        scheduler,
        epoch: int,
        best_metric: float,
        patience_counter: int,
        path: Path = Path("runs/checkpoint.pt"),
    ):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "patience_counter": patience_counter,
                "config": self.config.model_dump(mode="json"),
                "format_version": PT_FORMAT_VERSION,
            },
            path,
        )

    def load_checkpoint(self, optimizer, scheduler, path: Path) -> tuple[int, float, int]:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
    
            checkpoint_config = checkpoint.get("config")
            if checkpoint_config is not None:
                checkpoint_model_config = checkpoint_config.get("model")
                current_model_config = self.config.model.model_dump(mode="json")
                if checkpoint_model_config != current_model_config:
                    raise ValueError(
                        "Checkpoint model configuration does not match the current config"
                    )

            self.load_state_dict(
                checkpoint["model_state_dict"]
            )
    
            if optimizer:
                optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )

            if scheduler:
                scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )
    
            start_epoch = checkpoint["epoch"] + 1
    
            return (
                start_epoch,
                checkpoint["best_metric"],
                checkpoint["patience_counter"],
            )

    def save_weights(self, path: Path):
        torch.save(
            {
                "format_version": PT_FORMAT_VERSION,
                "model_config": self.config.model.model_dump(mode="json"),
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    def load_model(self, path: Path):
        pt_data = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        saved_model_config = model_config_from_pt_data(pt_data)
        if (
            saved_model_config is not None
            and saved_model_config != self.config.model
        ):
            raise ValueError(
                "Weights model configuration does not match the current model"
            )

        self.load_state_dict(self._get_model_state_dict(pt_data))

    @staticmethod
    def _get_model_state_dict(pt_data: object) -> dict[str, torch.Tensor]:
        if isinstance(pt_data, dict) and "model_state_dict" in pt_data:
            return pt_data["model_state_dict"]
        if isinstance(pt_data, dict):
            return pt_data
        raise ValueError("The .pt file does not contain a model state_dict")

    def clip_grad_norm(self):
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.config.training.max_grad_norm
        )

        return gradient_norm
