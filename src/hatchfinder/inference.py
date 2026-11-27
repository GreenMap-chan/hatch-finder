from pathlib import Path
from collections import OrderedDict
from collections.abc import Hashable
from contextlib import ExitStack, nullcontext
from threading import Lock
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from .image_geometry import fit_geometry, resize_input


InferencePrecision = Literal["fp32", "fp16", "bf16"]


class FeatureCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = Lock()

    def __reduce__(self):
        # Copies and serialized models start with an empty inference cache.
        return type(self), (self.capacity,)

    def get_or_compute(self, key, encode, *inputs):
        if key is None:
            return encode(*inputs)
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key]
            if len(self.entries) == self.capacity:
                self.entries.popitem(last=False)
        features = encode(*inputs)
        with self.lock:
            self.entries[key] = features
            self.entries.move_to_end(key)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return features


def _feature_context(device: torch.device):
    # Inspect the active scope so an external autocast works just like precision=.
    if hasattr(torch, "get_autocast_dtype"):
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
    elif device.type == "cpu":  # PyTorch < 2.4
        dtype = torch.get_autocast_cpu_dtype() if torch.is_autocast_cpu_enabled() else None
    else:
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
    return device, dtype, stream


def _autocast_context(device: torch.device, precision: InferencePrecision | None):
    if precision is None:
        return nullcontext()
    if precision == "fp32":
        return torch.autocast(device.type, enabled=False)
    if precision not in ("fp16", "bf16"):
        raise ValueError("precision must be 'fp32', 'fp16', 'bf16', or None")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    # Autocast checks BF16 support on the currently selected CUDA device.
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        return torch.autocast(device.type, dtype=dtype)


def image_to_tensor(model, image: Image.Image, mode: str) -> torch.Tensor:
    with image.convert(mode) as converted:
        return to_tensor(converted).unsqueeze(0).to(model.device)


def convert_images_to_tensors(model, drawing: Image.Image, mask: Image.Image, hatch: Image.Image):
    drawing_tensor = model._image_to_tensor(drawing, "RGB")
    mask_tensor = (model._image_to_tensor(mask, "L") > 0.5).float()
    hatch_tensor = model._image_to_tensor(hatch, "RGB")
    return drawing_tensor, mask_tensor, hatch_tensor


def infer(
    model,
    drawing: Image.Image | Path,
    mask: Image.Image | Path,
    hatch: Image.Image | Path,
    debug_path: Path | None = None,
    confidence: float | None = None,
    *,
    imgsz: int | tuple[int, int] | None = None,
    rect: bool = True,
    precision: InferencePrecision | None = None,
    drawing_key: Hashable | None = None,
    hatch_key: Hashable | None = None,
) -> torch.Tensor:
    autocast = _autocast_context(model.device, precision)
    model.eval()
    drawing_name = drawing.stem if isinstance(drawing, Path) else "inference"
    with ExitStack() as owned_images, torch.inference_mode():
        def open_if_path(value: Image.Image | Path) -> Image.Image:
            return owned_images.enter_context(Image.open(value)) if isinstance(value, Path) else value

        drawing = open_if_path(drawing)
        mask = open_if_path(mask)
        hatch = open_if_path(hatch)
        if mask.size != drawing.size:
            raise ValueError("Drawing and mask must have the same spatial size")
        if min(*drawing.size, *hatch.size) < 1:
            raise ValueError("Drawing and hatch dimensions must be positive")
        scale, padding = fit_geometry(drawing.size, imgsz, rect)
        scaled_drawing = resize_input(drawing, "RGB", scale, owned_images)
        drawing_tensor, mask_tensor, hatch_tensor = model.convert_images_to_tensors(
            scaled_drawing, resize_input(mask, "L", scale, owned_images),
            resize_input(hatch, "RGB", scale, owned_images),
        )
        model_drawing, model_mask = drawing_tensor, mask_tensor
        if any(padding):
            model_drawing = F.pad(drawing_tensor, padding, value=1.0)
            model_mask = F.pad(mask_tensor, padding, value=0.0)
        with autocast:
            if drawing_key is None and hatch_key is None:
                logits = model(model_drawing, model_mask, hatch_tensor)
            else:
                context = _feature_context(model_drawing.device)
                if drawing_key is not None:
                    drawing_key = (drawing_key, drawing.size, scaled_drawing.size, padding, context)
                if hatch_key is not None:
                    hatch_key = (hatch_key, hatch.size, hatch_tensor.shape[-2:], context)
                logits = model(model_drawing, model_mask, hatch_tensor, _feature_keys=(drawing_key, hatch_key))
        if any(padding):
            left, _, top, _ = padding
            logits = logits[..., top:top + scaled_drawing.height, left:left + scaled_drawing.width]
        if precision is not None:
            logits = logits.float()
        heatmap = torch.sigmoid(logits) * mask_tensor
        if scaled_drawing.size != drawing.size:
            heatmap = F.interpolate(heatmap, size=(drawing.height, drawing.width),
                                    mode="bilinear", align_corners=False)
            mask_tensor = (model._image_to_tensor(mask, "L") > 0.5).float()
            heatmap.mul_(mask_tensor)

        if debug_path is not None:
            confidence = 0.5 if confidence is None else confidence
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            if scaled_drawing.size != drawing.size:
                with drawing.convert("RGB") as original:
                    drawing_tensor = to_tensor(original).unsqueeze(0)
            model._save_inference_debug(
                drawing_tensor, mask_tensor, heatmap, debug_path, drawing_name, confidence
            )
    return heatmap


def save_inference_debug(
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
        confidence_alpha = ((heatmap_cpu - confidence) / (1.0 - confidence)).clamp(0.0, 1.0)
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
