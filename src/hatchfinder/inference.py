from pathlib import Path
from contextlib import ExitStack

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


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
) -> torch.Tensor:
    model.eval()
    drawing_name = drawing.stem if isinstance(drawing, Path) else "inference"
    with ExitStack() as owned_images:
        def open_if_path(value: Image.Image | Path, mode: str) -> Image.Image:
            if not isinstance(value, Path):
                return value
            with Image.open(value) as source:
                converted = source.convert(mode)
            owned_images.callback(converted.close)
            return converted

        drawing = open_if_path(drawing, "RGB")
        mask = open_if_path(mask, "L")
        hatch = open_if_path(hatch, "RGB")
        drawing_tensor, mask_tensor, hatch_tensor = model.convert_images_to_tensors(
            drawing, mask, hatch
        )
    with torch.no_grad():
        heatmap = torch.sigmoid(model(drawing_tensor, mask_tensor, hatch_tensor))
        heatmap = heatmap * mask_tensor

    if debug_path is not None:
        confidence = 0.5 if confidence is None else confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
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
