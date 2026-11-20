from contextlib import ExitStack

from PIL import Image, ImageOps


def _scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = size
    return max(1, round(width * scale)), max(1, round(height * scale))


def canvas_size(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    if (not isinstance(imgsz, tuple) or len(imgsz) != 2
            or any(type(side) is not int or side <= 0 for side in imgsz)):
        raise ValueError("imgsz must be a positive integer or a (height, width) pair of positive integers")
    return imgsz


def fit_geometry(size: tuple[int, int], imgsz: int | tuple[int, int] | None, rect: bool):
    if not isinstance(rect, bool):
        raise ValueError("rect must be a boolean")
    if imgsz is None:
        return 1.0, (0, 0, 0, 0)
    target_h, target_w = canvas_size(imgsz)
    width, height = size
    scale = min(target_w / width, target_h / height)
    if rect:
        return scale, (0, 0, 0, 0)
    scaled_w, scaled_h = _scaled_size(size, scale)
    extra_w, extra_h = target_w - scaled_w, target_h - scaled_h
    left, top = extra_w // 2, extra_h // 2
    return scale, (left, extra_w - left, top, extra_h - top)


def resize_input(image: Image.Image, mode: str, scale: float, owned: ExitStack,
                 padding: tuple[int, int, int, int] = (0, 0, 0, 0)) -> Image.Image:
    size = _scaled_size(image.size, scale)
    if image.size != size:
        interpolation = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.BILINEAR
        with image.convert(mode) as converted:
            image = converted.resize(size, interpolation)
        owned.callback(image.close)
    if any(padding):
        left, right, top, bottom = padding
        border = (left, top, right, bottom)
        image = ImageOps.expand(image, border, fill="white" if mode == "RGB" else 0)
        owned.callback(image.close)
    return image
