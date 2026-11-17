import random

import torch
from PIL import Image, ImageChops
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import AugmentationSettings


class TrainingAugmentation:
    TRANSFORMS = (
        "horizontal_flip", "vertical_flip", "rotate_90", "affine", "resize",
        "brightness", "contrast", "gamma", "blur", "noise",
    )

    def __init__(self, config: AugmentationSettings) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.torch_generator = torch.Generator(device="cpu")
        self.torch_generator.manual_seed(config.seed)
        self.drawing_config = config.for_branch("drawing")
        self.hatch_config = config.for_branch("hatch")

    def __call__(
        self, drawing: Image.Image, search_mask: Image.Image,
        hatch: Image.Image, target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        # Own all working images; never close or modify caller-owned inputs.
        originals = (drawing.convert("RGB"), search_mask.convert("L"),
                     hatch.convert("RGB"), target.convert("L"))
        try:
            if self.config.enabled:
                positive = originals[3].getbbox() is not None
                for _ in range(self.config.max_retries):
                    result = self._augment_once(*originals)
                    if result[1].getbbox() is not None and (
                        not positive or result[3].getbbox() is not None
                    ):
                        return result
                    self._close_images(result)
            return (originals[0].copy(), originals[1].copy(), originals[2].copy(),
                    self._clip_target(originals[3], originals[1]))
        finally:
            self._close_images(originals)

    def _transform_groups(self, transform: str):
        """Only overrides for THIS transform split a branch from the shared group."""
        shared = []
        independent = []
        for branch, config in (("drawing", self.drawing_config), ("hatch", self.hatch_config)):
            if not config.enabled:
                continue
            if self.config.independent_transform(branch, transform):
                independent.append(((branch,), config))
            else:
                shared.append(branch)
        if shared:
            yield tuple(shared), self.config
        yield from independent

    def _augment_once(
        self, drawing: Image.Image, search_mask: Image.Image,
        hatch: Image.Image, target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        groups = {
            "drawing": (drawing.copy(), search_mask.copy(), target.copy()),
            "hatch": (hatch.copy(),),
        }
        try:
            for transform in self.TRANSFORMS:
                for branches, config in self._transform_groups(transform):
                    sizes = [groups[branch][0].size for branch in branches]
                    parameters = self._sample_transform(transform, config, sizes)
                    if parameters is None:
                        continue
                    for branch in branches:
                        previous = groups[branch]
                        groups[branch] = self._apply_transform(transform, previous, parameters)
                        for old, new in zip(previous, groups[branch]):
                            if old is not new:
                                old.close()
            drawing, search_mask, target = groups["drawing"]
            clipped = self._clip_target(target, search_mask)
            target.close()
            return drawing, search_mask, groups["hatch"][0], clipped
        except Exception:
            for images in groups.values():
                self._close_images(images)
            raise

    def _sample_transform(self, transform: str, config: AugmentationSettings,
                          sizes: list[tuple[int, int]]) -> dict | None:
        probability = getattr(config, transform + "_probability")
        if probability == 0 or not self._happens(probability):
            return None
        if transform in ("horizontal_flip", "vertical_flip"):
            return {}
        if transform == "rotate_90":
            return {"angle": self.random.choice((90, 180, 270))}
        if transform == "affine":
            return {
                "angle": self.random.uniform(-config.affine_angle_degrees, config.affine_angle_degrees),
                "translate_fraction": [self.random.uniform(-config.affine_translate_percent,
                                                            config.affine_translate_percent) for _ in range(2)],
                "scale": self.random.uniform(config.affine_scale_min, config.affine_scale_max),
            }
        if transform == "resize":
            minimum_scale = config.resize_scale_min
            if config.resize_min_side > 1:
                minimum_scale = max(minimum_scale, *(config.resize_min_side/min(size) for size in sizes))
            if minimum_scale > config.resize_scale_max:
                return None
            scale = self.random.uniform(minimum_scale, config.resize_scale_max)
            interpolation = config.resize_interpolation
            if interpolation == "random":
                interpolation = self.random.choice(("nearest", "bilinear", "bicubic", "lanczos"))
            return {"scale": scale, "interpolation": interpolation}
        if transform in ("brightness", "contrast", "gamma"):
            return {"factor": self.random.uniform(getattr(config, transform + "_min"),
                                                   getattr(config, transform + "_max"))}
        if transform == "blur":
            return {"kernel_size": config.blur_kernel_size,
                    "sigma": self.random.uniform(config.blur_sigma_min, config.blur_sigma_max)}
        if transform == "noise":
            return {"sigma": self.random.uniform(config.noise_std_min, config.noise_std_max)}
        raise ValueError(f"Unknown augmentation: {transform}")

    def _apply_transform(self, transform: str, images: tuple[Image.Image, ...],
                         parameters: dict) -> tuple[Image.Image, ...]:
        """The first image is RGB; the remaining images are aligned binary masks."""
        if transform in ("horizontal_flip", "vertical_flip"):
            flip = TF.hflip if transform == "horizontal_flip" else TF.vflip
            return tuple(flip(image) for image in images)
        if transform == "rotate_90":
            return tuple(TF.rotate(image, parameters["angle"], expand=True, fill=255 if i == 0 else 0)
                         for i, image in enumerate(images))
        if transform == "affine":
            width, height = images[0].size
            translate = [round(dimension*fraction) for dimension, fraction in
                         zip((width, height), parameters["translate_fraction"])]
            return tuple(TF.affine(
                image, parameters["angle"], translate, parameters["scale"], shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR if i == 0 else InterpolationMode.NEAREST,
                fill=255 if i == 0 else 0,
            ) for i, image in enumerate(images))
        if transform == "resize":
            size = tuple(max(1, round(dimension*parameters["scale"])) for dimension in images[0].size)
            interpolation = {
                "nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS,
            }[parameters["interpolation"]]
            return tuple(image.resize(size, interpolation if i == 0 else Image.Resampling.NEAREST)
                         for i, image in enumerate(images))
        image = images[0]
        if transform == "brightness":
            image = TF.adjust_brightness(image, parameters["factor"])
        elif transform == "contrast":
            image = TF.adjust_contrast(image, parameters["factor"])
        elif transform == "gamma":
            image = TF.adjust_gamma(image, parameters["factor"])
        elif transform == "blur":
            image = TF.gaussian_blur(image, kernel_size=parameters["kernel_size"], sigma=parameters["sigma"])
        elif transform == "noise":
            tensor = TF.to_tensor(image)
            noise = torch.randn(tensor.shape, generator=self.torch_generator, dtype=tensor.dtype)
            image = TF.to_pil_image((tensor + noise*parameters["sigma"]).clamp(0.0, 1.0))
        else:
            raise ValueError(f"Unknown augmentation: {transform}")
        return (image, *images[1:])

    @staticmethod
    def _clip_target(target: Image.Image, search_mask: Image.Image) -> Image.Image:
        with target.point(lambda value: 255 if value > 127 else 0) as binary_target:
            with search_mask.point(lambda value: 255 if value > 127 else 0) as binary_mask:
                return ImageChops.multiply(binary_target, binary_mask)

    def _happens(self, probability: float) -> bool:
        return self.random.random() < probability

    @staticmethod
    def _close_images(images: tuple[Image.Image, ...]) -> None:
        for image in images:
            image.close()
