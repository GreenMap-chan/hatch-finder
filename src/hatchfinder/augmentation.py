import random

import torch
from PIL import Image, ImageChops
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import AugmentationSettings


class TrainingAugmentation:
    def __init__(self, config: AugmentationSettings) -> None:
        self.config = config
        self.random = random.Random(config.seed)
        self.torch_generator = torch.Generator(device="cpu")
        self.torch_generator.manual_seed(config.seed)
        self.separate_branches = config.drawing is not None or config.hatch is not None
        self.drawing_config = config.for_branch("drawing")
        self.hatch_config = config.for_branch("hatch")

    def __call__(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        hatch: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        # Work on owned copies; callers keep ownership of the input images.
        drawing = drawing.convert("RGB")
        search_mask = search_mask.convert("L")
        hatch = hatch.convert("RGB")
        target = target.convert("L")

        if not self.config.enabled:
            clipped_target = self._clip_target(target, search_mask)
            target.close()
            return drawing, search_mask, hatch, clipped_target

        positive_example = target.getbbox() is not None
        for _ in range(max(1, self.config.max_retries)):
            result = self._augment_once(drawing, search_mask, hatch, target)
            result_mask = result[1]
            result_target = result[3]

            if result_mask.getbbox() is None:
                self._close_images(result)
                continue
            if positive_example and result_target.getbbox() is None:
                self._close_images(result)
                continue

            self._close_images((drawing, search_mask, hatch, target))
            return result

        clipped_target = self._clip_target(target, search_mask)
        target.close()
        return drawing, search_mask, hatch, clipped_target

    def _augment_once(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        hatch: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        if self.separate_branches:
            return self._augment_separate(drawing, search_mask, hatch, target)

        # Preserve the legacy random draw order and synchronized flips.
        drawing = drawing.copy()
        search_mask = search_mask.copy()
        hatch = hatch.copy()
        target = target.copy()

        if self._happens(self.config.horizontal_flip_probability):
            drawing = TF.hflip(drawing)
            search_mask = TF.hflip(search_mask)
            hatch = TF.hflip(hatch)
            target = TF.hflip(target)

        if self._happens(self.config.vertical_flip_probability):
            drawing = TF.vflip(drawing)
            search_mask = TF.vflip(search_mask)
            hatch = TF.vflip(hatch)
            target = TF.vflip(target)

        if self._happens(self.config.drawing_rotate_90_probability):
            drawing_angle = self.random.choice((90, 180, 270))
            drawing = TF.rotate(drawing, drawing_angle, expand=True, fill=255)
            search_mask = TF.rotate(
                search_mask,
                drawing_angle,
                expand=True,
                fill=0,
            )
            target = TF.rotate(target, drawing_angle, expand=True, fill=0)

        if self._happens(self.config.hatch_rotate_90_probability):
            hatch_angle = self.random.choice((90, 180, 270))
            hatch = TF.rotate(hatch, hatch_angle, expand=True, fill=255)

        if self._happens(self.config.affine_probability):
            drawing, search_mask, target = self._affine_drawing_group(
                drawing,
                search_mask,
                target,
            )
            hatch = self._affine_image(hatch)

        drawing, search_mask, target = self._resize_group(
            (drawing, search_mask, target), self.config,
        )
        hatch, = self._resize_group((hatch,), self.config)
        drawing = self._augment_appearance(drawing)
        hatch = self._augment_appearance(hatch)
        target = self._clip_target(target, search_mask)

        return drawing, search_mask, hatch, target

    def _augment_separate(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        hatch: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        drawing, search_mask, hatch, target = (
            image.copy() for image in (drawing, search_mask, hatch, target)
        )
        dc, hc = self.drawing_config, self.hatch_config
        if dc.enabled:
            if self._happens(dc.horizontal_flip_probability):
                drawing, search_mask, target = (
                    TF.hflip(image) for image in (drawing, search_mask, target)
                )
            if self._happens(dc.vertical_flip_probability):
                drawing, search_mask, target = (
                    TF.vflip(image) for image in (drawing, search_mask, target)
                )
            if self._happens(dc.drawing_rotate_90_probability):
                angle = self.random.choice((90, 180, 270))
                drawing = TF.rotate(drawing, angle, expand=True, fill=255)
                search_mask = TF.rotate(search_mask, angle, expand=True, fill=0)
                target = TF.rotate(target, angle, expand=True, fill=0)
            if self._happens(dc.affine_probability):
                drawing, search_mask, target = self._affine_drawing_group(
                    drawing, search_mask, target, dc,
                )
            drawing, search_mask, target = self._resize_group(
                (drawing, search_mask, target), dc,
            )
            drawing = self._augment_appearance(drawing, dc)

        if hc.enabled:
            if self._happens(hc.horizontal_flip_probability):
                hatch = TF.hflip(hatch)
            if self._happens(hc.vertical_flip_probability):
                hatch = TF.vflip(hatch)
            if self._happens(hc.hatch_rotate_90_probability):
                hatch = TF.rotate(
                    hatch, self.random.choice((90, 180, 270)), expand=True, fill=255,
                )
            if self._happens(hc.affine_probability):
                hatch = self._affine_image(hatch, hc)
            hatch, = self._resize_group((hatch,), hc)
            hatch = self._augment_appearance(hatch, hc)

        return drawing, search_mask, hatch, self._clip_target(target, search_mask)

    def _resize_group(
        self,
        images: tuple[Image.Image, ...],
        config: AugmentationSettings,
    ) -> tuple[Image.Image, ...]:
        """Resize RGB followed by optional aligned masks, without cropping/padding."""
        # Do not consume RNG when disabled: old configurations keep their results.
        if config.resize_probability == 0 or not self._happens(config.resize_probability):
            return images
        scale = self.random.uniform(config.resize_scale_min, config.resize_scale_max)
        size = tuple(max(1, round(dimension * scale)) for dimension in images[0].size)
        interpolation = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[config.resize_interpolation]
        return tuple(
            image.resize(size, interpolation if i == 0 else Image.Resampling.NEAREST)
            for i, image in enumerate(images)
        )

    def _affine_drawing_group(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        target: Image.Image,
        config: AugmentationSettings | None = None,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        angle, translate, scale = self._affine_parameters(drawing.size, config)
        drawing = TF.affine(
            drawing,
            angle,
            translate,
            scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=255,
        )
        search_mask = TF.affine(
            search_mask,
            angle,
            translate,
            scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        target = TF.affine(
            target,
            angle,
            translate,
            scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return drawing, search_mask, target

    def _affine_image(
        self, image: Image.Image, config: AugmentationSettings | None = None,
    ) -> Image.Image:
        angle, translate, scale = self._affine_parameters(image.size, config)
        return TF.affine(
            image,
            angle,
            translate,
            scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=255,
        )

    def _affine_parameters(
        self,
        size: tuple[int, int],
        config: AugmentationSettings | None = None,
    ) -> tuple[float, list[int], float]:
        config = config if config is not None else self.config
        width, height = size
        max_dx = round(width * config.affine_translate_percent)
        max_dy = round(height * config.affine_translate_percent)
        angle = self.random.uniform(
            -config.affine_angle_degrees,
            config.affine_angle_degrees,
        )
        translate = [
            self.random.randint(-max_dx, max_dx),
            self.random.randint(-max_dy, max_dy),
        ]
        scale = self.random.uniform(
            config.affine_scale_min,
            config.affine_scale_max,
        )
        return angle, translate, scale

    def _augment_appearance(
        self, image: Image.Image, config: AugmentationSettings | None = None,
    ) -> Image.Image:
        config = config if config is not None else self.config
        if self._happens(config.brightness_probability):
            image = TF.adjust_brightness(
                image,
                self.random.uniform(
                    config.brightness_min,
                    config.brightness_max,
                ),
            )
        if self._happens(config.contrast_probability):
            image = TF.adjust_contrast(
                image,
                self.random.uniform(
                    config.contrast_min,
                    config.contrast_max,
                ),
            )
        if self._happens(config.gamma_probability):
            image = TF.adjust_gamma(
                image,
                self.random.uniform(
                    config.gamma_min,
                    config.gamma_max,
                ),
            )
        if self._happens(config.blur_probability):
            image = TF.gaussian_blur(
                image,
                kernel_size=config.blur_kernel_size,
                sigma=self.random.uniform(
                    config.blur_sigma_min,
                    config.blur_sigma_max,
                ),
            )
        if self._happens(config.noise_probability):
            tensor = TF.to_tensor(image)
            noise_std = self.random.uniform(
                config.noise_std_min,
                config.noise_std_max,
            )
            noise = torch.randn(
                tensor.shape,
                generator=self.torch_generator,
                dtype=tensor.dtype,
            )
            image = TF.to_pil_image((tensor + noise * noise_std).clamp(0.0, 1.0))
        return image

    @staticmethod
    def _clip_target(target: Image.Image, search_mask: Image.Image) -> Image.Image:
        binary_target = target.point(lambda value: 255 if value > 127 else 0)
        binary_mask = search_mask.point(lambda value: 255 if value > 127 else 0)
        return ImageChops.multiply(binary_target, binary_mask)

    def _happens(self, probability: float) -> bool:
        return self.random.random() < probability

    @staticmethod
    def _close_images(images: tuple[Image.Image, ...]) -> None:
        for image in images:
            image.close()
