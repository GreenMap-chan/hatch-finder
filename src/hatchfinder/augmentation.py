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

    def __call__(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        hatch: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
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

        drawing = self._augment_appearance(drawing)
        hatch = self._augment_appearance(hatch)
        target = self._clip_target(target, search_mask)

        return drawing, search_mask, hatch, target

    def _affine_drawing_group(
        self,
        drawing: Image.Image,
        search_mask: Image.Image,
        target: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        angle, translate, scale = self._affine_parameters(drawing.size)
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

    def _affine_image(self, image: Image.Image) -> Image.Image:
        angle, translate, scale = self._affine_parameters(image.size)
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
    ) -> tuple[float, list[int], float]:
        width, height = size
        max_dx = round(width * self.config.affine_translate_percent)
        max_dy = round(height * self.config.affine_translate_percent)
        angle = self.random.uniform(
            -self.config.affine_angle_degrees,
            self.config.affine_angle_degrees,
        )
        translate = [
            self.random.randint(-max_dx, max_dx),
            self.random.randint(-max_dy, max_dy),
        ]
        scale = self.random.uniform(
            self.config.affine_scale_min,
            self.config.affine_scale_max,
        )
        return angle, translate, scale

    def _augment_appearance(self, image: Image.Image) -> Image.Image:
        if self._happens(self.config.brightness_probability):
            image = TF.adjust_brightness(
                image,
                self.random.uniform(
                    self.config.brightness_min,
                    self.config.brightness_max,
                ),
            )
        if self._happens(self.config.contrast_probability):
            image = TF.adjust_contrast(
                image,
                self.random.uniform(
                    self.config.contrast_min,
                    self.config.contrast_max,
                ),
            )
        if self._happens(self.config.gamma_probability):
            image = TF.adjust_gamma(
                image,
                self.random.uniform(
                    self.config.gamma_min,
                    self.config.gamma_max,
                ),
            )
        if self._happens(self.config.blur_probability):
            image = TF.gaussian_blur(
                image,
                kernel_size=3,
                sigma=self.random.uniform(
                    self.config.blur_sigma_min,
                    self.config.blur_sigma_max,
                ),
            )
        if self._happens(self.config.noise_probability):
            tensor = TF.to_tensor(image)
            noise_std = self.random.uniform(
                self.config.noise_std_min,
                self.config.noise_std_max,
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
