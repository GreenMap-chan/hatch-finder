import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError

from hatchfinder.augmentation import TrainingAugmentation
from hatchfinder.config import AugmentationSettings, Config, DataSettings
from hatchfinder.hatch_dataset import HatchDataset


class ResizeAugmentationTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            horizontal_flip_probability=0, vertical_flip_probability=0,
            drawing_rotate_90_probability=0, hatch_rotate_90_probability=0,
            affine_probability=0, brightness_probability=0, contrast_probability=0,
            gamma_probability=0, blur_probability=0, noise_probability=0,
        )
        return AugmentationSettings(**(values | overrides))

    def inputs(self):
        rgb = np.full((16, 24, 3), 255, dtype=np.uint8)
        rgb[3:10, 4:12] = 0
        target = np.zeros((16, 24), dtype=np.uint8)
        target[3:10, 4:12] = 255
        images = (Image.fromarray(rgb), Image.new("L", (24, 16), 255),
                  Image.fromarray(rgb), Image.fromarray(target))
        for image in images:
            self.addCleanup(image.close)
        return images

    def augment(self, config, inputs):
        result = TrainingAugmentation(config)(*inputs)
        for image in result:
            self.addCleanup(image.close)
        return result

    def test_nearest_resize_changes_canvas_without_cropping_and_keeps_masks_aligned(self):
        inputs = self.inputs()
        result = self.augment(self.config(
            drawing=dict(resize_probability=1, resize_scale_min=2, resize_scale_max=2),
            hatch=dict(resize_probability=1, resize_scale_min=0.5, resize_scale_max=0.5),
        ), inputs)
        for i in (0, 1, 3):
            self.assertEqual(result[i].size, (48, 32))
            expected = np.repeat(np.repeat(np.array(inputs[i]), 2, axis=0), 2, axis=1)
            np.testing.assert_array_equal(np.array(result[i]), expected)
        self.assertEqual(result[2].size, (12, 8))
        self.assertEqual(inputs[0].size, (24, 16))
        self.assertEqual(inputs[0].getpixel((0, 0)), (255, 255, 255))

    def test_hatch_only_resize_matches_service_rounding(self):
        inputs = list(self.inputs())
        inputs[2] = Image.fromarray(np.random.default_rng(0).integers(
            0, 256, (148, 38, 3), dtype=np.uint8))
        self.addCleanup(inputs[2].close)
        result = self.augment(self.config(hatch=dict(
            resize_probability=1, resize_scale_min=0.365, resize_scale_max=0.365,
            resize_interpolation="nearest",
        )), inputs)
        self.assertEqual(result[2].size, (14, 54))
        with inputs[2].resize((14, 54), Image.Resampling.NEAREST) as expected:
            np.testing.assert_array_equal(np.array(result[2]), np.array(expected))
        for i in (0, 1, 3):
            np.testing.assert_array_equal(np.array(result[i]), np.array(inputs[i]))

    def test_interpolation_changes_rgb_but_masks_stay_binary(self):
        inputs = self.inputs()
        for interpolation in ("bilinear", "bicubic", "lanczos"):
            with self.subTest(interpolation=interpolation):
                result = self.augment(self.config(drawing=dict(
                    resize_probability=1, resize_scale_min=2, resize_scale_max=2,
                    resize_interpolation=interpolation,
                )), inputs)
                rgb = np.array(result[0])
                self.assertTrue(((rgb > 0) & (rgb < 255)).any())
                for i in (1, 3):
                    expected = np.repeat(np.repeat(np.array(inputs[i]), 2, axis=0), 2, axis=1)
                    np.testing.assert_array_equal(np.array(result[i]), expected)

    def test_shared_resize_and_partial_overrides_survive_serialization(self):
        config = self.config(
            resize_probability=1, resize_scale_min=2, resize_scale_max=2,
            resize_interpolation="bicubic", hatch={"resize_probability": 0},
        )
        config = Config.model_validate_json(Config(augmentation=config).model_dump_json()).augmentation
        self.assertEqual(config.for_branch("hatch").resize_interpolation, "bicubic")
        result = self.augment(config, self.inputs())
        self.assertEqual(result[0].size, (48, 32))
        self.assertEqual(result[2].size, (24, 16))
        # The shared configuration also works without either branch mapping.
        result = self.augment(self.config(
            resize_probability=1, resize_scale_min=2, resize_scale_max=2,
        ), self.inputs())
        self.assertEqual([im.size for im in result], [(48, 32)] * 4)

    def test_disabled_resize_does_not_advance_random_stream(self):
        inputs = self.inputs()
        a = TrainingAugmentation(AugmentationSettings(seed=17))
        b = TrainingAugmentation(AugmentationSettings(
            seed=17, resize_probability=0, resize_scale_min=0.1, resize_scale_max=10,
        ))
        for _ in range(3):
            left, right = a(*inputs), b(*inputs)
            for x, y in zip(left, right):
                self.addCleanup(x.close)
                self.addCleanup(y.close)
                np.testing.assert_array_equal(np.array(x), np.array(y))
            self.assertEqual(a.random.getstate(), b.random.getstate())

    def test_global_and_branch_disable_prevent_resize(self):
        for overrides in ({"enabled": False}, {"drawing": {"enabled": False}, "hatch": {"enabled": False}}):
            result = self.augment(self.config(
                resize_probability=1, resize_scale_min=2, resize_scale_max=2, **overrides,
            ), self.inputs())
            self.assertEqual([im.size for im in result], [(24, 16)] * 4)

    def test_tiny_hatch_is_clamped_to_one_pixel(self):
        result = self.augment(self.config(hatch=dict(
            resize_probability=1, resize_scale_min=0.001, resize_scale_max=0.001,
        )), self.inputs())
        self.assertEqual(result[2].size, (1, 1))

    def test_disappearing_positive_target_falls_back_to_original(self):
        # The single center pixel misses this off-center target under nearest resize.
        inputs = self.inputs()
        result = self.augment(self.config(max_retries=2, drawing=dict(
            resize_probability=1, resize_scale_min=0.001, resize_scale_max=0.001,
        )), inputs)
        for before, after in zip(inputs, result):
            np.testing.assert_array_equal(np.array(before), np.array(after))

    def test_invalid_settings_rejected(self):
        for invalid in (
            {"resize_probability": 1.1}, {"resize_probability": -1},
            {"resize_scale_min": 0}, {"resize_scale_max": -1},
            {"resize_scale_min": 2, "resize_scale_max": 1},
            {"resize_scale_max": float("inf")}, {"resize_scale_min": float("nan")},
            {"resize_interpolation": "unknown"},
        ):
            for overrides in (invalid, {"hatch": invalid}, {"drawing": invalid}):
                with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                    self.config(**overrides)

    def test_dataset_batches_resized_images_and_skips_validation_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.inputs()
            names = ("drawing", "search_mask", "hatch", "target")
            manifest = []
            for index, scale in enumerate((1, 2)):
                row = {}
                for name, image in zip(names, inputs):
                    filename = f"{index}_{name}.png"
                    with image.resize((24 * scale, 16 * scale), Image.Resampling.NEAREST) as resized:
                        resized.save(root / filename)
                    row[name] = filename
                manifest.append(row)
            for split in ("train", "valid"):
                (root / split).mkdir()
                (root / split / "manifest.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in manifest), encoding="utf-8")
            config = self.config(
                drawing=dict(resize_probability=1, resize_scale_min=2, resize_scale_max=2),
                hatch=dict(resize_probability=1, resize_scale_min=0.5, resize_scale_max=0.5),
            )
            train = HatchDataset(root, "train", DataSettings(), config, augment=True)
            items = [train[0], train[1]]
            self.assertEqual(tuple(items[0]["drawing"].shape), (3, 32, 48))
            batch = train.collate_fn(items)
            self.assertEqual(tuple(batch["drawing"].shape), (2, 3, 64, 128))
            self.assertEqual(tuple(batch["hatch"].shape), (2, 3, 16, 24))
            self.assertEqual(tuple(batch["target"].shape), (2, 1, 64, 128))
            valid = HatchDataset(root, "valid", DataSettings(), config)
            self.assertEqual(tuple(valid[0]["drawing"].shape), (3, 16, 24))
            self.assertEqual(tuple(valid[0]["hatch"].shape), (3, 16, 24))


if __name__ == "__main__":
    unittest.main()
