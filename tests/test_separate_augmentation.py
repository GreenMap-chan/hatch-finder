import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from hatchfinder.augmentation import TrainingAugmentation
from hatchfinder.config import AugmentationSettings, Config


class SeparateAugmentationTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            horizontal_flip_probability=0, vertical_flip_probability=0,
            rotate_90_probability=0,
            affine_probability=0, brightness_probability=0,
            contrast_probability=0, gamma_probability=0,
            blur_probability=0, noise_probability=0,
        )
        values.update(overrides)
        return AugmentationSettings(**values)

    def inputs(self):
        pixels = np.full((16, 24, 3), 255, dtype=np.uint8)
        pixels[3:10, 4:12] = 0
        mask = np.zeros((16, 24), dtype=np.uint8)
        mask[2:14, 2:22] = 255
        target = np.zeros_like(mask)
        target[3:10, 4:12] = 255
        images = tuple(Image.fromarray(a) for a in (pixels, mask, pixels, target))
        for image in images:
            self.addCleanup(image.close)
        return images

    def run_aug(self, config, originals):
        images = TrainingAugmentation(config)(*originals)
        for image in images:
            self.addCleanup(image.close)
        return tuple(np.array(image) for image in images)

    def test_partial_overrides_inherit_and_survive_saved_config(self):
        config = self.config(
            brightness_min=0.7, affine_scale_max=2,
            rotate_90_probability=0.2,
            drawing={"noise_probability": 0.4},
            hatch={"affine_scale_min": 1.5, "rotate_90_probability": 0},
        )
        for saved in (config, Config.model_validate_json(
            Config(augmentation=config).model_dump_json()
        ).augmentation):
            drawing, hatch = saved.for_branch("drawing"), saved.for_branch("hatch")
            self.assertEqual(drawing.noise_probability, 0.4)
            self.assertEqual(hatch.noise_probability, 0)
            self.assertEqual(hatch.brightness_min, 0.7)
            self.assertEqual(hatch.affine_scale_min, 1.5)
            self.assertEqual(hatch.affine_scale_max, 2)
            self.assertEqual(drawing.rotate_90_probability, 0.2)
            self.assertEqual(hatch.rotate_90_probability, 0)

    def test_invalid_branch_settings_are_rejected_after_inheritance(self):
        for branch in ("drawing", "hatch"):
            for overrides in (
                {"affine_scale_min": 2}, {"noise_probability": 1.1},
                {"blur_kernel_size": 4}, {"rotate_90_probability": -1},
                {"affine_scale_max": 0}, {"typo": True},
            ):
                with self.subTest(branch=branch, overrides=overrides):
                    with self.assertRaises(ValidationError):
                        self.config(**{branch: overrides})

    def test_drawing_flip_keeps_masks_aligned_and_hatch_unchanged(self):
        originals = self.inputs()
        result = self.run_aug(self.config(drawing={"horizontal_flip_probability": 1}), originals)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(result[i], np.array(originals[i])[:, ::-1])
        np.testing.assert_array_equal(result[2], np.array(originals[2]))
        self.assertTrue(originals[0].getpixel((0, 0)))

    def test_hatch_flip_leaves_entire_drawing_group_unchanged(self):
        originals = self.inputs()
        result = self.run_aug(self.config(hatch={"vertical_flip_probability": 1}), originals)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(result[i], np.array(originals[i]))
        np.testing.assert_array_equal(result[2], np.array(originals[2])[::-1])

    def test_rotations_are_independent_and_masks_follow_drawing(self):
        for branch in ("drawing", "hatch"):
            originals = self.inputs()
            aug = TrainingAugmentation(self.config(**{branch: {"rotate_90_probability": 1}}))
            with patch.object(aug.random, "choice", return_value=90):
                result = aug(*originals)
            for i, image in enumerate(result):
                self.addCleanup(image.close)
                expected = np.array(originals[i])
                if (i == 2) == (branch == "hatch"):
                    expected = np.rot90(expected)
                np.testing.assert_array_equal(np.array(image), expected)

    def test_branch_affine_changes_only_selected_group(self):
        for branch in ("drawing", "hatch"):
            originals = self.inputs()
            config = self.config(**{branch: dict(
                affine_probability=1, affine_scale_min=0.5, affine_scale_max=0.5,
                affine_angle_degrees=0, affine_translate_percent=0,
            )})
            result = self.run_aug(config, originals)
            for i in range(4):
                if (i == 2) != (branch == "hatch"):
                    np.testing.assert_array_equal(result[i], np.array(originals[i]))
            changed = 0 if branch == "drawing" else 2
            self.assertLess((result[changed][..., 0] < 128).sum(),
                            (np.array(originals[changed])[..., 0] < 128).sum())
            self.assertTrue(np.all(result[3] <= result[1]))
            if branch == "drawing":
                np.testing.assert_array_equal(result[0][..., 0] < 128, result[3] > 0)

    def test_appearance_only_affects_selected_image(self):
        originals = self.inputs()
        result = self.run_aug(self.config(hatch={
            "brightness_probability": 1, "brightness_min": 0, "brightness_max": 0,
        }), originals)
        self.assertFalse(result[2].any())
        for i in (0, 1, 3):
            np.testing.assert_array_equal(result[i], np.array(originals[i]))

    def test_branch_and_global_disable(self):
        originals = self.inputs()
        config = self.config(horizontal_flip_probability=1, drawing={"enabled": False})
        result = self.run_aug(config, originals)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(result[i], np.array(originals[i]))
        np.testing.assert_array_equal(result[2], np.array(originals[2])[:, ::-1])
        result = self.run_aug(self.config(
            enabled=False, horizontal_flip_probability=1, hatch={"enabled": True},
        ), originals)
        for i in range(4):
            np.testing.assert_array_equal(result[i], np.array(originals[i]))

    def test_legacy_flips_remain_synchronized_after_serialization(self):
        originals = self.inputs()
        config = self.config(horizontal_flip_probability=1)
        config = AugmentationSettings.model_validate_json(config.model_dump_json())
        result = self.run_aug(config, originals)
        for i in range(4):
            np.testing.assert_array_equal(result[i], np.array(originals[i])[:, ::-1])

    def test_shared_flips_with_resize_blocks_and_null_overrides(self):
        originals = self.inputs()
        config = self.config(
            horizontal_flip_probability=0.5, vertical_flip_probability=0.5,
            drawing={"resize_probability": 0},
            hatch={"resize_probability": 0, "horizontal_flip_probability": None},
        )
        config = Config.model_validate_json(Config(augmentation=config).model_dump_json()).augmentation
        aug = TrainingAugmentation(config)
        seen = set()
        for _ in range(20):
            result = aug(*originals)
            for image in result:
                self.addCleanup(image.close)
            # These inputs have identical RGB images and target matching black pixels.
            np.testing.assert_array_equal(np.array(result[0]), np.array(result[2]))
            np.testing.assert_array_equal(np.array(result[0])[..., 0] == 0, np.array(result[3]) > 0)
            seen.add(result[0].tobytes())
        self.assertGreater(len(seen), 1)

    def test_explicit_equal_probabilities_still_sample_independently(self):
        originals = self.inputs()
        config = self.config(
            horizontal_flip_probability=0.5,
            drawing={"horizontal_flip_probability": 0.5},
            hatch={"horizontal_flip_probability": 0.5},
        )
        config = AugmentationSettings.model_validate_json(config.model_dump_json())
        aug = TrainingAugmentation(config)
        different = False
        for _ in range(20):
            result = aug(*originals)
            for image in result:
                self.addCleanup(image.close)
            different |= result[0].tobytes() != result[2].tobytes()
        self.assertTrue(different)

    def test_one_axis_override_does_not_disable_other_shared_axis(self):
        originals = self.inputs()
        result = self.run_aug(self.config(
            horizontal_flip_probability=1, vertical_flip_probability=1,
            hatch={"horizontal_flip_probability": 0},
        ), originals)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(result[i], np.array(originals[i])[::-1, ::-1])
        np.testing.assert_array_equal(result[2], np.array(originals[2])[::-1])

    def test_separate_augmentation_is_reproducible(self):
        originals = self.inputs()
        config = AugmentationSettings(seed=123, drawing={}, hatch={"noise_probability": 1})
        a, b = TrainingAugmentation(config), TrainingAugmentation(config)
        for _ in range(3):
            left, right = a(*originals), b(*originals)
            for x, y in zip(left, right):
                self.addCleanup(x.close)
                self.addCleanup(y.close)
                np.testing.assert_array_equal(np.array(x), np.array(y))


if __name__ == "__main__":
    unittest.main()
