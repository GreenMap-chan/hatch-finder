import copy
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydantic import ValidationError
from torchvision.transforms import functional as TF

from hatchfinder.augmentation import TrainingAugmentation
from hatchfinder.config import AugmentationSettings, Config


class UnifiedAugmentationTests(unittest.TestCase):
    def config(self, **overrides):
        values = {name + "_probability": 0 for name in TrainingAugmentation.TRANSFORMS}
        return AugmentationSettings(**(values | overrides))

    def inputs(self, hatch_size=None):
        y, x = np.indices((48, 64))
        rgb = np.stack([(x*7) % 256, (y*11) % 256, ((x+y)*5) % 256], axis=-1).astype('uint8')
        images = (Image.fromarray(rgb), Image.new('L', (64, 48), 255),
                  Image.fromarray(rgb) if hatch_size is None else Image.new('RGB', hatch_size, 128),
                  Image.new('L', (64, 48), 255))
        for image in images:
            self.addCleanup(image.close)
        return images

    def run_aug(self, aug, inputs):
        output = aug(*inputs)
        for image in output:
            self.addCleanup(image.close)
        return output

    def test_every_shared_transform_uses_one_parameter_set(self):
        for name in TrainingAugmentation.TRANSFORMS:
            with self.subTest(transform=name):
                config = self.config(**{name + '_probability': 1}, drawing={}, hatch={})
                aug = TrainingAugmentation(config)
                inputs = self.inputs()
                with patch.object(aug, '_apply_transform', wraps=aug._apply_transform) as apply:
                    output = self.run_aug(aug, inputs)
                calls = [c for c in apply.call_args_list if c.args[0] == name]
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0].args[2], calls[1].args[2])
                if name == 'noise':
                    self.assertNotEqual(output[0].tobytes(), output[2].tobytes())
                else:
                    np.testing.assert_array_equal(np.array(output[0]), np.array(output[2]))
                if name in ('brightness', 'contrast', 'gamma', 'blur', 'noise', 'color_overlay'):
                    for i in (1, 3):
                        np.testing.assert_array_equal(np.array(output[i]), np.array(inputs[i]))

    def test_empty_and_null_branch_settings_do_not_change_results(self):
        inputs = self.inputs()
        configurations = [AugmentationSettings(seed=137), AugmentationSettings(seed=137, drawing={}, hatch={}),
                          AugmentationSettings(seed=137, hatch={'affine_angle_degrees': None,
                                                               'blur_sigma_max': None})]
        augmentations = [TrainingAugmentation(config) for config in configurations]
        for _ in range(5):
            results = [self.run_aug(aug, inputs) for aug in augmentations]
            for result in results[1:]:
                for original, actual in zip(results[0], result):
                    np.testing.assert_array_equal(np.array(original), np.array(actual))

    def test_each_family_override_splits_only_its_own_transform(self):
        fields = {
            'horizontal_flip': 'horizontal_flip_probability', 'vertical_flip': 'vertical_flip_probability',
            'rotate_90': 'rotate_90_probability', 'affine': 'affine_angle_degrees',
            'resize': 'resize_interpolation', 'brightness': 'brightness_min',
            'contrast': 'contrast_max', 'gamma': 'gamma_min', 'blur': 'blur_kernel_size',
            'noise': 'noise_std_max',
            'color_overlay': 'color_overlay_alpha_max',
        }
        for name, field in fields.items():
            # Even a value equal to the shared value is an explicit independent override.
            shared = AugmentationSettings()
            config = AugmentationSettings(hatch={field: getattr(shared, field)})
            config = Config.model_validate_json(Config(augmentation=config).model_dump_json()).augmentation
            aug = TrainingAugmentation(config)
            for candidate in fields:
                groups = [branches for branches, _ in aug._transform_groups(candidate)]
                self.assertEqual(groups, [('drawing',), ('hatch',)] if name == candidate else [('drawing', 'hatch')])

    def test_partial_override_samples_independently_and_keeps_other_parameters_shared(self):
        config = self.config(affine_probability=1, affine_angle_degrees=15,
                             brightness_probability=1, hatch={'affine_angle_degrees': 1})
        aug = TrainingAugmentation(config)
        with patch.object(aug, '_apply_transform', wraps=aug._apply_transform) as apply:
            self.run_aug(aug, self.inputs())
        affine = [c.args[2] for c in apply.call_args_list if c.args[0] == 'affine']
        brightness = [c.args[2] for c in apply.call_args_list if c.args[0] == 'brightness']
        self.assertEqual(len(affine), 2)
        self.assertNotEqual(affine[0], affine[1])
        self.assertLessEqual(abs(affine[1]['angle']), 1)
        self.assertEqual(brightness[0], brightness[1])
        self.assertEqual(aug.hatch_config.affine_scale_min, config.affine_scale_min)

    def test_shared_resize_considers_smaller_input_and_uses_same_scale(self):
        config = self.config(resize_probability=1, resize_scale_min=.3,
                             resize_scale_max=2, resize_min_side=18)
        aug = TrainingAugmentation(config)
        with patch.object(aug.random, 'uniform', wraps=aug.random.uniform) as sample:
            output = self.run_aug(aug, self.inputs(hatch_size=(100, 20)))
        sample.assert_called_once_with(.9, 2)
        self.assertGreaterEqual(min(output[2].size), 18)
        self.assertAlmostEqual(output[0].width/64, output[2].width/100, delta=.02)
        self.assertEqual(output[0].size, output[1].size)
        self.assertEqual(output[0].size, output[3].size)

    def test_impossible_shared_resize_skips_both_inputs(self):
        config = self.config(resize_probability=1, resize_scale_min=.3,
                             resize_scale_max=2, resize_min_side=18)
        inputs = self.inputs(hatch_size=(20, 4))
        output = self.run_aug(TrainingAugmentation(config), inputs)
        for original, actual in zip(inputs, output):
            np.testing.assert_array_equal(np.array(original), np.array(actual))

    def test_disabled_branch_does_not_constrain_shared_resize(self):
        config = self.config(resize_probability=1, resize_scale_min=.3,
                             resize_scale_max=2, resize_min_side=18, hatch={'enabled': False})
        inputs = self.inputs(hatch_size=(20, 4))
        output = self.run_aug(TrainingAugmentation(config), inputs)
        self.assertNotEqual(output[0].size, inputs[0].size)
        np.testing.assert_array_equal(np.array(output[2]), np.array(inputs[2]))

    def test_affine_translation_is_relative_and_masks_follow_drawing(self):
        config = self.config(affine_probability=1, affine_translate_percent=.2)
        aug = TrainingAugmentation(config)
        with patch.object(TF, 'affine', wraps=TF.affine) as affine:
            self.run_aug(aug, self.inputs(hatch_size=(128, 96)))
        calls = affine.call_args_list
        self.assertEqual(len(calls), 4)
        for c in calls[1:3]:
            self.assertEqual(c.args[1:4], calls[0].args[1:4])
        self.assertEqual(calls[0].args[1], calls[3].args[1])
        self.assertEqual(calls[0].args[3], calls[3].args[3])
        for drawing_shift, hatch_shift in zip(calls[0].args[2], calls[3].args[2]):
            self.assertLessEqual(abs(hatch_shift - 2*drawing_shift), 1)

    def test_old_rotation_names_migrate_without_mutating_input(self):
        raw = {'drawing_rotate_90_probability': .25, 'hatch_rotate_90_probability': .5,
               'hatch': {'noise_probability': 0}}
        unchanged = copy.deepcopy(raw)
        config = AugmentationSettings.model_validate(raw)
        self.assertEqual(raw, unchanged)
        self.assertEqual(config.drawing.rotate_90_probability, .25)
        self.assertEqual(config.hatch.rotate_90_probability, .5)
        saved = config.model_dump()
        self.assertNotIn('drawing_rotate_90_probability', saved)
        self.assertNotIn('hatch_rotate_90_probability', saved)
        self.assertEqual(AugmentationSettings.model_validate(saved), config)
        self.assertTrue(config.independent_transform('hatch', 'rotate_90'))

    def test_color_overlay_is_uniform_across_different_sizes_and_preserves_masks(self):
        config = self.config(color_overlay_probability=1, color_overlay_alpha_min=.25,
                             color_overlay_alpha_max=.25)
        aug = TrainingAugmentation(config)
        inputs = self.inputs(hatch_size=(20, 8))
        with patch.object(aug.random, 'randint', side_effect=[80, 120, 240]) as color:
            output = self.run_aug(aug, inputs)
            self.assertEqual(color.call_count, 3)
        for i in (0, 2):
            expected = np.floor(np.array(inputs[i])*.75 + np.array([80, 120, 240])*.25).astype('uint8')
            np.testing.assert_array_equal(np.array(output[i]), expected)
            self.assertEqual(output[i].size, inputs[i].size)
        for i in (1, 3):
            np.testing.assert_array_equal(np.array(output[i]), np.array(inputs[i]))

    def test_color_overlay_branch_override_inherits_range_and_samples_independently(self):
        config = self.config(color_overlay_probability=1, color_overlay_alpha_min=.25,
                             color_overlay_alpha_max=.25, hatch={'color_overlay_probability': 1})
        config = Config.model_validate_json(Config(augmentation=config).model_dump_json()).augmentation
        aug = TrainingAugmentation(config)
        inputs = self.inputs()
        with patch.object(aug.random, 'randint', side_effect=[255, 0, 0, 0, 0, 255]):
            output = self.run_aug(aug, inputs)
        for i, color in ((0, [255, 0, 0]), (2, [0, 0, 255])):
            expected = np.floor(np.array(inputs[i])*.75 + np.array(color)*.25).astype('uint8')
            np.testing.assert_array_equal(np.array(output[i]), expected)

    def test_color_overlay_seed_and_disabled_behavior(self):
        inputs = self.inputs()
        config = self.config(color_overlay_probability=1)
        left, right = TrainingAugmentation(config), TrainingAugmentation(config)
        seen = set()
        for _ in range(5):
            a, b = self.run_aug(left, inputs), self.run_aug(right, inputs)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(np.array(x), np.array(y))
            seen.add(a[0].tobytes())
        self.assertGreater(len(seen), 1)
        for extra in ({'color_overlay_probability': 0}, {'enabled': False},
                      {'drawing': {'enabled': False}, 'hatch': {'enabled': False}}):
            aug = TrainingAugmentation(self.config(**({'color_overlay_probability': 1} | extra)))
            state = aug.random.getstate()
            output = self.run_aug(aug, inputs)
            self.assertEqual(state, aug.random.getstate())
            for x, y in zip(inputs, output):
                np.testing.assert_array_equal(np.array(x), np.array(y))

    def test_invalid_color_overlay_settings_rejected(self):
        for invalid in ({'color_overlay_probability': 1.1}, {'color_overlay_alpha_min': -.1},
                        {'color_overlay_alpha_max': 1.1}, {'color_overlay_alpha_min': .2},
                        {'color_overlay_alpha_max': float('nan')}):
            for raw in (invalid, {'drawing': invalid}, {'hatch': invalid}):
                with self.subTest(raw=raw), self.assertRaises(ValidationError):
                    self.config(**raw)

    def test_conflicting_or_invalid_legacy_rotation_names_are_rejected(self):
        for raw in (
            {'drawing_rotate_90_probability': 2},
            {'hatch_rotate_90_probability': -.5},
            {'hatch_rotate_90_probability': .5, 'hatch': {'rotate_90_probability': .7}},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                AugmentationSettings.model_validate(raw)


if __name__ == '__main__':
    unittest.main()
