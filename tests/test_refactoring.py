import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from hatchfinder import HatchFinder, Train, load_config
from hatchfinder.augmentation import TrainingAugmentation
from hatchfinder.config import AugmentationSettings


EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml"


class RefactoringTests(unittest.TestCase):
    def test_augmentation_keeps_input_images_open(self):
        config = AugmentationSettings(
            horizontal_flip_probability=0,
            vertical_flip_probability=0,
            drawing_rotate_90_probability=0,
            hatch_rotate_90_probability=0,
            affine_probability=0,
            brightness_probability=0,
            contrast_probability=0,
            gamma_probability=0,
            blur_probability=0,
            noise_probability=0,
        )
        originals = (
            Image.new("RGB", (16, 16), "white"),
            Image.new("L", (16, 16), "white"),
            Image.new("RGB", (16, 16), "black"),
            Image.new("L", (16, 16), "black"),
        )
        augmented = TrainingAugmentation(config)(*originals)
        try:
            self.assertEqual(originals[0].getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(augmented[0].getpixel((0, 0)), (255, 255, 255))
        finally:
            for image in originals + augmented:
                image.close()

    def test_inference_accepts_paths_and_keeps_caller_images_open(self):
        model = HatchFinder(load_config(EXAMPLE_CONFIG))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drawing = Image.new("RGB", (16, 16), "white")
            mask = Image.new("L", (16, 16), "white")
            hatch = Image.new("RGB", (16, 16), "black")
            try:
                result = model.infer(drawing, mask, hatch, debug_path=root)
                self.assertEqual(tuple(result.shape), (1, 1, 16, 16))
                self.assertTrue((root / "inference_debug.png").exists())
                self.assertEqual(drawing.getpixel((0, 0)), (255, 255, 255))

                for name, image in (("drawing", drawing), ("mask", mask), ("hatch", hatch)):
                    image.save(root / f"{name}.png")
                from_paths = model.infer(
                    root / "drawing.png", root / "mask.png", root / "hatch.png"
                )
                torch.testing.assert_close(result, from_paths)

                weights_path = root / "model.pt"
                model.save_weights(weights_path)
                restored = HatchFinder(load_model_path=weights_path, device="cpu")
                torch.testing.assert_close(result, restored.infer(drawing, mask, hatch))
            finally:
                drawing.close()
                mask.close()
                hatch.close()

    def test_extending_legacy_warmup_scheduler_preserves_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(EXAMPLE_CONFIG)
            config.output.directory = Path(directory) / "run"
            trainer = Train(config)
            old_scheduler = trainer.create_scheduler(total_steps=3, warmup_steps=1)
            for _ in range(2):
                trainer.optimizer.step()
                old_scheduler.step()
            old_state = old_scheduler.state_dict()
            optimizer_state = trainer.optimizer.state_dict()

            config.output.directory = Path(directory) / "resumed"
            resumed = Train(config)
            scheduler = resumed.create_scheduler(total_steps=5, warmup_steps=1)
            resumed.optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(old_state)
            resumed.extend_scheduler(total_steps=5, warmup_steps=1)

            self.assertEqual(resumed.cosine_scheduler.T_max, 4)
            self.assertEqual(resumed.cosine_scheduler.last_epoch, 1)
            expected = config.training.eta_min + (
                config.training.learning_rate - config.training.eta_min
            ) * (1 + math.cos(math.pi / 4)) / 2
            self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], expected)


if __name__ == "__main__":
    unittest.main()
