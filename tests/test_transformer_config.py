import io
import unittest

import torch
from pydantic import ValidationError

from hatchfinder import HatchFinder
from hatchfinder.config import ModelSettings


class TransformerConfigTests(unittest.TestCase):
    def test_invalid_transformer_settings_are_rejected(self):
        cases = [
            {"level": -1},
            {"level": 3},
            {"level": 2, "num_head": 3},
            {"level": 2, "num_heads": 0},
            {"level": 2, "num_heads": 3},
            {"level": 2, "num_blocks": 0},
            {"level": 2, "downsample": 0},
            {"level": 2, "mlp_ratio": 0},
            {"level": 2, "mlp_ratio": 0.001},
            {"level": 2, "mlp_ratio": float("inf")},
            {"level": 2, "dropout": -0.1},
            {"level": 2, "dropout": 1.1},
            {"level": 2, "initial_gate": float("nan")},
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                ModelSettings(transformers=[settings])

    def test_duplicate_levels_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate transformer level"):
            ModelSettings(transformers=[{"level": 2}, {"level": 2}])

    def test_positional_encoding_channel_constraint(self):
        with self.assertRaisesRegex(ValidationError, "divisible by 4"):
            ModelSettings(
                drawing_channels=[32, 64, 126],
                group_norm_groups_drawings=2,
                transformers=[{"level": 2, "num_heads": 2}],
            )

    def test_valid_transformer_settings(self):
        config = ModelSettings(transformers=[{"level": 0}, {"level": 2}])
        self.assertEqual([t.level for t in config.transformers], [0, 2])


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.model = HatchFinder(device="cpu")

    def checkpoint_buffer(self, model_config):
        config = self.model.config.model_dump(mode="json")
        config["model"] = model_config
        buffer = io.BytesIO()
        torch.save(
            {
                "config": config,
                "model_state_dict": self.model.state_dict(),
                "epoch": 4,
                "best_metric": 0.25,
                "patience_counter": 2,
            },
            buffer,
        )
        buffer.seek(0)
        return buffer

    def test_old_checkpoint_without_transformers_resumes(self):
        config = self.model.config.model.model_dump(mode="json")
        del config["transformers"]
        with self.checkpoint_buffer(config) as buffer:
            result = self.model.load_checkpoint(None, None, buffer)
        self.assertEqual(result, (5, 0.25, 2))

    def test_current_checkpoint_resumes(self):
        config = self.model.config.model.model_dump(mode="json")
        with self.checkpoint_buffer(config) as buffer:
            result = self.model.load_checkpoint(None, None, buffer)
        self.assertEqual(result, (5, 0.25, 2))

    def test_different_architecture_is_still_rejected(self):
        config = self.model.config.model.model_dump(mode="json")
        config["transformers"] = [{"level": 2}]
        with self.checkpoint_buffer(config) as buffer:
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.model.load_checkpoint(None, None, buffer)


if __name__ == "__main__":
    unittest.main()
