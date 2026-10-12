import io
import unittest

import torch
from pydantic import ValidationError

from hatchfinder import HatchFinder
from hatchfinder.config import Config, ModelSettings


class LegacyGateConfigTests(unittest.TestCase):
    def legacy_config(self, levels):
        return {
            "drawing_channels": [8] * levels,
            "hatch_channels": [8] * levels,
            "hatch_pool_sizes": [2] * levels,
            "match_dims": [8] * levels,
            "match_feature_dims": [8] * levels,
            "drawing_residual_blocks": [1] * levels,
            "hatch_residual_blocks": [1] * levels,
        }

    def test_missing_gates_follow_level_count_without_mutating_input(self):
        for levels in (1, 3, 4):
            with self.subTest(levels=levels):
                data = self.legacy_config(levels)
                settings = ModelSettings.model_validate(data)
                self.assertEqual(settings.match_gate_initial_values,
                                 [0.01] * (levels - 1) + [1.0])
                self.assertNotIn("match_gate_initial_values", data)
        self.assertEqual(ModelSettings().match_gate_initial_values, [0.01, 0.01, 1.0])

    def test_explicit_gates_are_preserved_and_invalid_lengths_rejected(self):
        data = self.legacy_config(4)
        settings = ModelSettings(**data, match_gate_initial_values=[0.3] * 4)
        self.assertEqual(settings.match_gate_initial_values, [0.3] * 4)
        for gates in ([], [0.01, 0.01, 1.0]):
            with self.subTest(gates=gates), self.assertRaises(ValidationError):
                ModelSettings(**data, match_gate_initial_values=gates)

    def test_legacy_weights_and_checkpoint_preserve_trained_gates_and_output(self):
        data = self.legacy_config(4)
        original = HatchFinder(Config(model=ModelSettings(**data)), device="cpu").eval()
        with torch.no_grad():
            for gate in original.heatmap_decoder.match_gates.values():
                gate.fill_(0.42)
        drawing = torch.rand(1, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        hatch = torch.rand(1, 3, 16, 16)
        with torch.inference_mode():
            expected = original(drawing, mask, hatch)
        for config_key, saved_config in (("model_config", data), ("config", {"model": data})):
            with self.subTest(format=config_key), io.BytesIO() as buffer:
                torch.save({
                    "format_version": 1,
                    config_key: saved_config,
                    "model_state_dict": original.state_dict(),
                    "epoch": 4,
                    "best_metric": 0.25,
                    "patience_counter": 2,
                }, buffer)
                buffer.seek(0)
                loaded = HatchFinder(load_model_path=buffer, device="cpu").eval()
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(drawing, mask, hatch), expected)
                if config_key == "config":
                    buffer.seek(0)
                    self.assertEqual(loaded.load_checkpoint(None, None, buffer), (5, 0.25, 2))


if __name__ == "__main__":
    unittest.main()
