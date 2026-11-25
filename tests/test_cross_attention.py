import io
import unittest

import torch
from pydantic import ValidationError

from hatchfinder import Config, CrossAttentionSettings, HatchFinder, ModelSettings
from hatchfinder.cross_attention import CrossAttention


class CrossAttentionTests(unittest.TestCase):
    def config(self, enabled=True):
        return Config(model=ModelSettings(
            drawing_channels=[8, 16], hatch_channels=[16, 8],
            hatch_pool_sizes=[2, 3], match_dims=[8, 16],
            match_feature_dims=[4, 8], matcher_channels=[8],
            drawing_residual_blocks=[0, 0], hatch_residual_blocks=[0, 0],
            cross_attention=[{"level": 0, "num_heads": 2},
                             {"level": 1, "num_heads": 4, "downsample": 2}] if enabled else [],
        ))

    def inputs(self):
        return torch.rand(2, 3, 17, 23), torch.ones(2, 1, 17, 23), torch.rand(2, 3, 9, 13)

    def test_invalid_settings(self):
        for item in ({"level": -1}, {"level": 3}, {"level": 2, "num_heads": 3},
                     {"level": 2, "num_heads": 0}, {"level": 2, "downsample": 0},
                     {"level": 2, "dropout": 1.1}, {"level": 2, "initial_gate": float("nan")},
                     {"level": 2, "num_head": 8}):
            with self.subTest(item=item), self.assertRaises(ValidationError):
                ModelSettings(cross_attention=[item])
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            ModelSettings(cross_attention=[{"level": 2}, {"level": 2}])
        with self.assertRaisesRegex(ValidationError, "enabled matching"):
            ModelSettings(match_dims=[0, 128, 256], cross_attention=[{"level": 0}])

    def test_forward_gradients_and_zero_gate(self):
        torch.manual_seed(31)
        model = HatchFinder(self.config(), device="cpu")
        inputs = self.inputs()
        result = model(*inputs)
        self.assertEqual(result.shape, (2, 1, 17, 23))
        result.square().mean().backward()
        for branch in model.cross_attention.values():
            for parameter in branch.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(branch.attention.in_proj_weight.grad.abs().sum().item(), 0)
            self.assertGreater(branch.hatch_projection.weight.grad.abs().sum().item(), 0)
            with torch.no_grad():
                branch.gate.zero_()
        baseline = HatchFinder(self.config(False), device="cpu")
        baseline.load_state_dict({key: value for key, value in model.state_dict().items()
                                  if not key.startswith("cross_attention.")})
        with torch.inference_mode():
            torch.testing.assert_close(model(*inputs), baseline(*inputs), rtol=0, atol=0)

    def test_local_context_depends_on_hatch_and_handles_tiny_queries(self):
        torch.manual_seed(12)
        branch = CrossAttention(8, 8, 4, 2, CrossAttentionSettings(
            level=0, num_heads=2, downsample=4, initial_gate=1,
        )).eval()
        drawing, hatch = torch.rand(2, 8, 1, 3), torch.rand(2, 8, 2, 2)
        first = branch(drawing, hatch)
        second = branch(drawing, torch.rand_like(hatch))
        self.assertEqual(first.shape, (2, 4, 1, 3))
        self.assertFalse(torch.allclose(first, second))

    def test_model_roundtrip_and_legacy_formats(self):
        inputs = self.inputs()
        for enabled in (False, True):
            original = HatchFinder(self.config(enabled), device="cpu").eval()
            config = original.config.model.model_dump(mode="json")
            if not enabled:
                del config["cross_attention"]
                self.assertFalse(any(k.startswith("cross_attention.") for k in original.state_dict()))
            payloads = [
                {"format_version": 1, "model_config": config, "model_state_dict": original.state_dict()},
                {"format_version": 1, "config": {"model": config}, "model_state_dict": original.state_dict()},
                original.state_dict(),
            ]
            with torch.inference_mode():
                expected = original(*inputs)
            for index, payload in enumerate(payloads):
                with self.subTest(enabled=enabled, format=index), io.BytesIO() as buffer:
                    torch.save(payload, buffer)
                    buffer.seek(0)
                    restored = HatchFinder(
                        self.config(enabled) if index == 2 else None,
                        load_model_path=buffer, device="cpu",
                    ).eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(*inputs), expected, rtol=0, atol=0)
                    buffer.seek(0)
                    restored.load_model(buffer)
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(*inputs), expected, rtol=0, atol=0)

    def test_save_weights_restores_cross_attention(self):
        original = HatchFinder(self.config(), device="cpu").eval()
        with io.BytesIO() as buffer:
            original.save_weights(buffer)
            buffer.seek(0)
            restored = HatchFinder(load_model_path=buffer, device="cpu").eval()
        self.assertEqual(original.config.model, restored.config.model)
        inputs = self.inputs()
        with torch.inference_mode():
            torch.testing.assert_close(original(*inputs), restored(*inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
