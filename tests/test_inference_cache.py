import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from hatchfinder import CrossAttentionSettings, HatchFinder, load_config


class InferenceCacheTests(unittest.TestCase):
    def setUp(self):
        config = load_config(Path(__file__).resolve().parents[1] / "examples/smoke_test.yaml")
        config.model.pixel_correlation.enabled = True
        config.model.cross_attention = [CrossAttentionSettings(level=2, num_heads=2)]
        self.model = HatchFinder(config, device="cpu")
        self.drawing = Image.new("RGB", (24, 18), "white")
        self.drawing.paste("black", (5, 3, 17, 11))
        self.mask = Image.new("L", self.drawing.size, 255)
        self.hatch = Image.new("RGB", (7, 9), "gray")
        self.inputs = (self.drawing, self.mask, self.hatch)

    def test_branch_hits_and_outputs_with_new_tiles_masks_and_legends(self):
        tile = self.drawing.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        hatch = self.hatch.copy()
        hatch.paste("black", (2, 0, 5, 9))
        mask = self.mask.copy()
        mask.paste(0, (0, 0, 10, 18))
        cases = [
            (self.inputs, (0, 0), (1, 1)),
            ((self.drawing, self.mask, hatch), (0, 1), (1, 2)),
            ((tile, self.mask, self.hatch), (1, 0), (2, 2)),
            (self.inputs, (0, 0), (3, 2)),  # Only the most recent tile is retained.
            ((self.drawing, mask, self.hatch), ((0, "inset"), 0), (4, 2)),
            (self.inputs, (None, 0), (5, 2)),
            (self.inputs, (0, None), (6, 3)),
            (self.inputs, (0, None), (6, 4)),
            (self.inputs, (None, None), (7, 5)),
        ]
        expected = [self.model.infer(*args) for args, _, _ in cases]
        with (patch.object(self.model.drawing_encoder, "forward", wraps=self.model.drawing_encoder.forward) as d,
              patch.object(self.model.hatch_encoder, "forward", wraps=self.model.hatch_encoder.forward) as h):
            for (args, (dk, hk), counts), reference in zip(cases, expected):
                actual = self.model.infer(*args, drawing_key=dk, hatch_key=hk)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertEqual((d.call_count, h.call_count), counts)

    def test_geometry_and_effective_autocast(self):
        keys = dict(drawing_key="tile", hatch_key="legend")
        with (patch.object(self.model.drawing_encoder, "forward", wraps=self.model.drawing_encoder.forward) as d,
              patch.object(self.model.hatch_encoder, "forward", wraps=self.model.hatch_encoder.forward) as h):
            self.model.infer(*self.inputs, **keys)
            self.model.infer(*self.inputs, **keys, imgsz=24, precision="fp32")
            self.model.infer(*self.inputs, **keys, imgsz=24, rect=False)
            self.assertEqual((d.call_count, h.call_count), (2, 1))
            self.model.infer(*self.inputs, **keys, imgsz=48, rect=False)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.model.infer(*self.inputs, **keys, imgsz=48, rect=False)
            result = self.model.infer(*self.inputs, **keys, imgsz=48, rect=False, precision="bf16")
            self.assertEqual((d.call_count, h.call_count), (4, 3))
        reference = self.model.infer(*self.inputs, imgsz=48, rect=False, precision="bf16")
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_lru_and_cache_lifetime(self):
        def predict(key):
            return self.model.infer(*self.inputs, hatch_key=key, drawing_key=0)

        with (patch.object(self.model.hatch_encoder, "forward", wraps=self.model.hatch_encoder.forward) as h,
              patch.object(self.model.drawing_encoder, "forward", wraps=self.model.drawing_encoder.forward) as d):
            for key in range(16):
                predict(key)
            for key in (0, 16, 0, 1):
                predict(key)
            self.assertEqual((h.call_count, d.call_count), (18, 1))
            for reset in (self.model.clear_inference_cache, self.model.train, self.model.cpu,
                          lambda: self.model.load_state_dict(self.model.state_dict())):
                before = h.call_count, d.call_count
                reset()
                predict(1)
                self.assertEqual((h.call_count, d.call_count), (before[0] + 1, before[1] + 1))
