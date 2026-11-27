import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from hatchfinder import HatchFinder, load_config


class InferenceResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml"
        cls.model = HatchFinder(load_config(config))

    def test_landscape_tile_and_legend_resize_together_and_heatmap_aligns(self):
        with (Image.new("RGB", (14, 8)) as drawing, Image.new("L", drawing.size, 255) as mask,
              Image.new("RGB", (7, 5), "red") as hatch):
            drawing.paste("white", (8, 2, 13, 7))
            mask.paste(0, (0, 0, 3, 8))
            mask.putpixel((10, 5), 0)
            with patch.object(self.model, "forward", side_effect=lambda d, m, h: 8 * d[:, :1] - 4) as forward:
                rect_result = self.model.infer(drawing, mask, hatch, imgsz=8)
                fitted, fitted_mask, legend = forward.call_args.args
                self.assertEqual(fitted.shape[-2:], (5, 8))
                self.assertEqual(legend.shape[-2:], (3, 4))
                with mask.resize((8, 5), Image.Resampling.NEAREST) as resized_mask:
                    self.assertTrue(fitted_mask.equal(to_tensor(resized_mask).unsqueeze(0)))
                expected = F.interpolate((8 * fitted[:, :1] - 4).sigmoid() * fitted_mask,
                                         (8, 14), mode="bilinear", align_corners=False)
                expected *= to_tensor(mask).unsqueeze(0)
                torch.testing.assert_close(rect_result, expected)
                self.assertEqual(rect_result[0, 0, 5, 10].item(), 0)
                self.assertGreater(rect_result[0, 0, 4, 11].item(), .5)

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    drawing.save(root / "tile.png")
                    square_result = self.model.infer(root / "tile.png", mask, hatch, imgsz=8,
                                                     rect=False, debug_path=root)
                    with Image.open(root / "tile_debug.png") as overlay:
                        self.assertEqual(overlay.size, drawing.size)
                canvas, canvas_mask, square_legend = forward.call_args.args
                self.assertEqual(canvas.shape[-2:], (8, 8))
                self.assertTrue(canvas[..., 1:6, :].equal(fitted))
                self.assertTrue(canvas_mask[..., 1:6, :].equal(fitted_mask))
                for rows in (slice(0, 1), slice(6, 8)):
                    self.assertTrue((canvas[..., rows, :] == 1).all())
                    self.assertFalse(canvas_mask[..., rows, :].any())
                self.assertTrue(square_legend.equal(legend))
                torch.testing.assert_close(square_result, rect_result)

    def test_portrait_upscaling_and_fixed_rectangular_canvas(self):
        with (Image.new("RGB", (5, 9)) as drawing, Image.new("L", drawing.size, 255) as mask,
              Image.new("RGB", (3, 5)) as hatch):
            with patch.object(self.model, "forward", side_effect=lambda d, m, h: d[:, :1]) as forward:
                fitted = self.model.infer(drawing, mask, hatch, imgsz=(16, 14))
                self.assertEqual(forward.call_args.args[0].shape[-2:], (16, 9))
                restored = self.model.infer(drawing, mask, hatch, imgsz=(16, 14), rect=False)
                canvas, canvas_mask, legend = forward.call_args.args
                self.assertEqual(canvas.shape[-2:], (16, 14))
                self.assertEqual(legend.shape[-2:], (9, 5))
                self.assertTrue(canvas_mask[..., 2:11].all())
                self.assertFalse(canvas_mask[..., :2].any())
                self.assertFalse(canvas_mask[..., 11:].any())
                self.assertEqual(restored.shape, (1, 1, 9, 5))
                torch.testing.assert_close(restored, fitted)

    def test_precision_applies_to_forward_and_preserves_caller_autocast(self):
        dtypes = []
        conv = self.model.drawing_encoder.blocks[0][0]
        with (Image.new("RGB", (19, 13), "white") as drawing, Image.new("L", drawing.size, 255) as mask,
              Image.new("RGB", (7, 5), "gray") as hatch,
              conv.register_forward_hook(lambda module, args, result: dtypes.append(result.dtype))):
            mask.paste(0, (0, 0, 4, 13))
            options = dict(imgsz=24, rect=False)
            baseline = self.model.infer(drawing, mask, hatch, **options)
            mixed = self.model.infer(drawing, mask, hatch, precision="bf16", **options)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                full = self.model.infer(drawing, mask, hatch, precision="fp32", **options)
                self.model.infer(drawing, mask, hatch, **options)

        self.assertEqual(dtypes, [torch.float32, torch.bfloat16, torch.float32, torch.bfloat16])
        torch.testing.assert_close(full, baseline, atol=0, rtol=0)
        self.assertEqual(mixed.dtype, torch.float32)
        self.assertEqual(mixed.shape, baseline.shape)
        self.assertTrue(mixed.isfinite().all())
        self.assertFalse(mixed[..., :4].any())

    def test_native_size_and_invalid_arguments(self):
        with (Image.new("RGB", (19, 17)) as drawing, Image.new("L", drawing.size, 255) as mask,
              Image.new("RGB", (7, 5)) as hatch):
            native = self.model.infer(drawing, mask, hatch)
            for kwargs in ({"imgsz": None, "rect": False}, {"imgsz": 19}, {"imgsz": (17, 19), "rect": False}):
                torch.testing.assert_close(self.model.infer(drawing, mask, hatch, **kwargs), native, rtol=0, atol=0)
            for imgsz in (0, -1, True, 16.5, (0, 16), (16,), (16, 8, 4)):
                with self.subTest(imgsz=imgsz), self.assertRaisesRegex(ValueError, "imgsz"):
                    self.model.infer(drawing, mask, hatch, imgsz=imgsz)
            with self.assertRaisesRegex(ValueError, "rect"):
                self.model.infer(drawing, mask, hatch, imgsz=16, rect="false")
            with self.assertRaisesRegex(ValueError, "precision"):
                self.model.infer(drawing, mask, hatch, precision="fp8")
            with Image.new("L", (18, 17)) as wrong_mask, self.assertRaisesRegex(ValueError, "same spatial size"):
                self.model.infer(drawing, wrong_mask, hatch, imgsz=16)
