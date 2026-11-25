import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from pydantic import ValidationError

from hatchfinder import AugmentationSettings, Config, DataSettings, HatchFinder, Train, load_config
from hatchfinder.hatch_dataset import HatchDataset
from hatchfinder.search_windows import choose_window
from hatchfinder.train import UNSET


class TrainingImageSizeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        size = (60, 32)
        Image.new("RGB", size).save(self.root / "drawing.png")
        Image.new("L", size, 255).save(self.root / "mask.png")
        target = Image.new("L", size)
        target.paste(255, (15, 5, 21, 15))
        target.save(self.root / "target.png")
        Image.new("L", size).save(self.root / "empty.png")
        Image.new("RGB", (12, 8), "red").save(self.root / "hatch.png")
        for split in ("train", "valid"):
            (self.root / split).mkdir()

    def dataset(self, split, data, augmentation=None, *, saved=True):
        row = dict(drawing="drawing.png", hatch="hatch.png", target="target.png")
        if saved:
            row["search_mask"] = "mask.png"
        (self.root / split / "manifest.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(dict(row, target="empty.png")), encoding="utf-8")
        return HatchDataset(self.root, split, data, augmentation or AugmentationSettings(enabled=False),
                            augment=True)

    def test_base_size_precedes_independent_resize_and_matches_inference_on_valid(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "examples/smoke_test.yaml")
        model = HatchFinder(cfg)
        aug = AugmentationSettings(horizontal_flip_probability=0, vertical_flip_probability=0,
                                   rotate_90_probability=0, affine_probability=0,
                                   drawing={"resize_probability": 1, "resize_scale_min": 1.5, "resize_scale_max": 1.5},
                                   hatch={"resize_probability": 1, "resize_scale_min": .5, "resize_scale_max": .5})
        for rect, shape in ((True, (24, 45)), (False, (40, 45))):
            data = DataSettings(imgsz=[40, 45], rect=rect)
            train = self.dataset("train", data, aug)[0]
            valid = self.dataset("valid", data, aug)[0]
            self.assertEqual(train["drawing"].shape[-2:], tuple(round(v * 1.5) for v in shape))
            self.assertEqual(train["target"].shape, train["search_mask"].shape)
            self.assertEqual(train["target"].shape[-2:], train["drawing"].shape[-2:])
            self.assertEqual(train["hatch"].shape[-2:], (3, 4))
            self.assertEqual(valid["drawing"].shape[-2:], shape)
            self.assertEqual(valid["hatch"].shape[-2:], (6, 9))
            with patch.object(model, "forward", side_effect=lambda drawing, mask, hatch: mask) as forward:
                model.infer(self.root / "drawing.png", self.root / "mask.png", self.root / "hatch.png",
                            imgsz=(40, 45), rect=rect)
                for key, tensor in zip(("drawing", "search_mask", "hatch"), forward.call_args.args):
                    self.assertTrue(tensor[0].equal(valid[key]), key)
            with Image.open(self.root / "target.png") as target:
                with target.resize((45, 24), Image.Resampling.NEAREST) as expected:
                    top = (shape[0] - 24) // 2
                    content = valid["target"][0, top:top + 24].mul(255).byte()
                    self.assertEqual(content.numpy().tobytes(), expected.tobytes())

    def test_generated_windows_ignore_transformed_padding_for_all_types(self):
        aug = AugmentationSettings(horizontal_flip_probability=0, vertical_flip_probability=0,
                                   rotate_90_probability=0, affine_probability=1, affine_angle_degrees=25,
                                   affine_translate_percent=0, affine_scale_min=1, affine_scale_max=1)
        for kind, negative, outside, full in (("positive", 0, 0, 0), ("outside", 1, 1, 0),
                                              ("absent", 1, 0, 0), ("positive", 0, 0, 1)):
            data = DataSettings(imgsz=60, rect=False, search_mask_generation=dict(
                enabled=True, negative_fraction=negative, outside_window_share_of_negatives=outside,
                min_positive_pixels=3, area_fraction_min=.04, area_fraction_max=.28,
                full_tile_probability=full))
            ds = self.dataset("train", data, aug, saved=False)
            for _ in range(8):
                sample = ds[0]
                self.assertEqual(sample["sample_kind"], kind)
                mask = sample["search_mask"][0].bool()
                self.assertTrue(mask.any())
                self.assertFalse((mask & (sample["drawing"][0] == 1)).any())
                self.assertEqual(bool(sample["target"].any()), kind == "positive")
                if full:
                    self.assertTrue(mask[sample["drawing"][0] == 0].all())
                else:
                    y, x = mask.nonzero(as_tuple=True)
                    self.assertEqual(mask.sum(), (y.max() - y.min() + 1) * (x.max() - x.min() + 1))

    def test_exhaustive_positive_respects_irregular_content(self):
        content = Image.new("L", (20, 20))
        content.paste(255, (5, 6, 13, 14))
        target = Image.new("L", content.size)
        target.putpixel((5, 6), 255)
        target.putpixel((12, 13), 255)
        cfg = DataSettings(search_mask_generation={"min_positive_pixels": 2, "area_fraction_min": .1,
                                                  "area_fraction_max": .8}).search_mask_generation
        self.assertEqual(choose_window(target, "positive", cfg, random.Random(3), (.5, .5),
                                       False, exhaustive=True, content=content), (5, 6, 13, 14))

    def test_size_validation_and_resume(self):
        for size in (0, -1, True, 3.5, [0, 4], [16], [True, 4]):
            with self.subTest(size=size), self.assertRaises(ValidationError):
                DataSettings(imgsz=size)
        checkpoint = self.root / "last.pt"
        cfg = Config(data={"dataset": self.root, "imgsz": 96, "rect": False})
        torch.save({"config": cfg.model_dump(mode="json")}, checkpoint)
        for overrides, expected_size, rect, changed in (
            ({}, 96, False, False), ({"imgsz": [96, 96]}, (96, 96), False, False),
            ({"rect": True}, 96, True, True), ({"imgsz": 80}, 80, False, True),
            ({"imgsz": None}, None, False, True),
        ):
            resumed, reset = Train._prepare_config(Config(training={"checkpoint_path": checkpoint},
                                                          data=overrides), UNSET, self.root, UNSET, {})
            self.assertEqual((resumed.data.imgsz, resumed.data.rect, reset), (expected_size, rect, changed))
