from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
from pydantic import ValidationError
import torch

from hatchfinder import AugmentationSettings, Config, DataSettings, SearchMaskGenerationSettings, Train
from hatchfinder.hatch_dataset import HatchDataset
from hatchfinder.search_windows import choose_window
from hatchfinder.train import UNSET


class SearchMaskGenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.size = (80, 48)
        drawing = Image.new("RGB", self.size)
        target = Image.new("L", self.size)
        target.paste(255, (16, 18, 62, 20))
        drawing.paste((255, 0, 0), (16, 18, 62, 20))
        drawing.save(self.root / "present.png")
        target.save(self.root / "full.png")
        Image.new("L", self.size).save(self.root / "empty.png")
        Image.new("RGB", self.size).save(self.root / "absent.png")
        Image.new("RGB", (8, 5), "gray").save(self.root / "legend.png")
        self.rows = [
            dict(drawing="present.png", hatch="legend.png", target="full.png"),
            dict(drawing="absent.png", hatch="legend.png", target="empty.png"),
        ]
        for split in ("train", "valid"):
            (self.root / split).mkdir()
            self.manifest(split, self.rows)

    def manifest(self, split, rows):
        (self.root / split / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def dataset(self, split="train", augment=False, augmentation=None, **overrides):
        settings = SearchMaskGenerationSettings(enabled=True, **overrides)
        return HatchDataset(self.root, split, DataSettings(search_mask_generation=settings),
                            augmentation or AugmentationSettings(enabled=False), augment=augment)

    def test_distribution_and_true_pixel_semantics(self):
        ds = self.dataset(min_margin_fraction=0.05, min_positive_pixels=12,
                          area_fraction_min=0.10, area_fraction_max=0.40)
        counts = Counter()
        for i in range(500):
            item = ds[i % len(ds)]
            kind = item["sample_kind"]
            counts[kind] += 1
            full = item["drawing"][0] > 0.5
            mask = item["search_mask"][0].bool()
            self.assertTrue(torch.equal(item["target"][0].bool(), full & mask))
            self.assertEqual(bool(full.any()), kind != "absent")
            n = int((full & mask).sum())
            if kind == "positive":
                self.assertGreaterEqual(n, 12)
            else:
                self.assertEqual(n, 0)
            yy, xx = mask.nonzero(as_tuple=True)
            w = int(xx.max()-xx.min()+1)
            h = int(yy.max()-yy.min()+1)
            self.assertEqual(int(mask.sum()), w*h)
            self.assertTrue(4 <= xx.min() and xx.max() < 76)
            self.assertTrue(3 <= yy.min() and yy.max() < 45)
            self.assertTrue(.10 <= w*h/(80*48) <= .40)
            self.assertTrue(.5 <= w/h <= 2)
        self.assertTrue(360 <= counts["positive"] <= 440, counts)
        for kind in ("absent", "outside"):
            self.assertTrue(25 <= counts[kind] <= 75, counts)

    def test_sampling_after_resize_and_full_tile_exception(self):
        aug = AugmentationSettings(rotate_90_probability=0, affine_probability=0, resize_probability=1,
                                   resize_scale_min=.5, resize_scale_max=.5)
        positive = self.dataset(augment=True, augmentation=aug, negative_fraction=0,
                                min_positive_pixels=8, full_tile_probability=1, min_margin_fraction=.2)
        result = positive[0]
        self.assertEqual(tuple(result["search_mask"].shape), (1, 24, 40))
        self.assertTrue(result["search_mask"].all())
        self.assertGreaterEqual(int(result["target"].sum()), 8)
        outside = self.dataset(negative_fraction=1, outside_window_share_of_negatives=1, full_tile_probability=1)
        result = outside[1]
        self.assertEqual(result["sample_kind"], "outside")
        self.assertFalse(result["search_mask"].all())
        self.assertFalse(result["target"].any())

    def test_fixed_validation_with_and_without_stored_mask(self):
        ds = self.dataset("valid", min_positive_pixels=5)
        for i in (1, 0, 1, 0):
            fresh = self.dataset("valid", min_positive_pixels=5)
            self.assertTrue(torch.equal(ds[i]["search_mask"], fresh[i]["search_mask"]))
            self.assertEqual(bool(ds[i]["target"].any()), i == 0)
        stored = Image.new("L", self.size)
        stored.paste(255, (1, 1, 8, 5))
        stored.save(self.root / "saved.png")
        self.manifest("valid", [dict(self.rows[0], target="empty.png", search_mask="saved.png")])
        result = self.dataset("valid")[0]
        self.assertEqual(int(result["search_mask"].sum()), 28)
        self.assertFalse(result["target"].any())

    def test_saved_training_rows_preserve_the_original_window_and_target(self):
        stored = Image.new("L", self.size)
        stored.paste(255, (10, 12, 32, 25))
        stored.save(self.root / "saved.png")
        target = Image.new("L", self.size)
        target.paste(255, (16, 18, 32, 20))
        target.save(self.root / "clipped.png")
        row = dict(self.rows[0], target="clipped.png", search_mask="saved.png")
        self.manifest("train", [row])
        legacy = HatchDataset(self.root, "train", DataSettings(), AugmentationSettings(enabled=False))
        saved = self.dataset()[0]
        expected = legacy[0]
        self.assertEqual(saved.keys(), expected.keys())
        self.assertNotIn("sample_kind", saved)
        for key in expected:
            self.assertTrue(torch.equal(saved[key], expected[key]), key)
        self.assertEqual(int(saved["search_mask"].sum()), 22 * 13)
        self.assertEqual(int(saved["target"].sum()), 16 * 2)

    def test_saved_rows_are_excluded_from_generation_pools_and_counts(self):
        stored = Image.new("L", self.size)
        stored.paste(255, (1, 1, 8, 5))
        stored.save(self.root / "saved.png")
        fixed = dict(self.rows[0], search_mask="saved.png", target="empty.png")
        self.manifest("train", [fixed, self.rows[0]])
        with self.assertRaisesRegex(ValueError, "absent"):
            self.dataset()
        self.manifest("train", [fixed, *self.rows])
        ds = self.dataset(negative_fraction=1, outside_window_share_of_negatives=0)
        stored, generated = ds[0], ds[1]
        self.assertTrue(stored["drawing"].any())
        self.assertFalse(generated["drawing"].any())
        self.assertEqual(generated["sample_kind"], "absent")
        self.assertEqual(ds[-1]["sample_kind"], "absent")
        for items in ([stored, generated], [generated, stored]):
            self.assertEqual(ds.collate_fn(items)["sample_kind"], ["absent"])

    def test_missing_search_mask_requires_generation(self):
        for split in ("train", "valid"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "Missing search_mask"):
                HatchDataset(self.root, split, DataSettings(), AugmentationSettings())

    def test_rare_feasible_validation_window_and_narrow_empty_window(self):
        target = Image.new("L", (100, 100))
        for point in ((4, 4), (95, 4), (4, 95), (95, 95)):
            target.putpixel(point, 255)
        target.save(self.root / "corners.png")
        row = dict(self.rows[0], drawing="corners.png", target="corners.png")
        self.manifest("valid", [row])
        ds = self.dataset("valid", min_positive_pixels=4, area_fraction_min=.05, area_fraction_max=.9)
        self.assertEqual(int(ds[0]["target"].sum()), 4)
        self.assertTrue(torch.equal(ds[0]["search_mask"], ds[0]["search_mask"]))

        target.paste(255, (0, 0, 100, 100))
        target.paste(0, (20, 30, 30, 40))
        settings = SearchMaskGenerationSettings(area_fraction_min=.01, area_fraction_max=.9)
        bounds = choose_window(target, "outside", settings, random.Random(3), (.9, .9),
                               False, exhaustive=True)
        self.assertEqual(bounds, (20, 30, 30, 40))

    def test_resuming_changes_training_windows_and_preserves_validation_windows(self):
        for split in ("train", "valid"):
            fresh = self.dataset(split, negative_fraction=0)
            resumed = self.dataset(split, negative_fraction=0)
            resumed.window_start_epoch = 1
            identical = all(torch.equal(fresh[0]["search_mask"], resumed[0]["search_mask"])
                            for _ in range(5))
            self.assertEqual(identical, split == "valid")

    def test_outside_resamples_another_pair_and_rejects_impossible_requests(self):
        Image.new("L", self.size, 255).save(self.root / "solid.png")
        solid = dict(self.rows[0], target="solid.png")
        self.manifest("train", [solid, self.rows[0]])
        settings = dict(negative_fraction=1, outside_window_share_of_negatives=1)
        result = self.dataset(**settings)[0]
        self.assertEqual(result["sample_kind"], "outside")
        self.assertFalse(result["target"].any())
        self.manifest("train", [solid])
        with self.assertRaisesRegex(ValueError, "outside"):
            self.dataset(**settings)[0]
        with self.assertRaisesRegex(ValueError, "absent"):
            self.dataset()
        self.manifest("train", [self.rows[1]])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.dataset(negative_fraction=0)
        ds = self.dataset(negative_fraction=1, outside_window_share_of_negatives=0)
        self.assertEqual(ds[0]["sample_kind"], "absent")

    def test_integer_dimension_constraints_and_config_roundtrip(self):
        config = SearchMaskGenerationSettings(enabled=True, area_fraction_min=.25, area_fraction_max=.25,
                                              aspect_ratio_min=1, aspect_ratio_max=1, min_positive_pixels=1)
        with Image.new("L", (80, 20)) as target:
            target.putpixel((0, 0), 255)
            bounds = choose_window(target, "positive", config, random.Random(1), (.5,.5), False)
            self.assertEqual(bounds, (0, 0, 20, 20))
        self.assertFalse(Config().data.search_mask_generation.enabled)
        cfg = Config(data=DataSettings(search_mask_generation=config))
        self.assertEqual(Config.model_validate_json(cfg.model_dump_json()), cfg)
        for bad in (dict(min_margin_fraction=.5), dict(aspect_ratio_min=3, aspect_ratio_max=2),
                    dict(area_fraction_min=.7, area_fraction_max=.6), dict(min_positive_pixels=0)):
            with self.assertRaises(ValidationError):
                SearchMaskGenerationSettings(**bad)

    def test_resume_inherits_saved_settings_and_applies_explicit_overrides(self):
        checkpoint = self.root / "last.pt"
        saved = Config(data=DataSettings(dataset=self.root, search_mask_generation={
            "enabled": True, "negative_fraction": .3, "min_positive_pixels": 7,
        }))
        torch.save({"config": saved.model_dump(mode="json")}, checkpoint)
        cfg = Config(training={"checkpoint_path": checkpoint}, data={"search_mask_generation": {
            "negative_fraction": .2,
        }})
        resumed, changed = Train._prepare_config(cfg, UNSET, self.root, UNSET, {})
        self.assertFalse(changed)
        settings = resumed.data.search_mask_generation
        self.assertTrue(settings.enabled)
        self.assertEqual((settings.negative_fraction, settings.min_positive_pixels), (.2, 7))
        inherited, changed = Train._prepare_config(Config(training={"checkpoint_path": checkpoint}),
                                                   UNSET, self.root, UNSET, {})
        self.assertFalse(changed)
        self.assertEqual(inherited.data.search_mask_generation, saved.data.search_mask_generation)
        cfg.data.search_mask_generation.min_positive_pixels = 11
        _, changed = Train._prepare_config(cfg, UNSET, self.root, UNSET, {})
        self.assertTrue(changed)
        Image.new("L", self.size, 255).save(self.root / "saved.png")
        self.manifest("valid", [dict(self.rows[1], search_mask="saved.png")])
        _, changed = Train._prepare_config(cfg, UNSET, self.root, UNSET, {})
        self.assertFalse(changed)
