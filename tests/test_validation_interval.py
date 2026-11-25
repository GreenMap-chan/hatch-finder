import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from pydantic import ValidationError

from hatchfinder import Config, Train, load_config
from hatchfinder.train import _intermediate_validation_batches


class ValidationIntervalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        Image.new("RGB", (16, 16), "white").save(self.root / "tile.png")
        for split, count in (("train", 7), ("valid", 1)):
            (self.root / split).mkdir()
            row = dict.fromkeys(("drawing", "hatch", "search_mask", "target"), "tile.png")
            (self.root / split / "manifest.jsonl").write_text(
                "".join(json.dumps(dict(row, id=str(i))) + "\n" for i in range(count)), encoding="utf-8"
            )
        self.config = load_config(Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml")
        self.config.output.directory = self.root / "run"
        self.config.data.dataset = self.root
        self.config.training.gradient_accumulation_steps = 2
        self.config.augmentation.enabled = False

    def test_schedule(self):
        self.assertEqual(Config().training.validation_interval, 1)
        for bad in (0, -1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                Config(training={"validation_interval": bad})
        for size, accum, interval, expected in (
            (7, 2, 1, set()), (7, 2, .5, {4}), (8, 2, .25, {2, 4, 6}),
            (10, 1, .3, {3, 6, 9}), (100, 1, .07, set(range(7, 100, 7))),
            (5, 2, 1e-300, {2, 4}), (3, 16, .25, set()),
        ):
            with self.subTest(size=size, accum=accum, interval=interval):
                self.assertEqual(_intermediate_validation_batches(size, accum, interval), expected)

    def observe_training(self, trainer, scores):
        seen, checks = [], []
        scores = iter(scores)
        normal_loss = trainer.train_one_example
        normal_validation = trainer.get_valid_loss

        def train_batch(batch):
            self.assertTrue(trainer.model.training)
            seen.extend(batch["sample_ids"])
            loss, bce, dice = normal_loss(batch)
            return loss * torch.rand(()), bce, dice

        def validate(loader):
            checks.append(trainer.scheduler.last_epoch)
            self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
            _, bce, dice = normal_validation(loader)
            return next(scores), bce, dice

        with patch.object(trainer, "train_one_example", side_effect=train_batch), \
             patch.object(trainer, "get_valid_loss", side_effect=validate):
            trainer.train()
        return seen, checks

    def test_mid_epoch_best_resumes_the_remainder_with_correct_updates_and_weights(self):
        trainer = Train(self.config, validation_interval=.5)
        seen, checks = self.observe_training(trainer, [5, 1, 2])
        self.assertEqual(checks, [0, 2, 4])
        best_path = trainer.output_path / "best.pt"
        best = torch.load(best_path, weights_only=True)
        self.assertEqual(best["best_metric"], 1)
        self.assertEqual(best["epoch_progress"]["batches_completed"], 4)
        last = torch.load(trainer.output_path / "last.pt", weights_only=True)
        self.assertIsNone(last["epoch_progress"])
        self.assertEqual(last["patience_counter"], 0)
        self.assertEqual(last["best_metric"], 1)
        self.assertIn("batch 4/7", (trainer.output_path / "log.txt").read_text(encoding="utf-8"))

        resumed = Train(checkpoint_path=best_path, output_path=self.root / "resumed")
        self.assertEqual(resumed.config.training.validation_interval, .5)
        tail, checks = self.observe_training(resumed, [1, 2])
        self.assertEqual(tail, seen[4:])
        self.assertEqual(checks, [2, 4])
        resumed_last = torch.load(resumed.output_path / "last.pt", weights_only=True)
        self.assertEqual(resumed_last["patience_counter"], 0)
        for name, weight in last["model_state_dict"].items():
            self.assertTrue(torch.equal(weight, resumed_last["model_state_dict"][name]), name)

        changed = Train(checkpoint_path=best_path, output_path=self.root / "changed", validation_interval=.25)
        self.assertEqual(self.observe_training(changed, [1, 2, 3])[1], [2, 3, 4])

    def test_extra_checks_keep_epoch_patience_and_training_rng(self):
        midway = Train(self.config, epochs=4, patience=2, validation_interval=.5)
        seen, checks = self.observe_training(midway, [1] * 5)
        self.assertEqual(checks, [0, 2, 4, 6, 8])
        regular = Train(self.config, epochs=4, patience=2)
        self.assertEqual(self.observe_training(regular, [1] * 3), (seen, [0, 4, 8]))
        for actual, expected in zip(midway.model.parameters(), regular.model.parameters()):
            self.assertTrue(torch.equal(actual, expected))
