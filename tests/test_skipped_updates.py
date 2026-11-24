import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from hatchfinder import Config, Train, load_config


class SkippedUpdateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.config = load_config(Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml")
        self.config.output.directory = Path(temp.name) / "run"

    def run_epoch(self, trainer, scheduler, samples, losses, bad_grad_id=None, epoch=0):
        loader = DataLoader([{"drawing": torch.zeros(1), "sample_ids": sample} for sample in samples],
                            batch_size=trainer.config.training.batch_size)
        param = next(trainer.model.parameters()).flatten()[0]
        seen = []

        def objective(batch):
            ids = batch["sample_ids"]
            seen.extend(ids)
            value = losses.get(ids[0], 2.0)
            loss = (param - param.detach()) + value
            if not torch.isfinite(loss.detach()):
                loss.register_hook(lambda grad: self.fail("backward called for a non-finite loss"))
            elif bad_grad_id in ids:
                loss.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return loss, None, None

        with patch.object(trainer, "train_one_example", side_effect=objective):
            trainer._run_training_epoch(loader, scheduler, epoch, 3)
        return seen

    def test_loss_discards_group_resets_streak_and_keeps_clean_metrics_and_weights(self):
        trainer = Train(self.config, gradient_accumulation_steps=3, max_consecutive_skipped_updates=1)
        scheduler = trainer.create_scheduler(10, 0)
        seen = self.run_epoch(trainer, scheduler, list("ABCDEFG"),
                              {"A": 100.0, "B": float("nan"), "G": float("inf")})
        self.assertEqual(seen, list("ABDEFG"))
        self.assertEqual(trainer.consecutive_skipped_updates, 1)
        self.assertEqual(scheduler.last_epoch, 1)
        clean = Train(self.config, gradient_accumulation_steps=3)
        self.run_epoch(clean, clean.create_scheduler(10, 0), list("DEF"), {})
        for actual, expected in zip(trainer.model.parameters(), clean.model.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        log = (trainer.output_path / "log.txt").read_text(encoding="utf-8")
        self.assertIn("batch 2/7: skipped update (batches 1-3): non-finite loss=nan, batch sample IDs=['B']", log)
        self.assertIn("batch 7/7: skipped update (batches 7-7): non-finite loss=inf", log)
        self.assertIn("loss: 2.000", log)
        self.assertIn("skipped_updates: 2", log)

    def test_nonfinite_gradient_logs_whole_group_and_default_stops_before_step(self):
        trainer = Train(self.config, gradient_accumulation_steps=3, batch_size=2)
        scheduler = trainer.create_scheduler(10, 0)
        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient norm=inf"):
            self.run_epoch(trainer, scheduler, list("ABCDE"), {}, bad_grad_id="B")
        log = (trainer.output_path / "log.txt").read_text(encoding="utf-8")
        self.assertIn("group sample IDs=['A', 'B', 'C', 'D', 'E']", log)
        self.assertIn("consecutive=1, limit=0", log)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(trainer.optimizer.state)
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))

    def test_skip_limit_survives_epoch_boundary_and_resume(self):
        trainer = Train(self.config, max_consecutive_skipped_updates=1)
        scheduler = trainer.create_scheduler(10, 0)
        self.run_epoch(trainer, scheduler, ["A"], {}, bad_grad_id="A")
        self.assertIn("no successful updates", (trainer.output_path / "log.txt").read_text(encoding="utf-8"))
        path = trainer.output_path / "last.pt"
        trainer._save_checkpoint(scheduler, 0, 0.5, 0, path)

        self.config.training.checkpoint_path = path
        for limit in (1, 2):
            with self.subTest(limit=limit):
                if limit == 2:
                    self.config.training.max_consecutive_skipped_updates = limit
                resumed = Train(self.config)
                self.assertEqual(resumed.config.training.max_consecutive_skipped_updates, limit)
                schedule = resumed.create_scheduler(10, 0)
                resumed._restore_training(schedule, 10, 0)
                self.assertEqual(resumed.consecutive_skipped_updates, 1)
                samples = list("BC")[:limit]
                with self.assertRaisesRegex(FloatingPointError, f"consecutive={limit + 1}, limit={limit}"):
                    self.run_epoch(resumed, schedule, samples, dict.fromkeys(samples, float("nan")), epoch=1)
                self.assertEqual(schedule.last_epoch, 0)

    def test_limit_requires_nonnegative_integer(self):
        for value in (-1, 0.5, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Config(training={"max_consecutive_skipped_updates": value})
