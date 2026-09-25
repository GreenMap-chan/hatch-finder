import unittest

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from hatchfinder.config import Config, TrainingSettings
from hatchfinder.losses import get_loss, get_tversky
from hatchfinder.train import Train


class LossSemanticsTests(unittest.TestCase):
    def test_mixed_batch_dice_uses_only_positive_targets(self):
        logits = torch.tensor([[[[0.2, -0.4]]], [[[0.8, 0.1]]]], requires_grad=True)
        targets = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
        mask = torch.ones_like(targets)

        loss, bce, dice = get_loss(logits, targets, mask)
        probabilities = torch.sigmoid(logits[0])
        expected_dice = 1 - (
            2 * probabilities[0, 0, 0] + 1e-6
        ) / (probabilities.sum() + 1 + 1e-6)
        expected_bce = F.binary_cross_entropy_with_logits(logits, targets)

        torch.testing.assert_close(dice, expected_dice)
        torch.testing.assert_close(bce, expected_bce)
        torch.testing.assert_close(loss, expected_bce + expected_dice)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_negative_batch_has_zero_dice_loss(self):
        logits = torch.zeros((2, 1, 2, 2), requires_grad=True)
        targets = torch.zeros_like(logits)
        mask = torch.ones_like(logits)

        loss, bce, dice = get_loss(logits, targets, mask)
        self.assertEqual(dice.item(), 0.0)
        torch.testing.assert_close(loss, bce)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_tversky_weights_false_negatives_more_than_false_positives(self):
        logits = torch.zeros((1, 1, 1, 3), requires_grad=True)
        targets = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        mask = torch.tensor([[[[1.0, 1.0, 0.0]]]])
        loss = get_tversky(logits, targets, mask)
        expected = 1 - (0.5 + 1e-6) / (0.5 + 0.3 * 0.5 + 0.7 * 0.5 + 1e-6)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_tversky_training_uses_common_validation_loss(self):
        class FixedModel:
            device = torch.device("cpu")

            def __call__(self, drawing, mask, hatch):
                return torch.zeros_like(mask, requires_grad=True)

            def get_example_loss(self, drawing, mask, hatch, target):
                return get_loss(self(drawing, mask, hatch), target, mask)

            def eval(self):
                return self

            def train(self):
                return self

        trainer = object.__new__(Train)
        trainer.config = Config(training=TrainingSettings(loss="bce_tversky"))
        trainer.model = FixedModel()
        trainer.bf16_enabled = False
        example = {
            "drawing": torch.zeros(1, 3, 1, 3),
            "search_mask": torch.ones(1, 1, 1, 3),
            "hatch": torch.zeros(1, 3, 1, 3),
            "target": torch.tensor([[[[1.0, 0.0, 0.0]]]]),
        }
        total, bce, tversky = trainer.train_one_example(example)
        torch.testing.assert_close(total, bce + tversky)
        self.assertNotAlmostEqual(tversky.item(), get_loss(
            torch.zeros_like(example["target"]), example["target"],
            example["search_mask"],
        )[2].item())
        validation_loss, _, validation_dice = trainer.get_valid_loss([example])
        expected_validation = get_loss(
            torch.zeros_like(example["target"]), example["target"],
            example["search_mask"],
        )
        self.assertAlmostEqual(validation_loss, expected_validation[0].item())
        self.assertAlmostEqual(validation_dice, expected_validation[2].item())

    def test_invalid_tversky_weights_are_rejected(self):
        with self.assertRaises(ValidationError):
            TrainingSettings(loss="bce_tversky", tversky_fp_weight=0.3,
                             tversky_fn_weight=0.3)


if __name__ == "__main__":
    unittest.main()
