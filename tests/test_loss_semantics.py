import unittest

import torch
import torch.nn.functional as F

from hatchfinder.losses import get_loss


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


if __name__ == "__main__":
    unittest.main()
