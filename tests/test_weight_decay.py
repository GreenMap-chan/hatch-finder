import unittest

import torch
from pydantic import ValidationError
from torch import nn

from hatchfinder import HatchFinder, Train
from hatchfinder.config import TrainingSettings


class WeightDecayTests(unittest.TestCase):
    @staticmethod
    def optimizer_for(model, lr=0.1, weight_decay=0.2):
        trainer = Train.__new__(Train)
        trainer.model = model
        return trainer.create_optimizer(lr, weight_decay)

    def test_model_parameter_groups(self):
        model = HatchFinder(
            device="cpu",
            transformers=[{"level": 2}],
            hatch_transformers=[{"level": 2}],
        )
        frozen = next(model.parameters())
        frozen.requires_grad_(False)
        optimizer = self.optimizer_for(model)
        grouped = [p for group in optimizer.param_groups for p in group["params"]]
        grouped_ids = [id(p) for p in grouped]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(
            set(grouped_ids), {id(p) for p in model.parameters() if p.requires_grad}
        )
        self.assertNotIn(id(frozen), grouped_ids)
        decay = {
            id(p): group["weight_decay"]
            for group in optimizer.param_groups
            for p in group["params"]
        }
        attention_count = 0
        norm_count = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight.requires_grad:
                    self.assertEqual(decay[id(module.weight)], 0.2)
            if isinstance(module, nn.MultiheadAttention):
                attention_count += 1
                self.assertEqual(decay[id(module.in_proj_weight)], 0.2)
                self.assertEqual(decay[id(module.in_proj_bias)], 0.0)
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                norm_count += 1
                for param in module.parameters(recurse=False):
                    self.assertEqual(decay[id(param)], 0.0)
        self.assertEqual(attention_count, 2)
        self.assertGreater(norm_count, 0)
        for name, param in model.named_parameters():
            if "bias" in name or "gate" in name:
                self.assertEqual(decay[id(param)], 0.0, name)

    def test_zero_gradient_step_only_decays_matrix_weights(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
                param.grad = torch.zeros_like(param)
        optimizer = self.optimizer_for(model)
        optimizer.step()
        torch.testing.assert_close(model[0].weight, torch.full_like(model[0].weight, 0.98))
        for param in (model[0].bias, model[1].weight, model[1].bias):
            torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)

    def test_zero_weight_decay_disables_decay(self):
        model = nn.Linear(3, 2)
        before = [param.detach().clone() for param in model.parameters()]
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        self.optimizer_for(model, weight_decay=0.0).step()
        for param, original in zip(model.parameters(), before):
            torch.testing.assert_close(param, original, rtol=0, atol=0)

    def test_legacy_filter_is_ignored_and_not_serialized(self):
        data = {"decay_parameters": ["nonexistent"], "weight_decay": 0.03}
        settings = TrainingSettings.model_validate(data)
        self.assertEqual(settings.weight_decay, 0.03)
        self.assertNotIn("decay_parameters", settings.model_dump())
        self.assertNotIn("decay_parameters", TrainingSettings.model_json_schema()["properties"])
        self.assertEqual(data["decay_parameters"], ["nonexistent"])
        with self.assertRaises(ValidationError):
            TrainingSettings(weight_decayy=0.03)


if __name__ == "__main__":
    unittest.main()
