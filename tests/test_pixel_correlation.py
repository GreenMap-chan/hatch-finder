import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from hatchfinder import HatchFinder, Train, load_config, PixelCorrelationSettings
from hatchfinder.pixel_correlation import PixelCorrelation
from hatchfinder.config import TrainingSettings


class PixelCorrelationTests(unittest.TestCase):
    def config(self, width=16):
        config = load_config(Path(__file__).resolve().parents[1] / 'examples/smoke_test.yaml')
        config.model.pixel_correlation = PixelCorrelationSettings(enabled=True, hidden_channels=width)
        return config

    def test_rotated_rectangles_have_correct_centers_and_regions(self):
        torch.manual_seed(71)
        branch = PixelCorrelation(PixelCorrelationSettings(downsample=1))
        for height, width in [(4, 6), (5, 7), (4, 7)]:
            template = torch.rand(1, 1, height, width).expand(-1, 3, -1, -1)
            for rotation in range(4):
                drawing = torch.ones(1, 3, 32, 40)
                crop = torch.rot90(template, rotation, (-2, -1))
                h, w = crop.shape[-2:]
                drawing[:, :, 9:9+h, 13:13+w] = crop
                centers, regions = branch.correlation_maps(drawing, template)
                self.assertGreater(float(centers.max()), 0.999)
                expected = torch.zeros_like(regions, dtype=torch.bool)
                expected[:, :, 9:9+h, 13:13+w] = True
                self.assertTrue(torch.equal(regions > 0.999, expected))

    def test_zero_initial_correction_and_learning(self):
        torch.manual_seed(13)
        for width in [16, 32]:
            model = HatchFinder(self.config(width))
            self.assertEqual(sum(p.numel() for p in model.pixel_correlation.parameters()), 29*width+1)
            drawing, mask, hatch = torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32), torch.rand(1, 3, 16, 24)
            branch = model.pixel_correlation
            model.pixel_correlation = None
            baseline = model(drawing, mask, hatch).detach()
            model.pixel_correlation = branch
            result = model(drawing, mask, hatch)
            torch.testing.assert_close(result, baseline, atol=0, rtol=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            for _ in range(2):
                optimizer.zero_grad()
                model(drawing, mask, hatch).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertGreater(float(branch.head[0].weight.grad.abs().sum()), 0)

    def test_old_and_new_weights_roundtrip(self):
        inputs = (torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32), torch.rand(1, 3, 16, 16))
        for enabled in [False, True]:
            config = self.config(32)
            config.model.pixel_correlation.enabled = enabled
            model = HatchFinder(config).eval()
            if enabled:
                torch.nn.init.normal_(model.pixel_correlation.head[-1].weight)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                model.save_weights(path)
                if not enabled:
                    saved = torch.load(path, weights_only=True)
                    del saved['model_config']['pixel_correlation']
                    torch.save(saved, path)
                restored = HatchFinder(load_model_path=path, device='cpu').eval()
                self.assertEqual(restored.pixel_correlation is not None, enabled)
                torch.testing.assert_close(model(*inputs), restored(*inputs), atol=0, rtol=0)

    def test_optimizer_scheduler_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config.output.directory = Path(directory)
            config.training.pixel_correlation_lr_multiplier = 2
            trainer = Train(config)
            branch_ids = {id(p) for p in trainer.model.pixel_correlation.parameters()}
            grouped = []
            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    grouped.append(id(p))
                    expected = config.training.learning_rate * (2 if id(p) in branch_ids else 1)
                    self.assertEqual(group['lr'], expected)
                    self.assertEqual(group['weight_decay'], config.training.weight_decay if p.ndim >= 2 else 0)
            self.assertEqual(len(grouped), len(set(grouped)))
            self.assertEqual(set(grouped), {id(p) for p in trainer.model.parameters()})
            scheduler = trainer.create_scheduler(4, 0)
            inputs = (torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32), torch.rand(1, 3, 16, 16))
            trainer.model(*inputs).square().mean().backward()
            trainer.optimizer.step()
            scheduler.step()
            path = Path(directory) / 'checkpoint.pt'
            trainer.model.save_checkpoint(trainer.optimizer, scheduler, 0, 0.5, 0, path)
            config.training.checkpoint_path = path
            resumed = Train(config)
            resumed_scheduler = resumed.create_scheduler(4, 0)
            self.assertEqual(resumed.model.load_checkpoint(resumed.optimizer, resumed_scheduler, path), (1, 0.5, 0))
            torch.testing.assert_close(trainer.model(*inputs), resumed.model(*inputs), atol=0, rtol=0)
            self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())
            for _ in range(3):
                resumed.optimizer.step()
                resumed_scheduler.step()
            for group in resumed.optimizer.param_groups:
                self.assertAlmostEqual(group['lr'], config.training.eta_min)

    def test_constant_and_small_inputs_are_finite(self):
        branch = PixelCorrelation(PixelCorrelationSettings())
        for hatch in [torch.ones(1, 3, 1, 1), torch.zeros(1, 3, 8, 12)]:
            maps = branch.correlation_maps(torch.ones(1, 3, 17, 19), hatch)
            for value in maps:
                self.assertEqual(value.shape, (1, 1, 4, 4))
                self.assertTrue(torch.isfinite(value).all())

    def test_validation(self):
        for arguments in [{'hidden_channels': 0}, {'downsample': 0}]:
            with self.assertRaises(ValidationError):
                PixelCorrelationSettings(**arguments)
        for multiplier in [0, float('inf'), float('nan')]:
            with self.assertRaises(ValidationError):
                TrainingSettings(pixel_correlation_lr_multiplier=multiplier)
        model = HatchFinder(self.config())
        with self.assertRaisesRegex(ValueError, 'batch_size=1'):
            model(torch.ones(2, 3, 32, 32), torch.ones(2, 1, 32, 32), torch.ones(2, 3, 16, 16))
        config = self.config()
        config.training.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'batch_size=1'):
            Train(config)


if __name__ == '__main__':
    unittest.main()
