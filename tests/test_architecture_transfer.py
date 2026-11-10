import tempfile
import unittest
from pathlib import Path

import torch

from hatchfinder import Config, HatchFinder, Train, load_config
from hatchfinder.architecture_transfer import transfer_weights
from hatchfinder.config import ModelSettings, TransformerSettings, config_from_pt_data


class ArchitectureTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(Path(__file__).resolve().parents[1] / "examples/smoke_test.yaml")
        self.source = HatchFinder(self.config)
        self.path = self.root / "source.pt"
        self.source.save_weights(self.path)

    def experiment(self, model=None, **training):
        return Config.model_validate({
            "model": model or {},
            "training": {"load_new_model_config": True, "load_model_path": self.path,
                         "bf16": False, "num_workers": 0, **training},
            "runtime": {"device": "cpu"},
            "data": {"dataset": self.config.data.dataset},
            "output": {"directory": self.root / "run", "unique_run_directory": False},
        })

    def test_partial_yaml_inherits_before_validation_and_resolves_source(self):
        path = self.root / "config.yaml"
        path.write_text("training:\n  load_new_model_config: true\n  load_model_path: source.pt\nmodel:\n  transformers:\n    - level: 2\n      num_heads: 4\n", encoding="utf-8")
        config = load_config(path)
        self.assertEqual(config.model.drawing_channels, [8, 16, 32])
        self.assertEqual(config.model.transformers[0].num_blocks, 1)

    def test_zero_gate_branch_preserves_predictions_and_old_weights(self):
        trainer = Train(self.experiment({"transformers": [{"level": 2, "initial_gate": 0.0}]}))
        for key, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]), key)
        inputs = (torch.randn(1, 3, 16, 16), torch.ones(1, 1, 16, 16), torch.randn(1, 3, 16, 16))
        self.source.eval()
        trainer.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.source(*inputs), trainer.model(*inputs))
        trainer.model(*inputs).mean().backward()
        self.assertIsNotNone(trainer.model.drawing_encoder.transformers["2"].gate.grad)

    def test_width_change_resets_entire_layer_and_loads_unaffected_layers(self):
        config = self.experiment({"drawing_channels": [16, 16, 32]})
        model = HatchFinder(config)
        original_bias = model.drawing_encoder.blocks[1][0].bias.detach().clone()
        report = transfer_weights(model, self.path)
        self.assertTrue(torch.equal(original_bias, model.drawing_encoder.blocks[1][0].bias))
        self.assertTrue(torch.equal(self.source.hatch_encoder.blocks[0][0].weight, model.hatch_encoder.blocks[0][0].weight))
        self.assertTrue(any("parameter shape changed" in line for line in report))

    def test_added_level_maps_decoder_by_encoder_level(self):
        changes = {"drawing_channels": [8, 16, 32, 32], "hatch_channels": [8, 16, 32, 32],
                   "hatch_pool_sizes": [2, 3, 4, 4], "match_dims": [0, 0, 32, 32],
                   "match_feature_dims": [0, 0, 8, 8], "match_gate_initial_values": [.01, .01, 1., 1.],
                   "drawing_residual_blocks": [0, 0, 0, 0], "hatch_residual_blocks": [0, 0, 0, 0]}
        model = HatchFinder(self.experiment(changes))
        transfer_weights(model, self.path)
        for key, value in self.source.heatmap_decoder.blocks[0].state_dict().items():
            self.assertTrue(torch.equal(value, model.heatmap_decoder.blocks[1].state_dict()[key]), key)
        output = model(torch.randn(1, 3, 16, 16), torch.ones(1, 1, 16, 16), torch.randn(1, 3, 16, 16))
        self.assertEqual(output.shape, (1, 1, 16, 16))
        model.save_weights(self.path)
        reduced = HatchFinder(self.config)
        transfer_weights(reduced, self.path)
        for key, value in model.heatmap_decoder.blocks[1].state_dict().items():
            self.assertTrue(torch.equal(value, reduced.heatmap_decoder.blocks[0].state_dict()[key]), key)

    def test_matcher_output_is_not_loaded_into_new_hidden_layer(self):
        model = HatchFinder(self.experiment({"matcher_channels": [32, 16, 16]}))
        initial = model.output_comparison.matchers["2"][4].weight.detach().clone()
        transfer_weights(model, self.path)
        self.assertTrue(torch.equal(initial, model.output_comparison.matchers["2"][4].weight))
        self.assertTrue(torch.equal(self.source.output_comparison.matchers["2"][4].weight,
                                    model.output_comparison.matchers["2"][6].weight))

    def test_checkpoint_starts_fresh_and_result_can_resume_without_source(self):
        source_trainer = Train(self.config, output_path=self.root / "source_run")
        scheduler = source_trainer.create_scheduler(10, 0)
        checkpoint = self.root / "checkpoint.pt"
        source_trainer.model.save_checkpoint(source_trainer.optimizer, scheduler, 7, .2, 3, checkpoint)
        config = self.experiment(load_model_path=None, checkpoint_path=checkpoint)
        trainer = Train(config)
        self.assertIsNone(trainer.checkpoint_path)
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(len(trainer.optimizer.state), 0)
        second = Train(config)
        self.assertNotEqual(trainer.output_path, second.output_path)
        saved = trainer.output_path / "last.pt"
        trainer.model.save_checkpoint(trainer.optimizer, trainer.create_scheduler(10, 0), 0, .4, 0, saved)
        checkpoint.unlink()
        restored = config_from_pt_data(torch.load(saved, weights_only=True))
        self.assertEqual(restored.model, trainer.config.model)
        resumed = Train(Config(training={"checkpoint_path": saved}, runtime={"device": "cpu"}),
                        load_new_model_config=False)
        self.assertFalse(resumed.config.training.load_new_model_config)
        self.assertEqual(resumed._restore_training(resumed.create_scheduler(10, 0), 10, 0), (.4, 0))
        self.assertEqual(resumed.start_epoch, 1)

    def test_nested_inheritance_and_explicit_empty_list(self):
        self.config.model.pixel_correlation.hidden_channels = 24
        self.config.model.transformers = [TransformerSettings(level=2)]
        self.config.model = ModelSettings.model_validate(self.config.model.model_dump())
        HatchFinder(self.config).save_weights(self.path)
        config = self.experiment({"pixel_correlation": {"enabled": True}, "transformers": []})
        self.assertEqual(config.model.pixel_correlation.hidden_channels, 24)
        self.assertTrue(config.model.pixel_correlation.enabled)
        self.assertEqual(config.model.transformers, [])

    def test_ambiguous_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            self.experiment(checkpoint_path=self.path)

    def test_heads_reset_attention_atomically_but_keep_feedforward_and_gate(self):
        self.config.model.transformers = [TransformerSettings(level=2, num_heads=4)]
        source = HatchFinder(self.config)
        with torch.no_grad():
            source.drawing_encoder.transformers["2"].gate.fill_(.7)
            source.drawing_encoder.transformers["2"].transformer.layers[0].self_attn.out_proj.bias.fill_(5.)
        source.save_weights(self.path)
        model = HatchFinder(self.experiment({"transformers": [{"level": 2, "num_heads": 8, "initial_gate": 0.0}]}))
        layer = model.drawing_encoder.transformers["2"].transformer.layers[0]
        initial = layer.self_attn.in_proj_weight.detach().clone()
        transfer_weights(model, self.path)
        self.assertTrue(torch.equal(initial, layer.self_attn.in_proj_weight))
        self.assertTrue(torch.equal(layer.self_attn.out_proj.bias, torch.zeros_like(layer.self_attn.out_proj.bias)))
        self.assertTrue(torch.equal(layer.linear1.weight, source.drawing_encoder.transformers["2"].transformer.layers[0].linear1.weight))
        self.assertAlmostEqual(model.drawing_encoder.transformers["2"].gate.item(), .7, places=6)

    def test_add_remove_blocks_and_branches(self):
        changes = {"transformers": [{"level": 2, "num_blocks": 2}],
                   "hatch_transformers": [{"level": 2}],
                   "cross_attention": [{"level": 2}],
                   "drawing_residual_blocks": [1, 0, 1],
                   "pixel_correlation": {"enabled": True}}
        expanded = HatchFinder(self.experiment(changes))
        transfer_weights(expanded, self.path)
        expanded.save_weights(self.path)
        reduced = HatchFinder(self.experiment({"transformers": [{"level": 2, "num_blocks": 1}],
                                               "hatch_transformers": [], "cross_attention": [],
                                               "drawing_residual_blocks": [0, 0, 0],
                                               "pixel_correlation": {"enabled": False}}))
        transfer_weights(reduced, self.path)
        for key, value in reduced.state_dict().items():
            self.assertTrue(torch.equal(value, expanded.state_dict()[key]), key)

    def test_enabling_matching_retains_existing_learned_gate(self):
        with torch.no_grad():
            self.source.heatmap_decoder.match_gates["2"].fill_(.6)
        self.source.save_weights(self.path)
        model = HatchFinder(self.experiment({"match_dims": [8, 0, 32], "match_feature_dims": [4, 0, 8]}))
        transfer_weights(model, self.path)
        self.assertAlmostEqual(model.heatmap_decoder.match_gates["2"].item(), .6, places=6)

    def test_equal_size_projection_with_changed_pool_layout_is_reset(self):
        model = HatchFinder(self.experiment({"hatch_channels": [8, 16, 128], "hatch_pool_sizes": [2, 3, 2]}))
        initial = model.hatch_projections["2"].weight.detach().clone()
        self.assertEqual(initial.shape, self.source.hatch_projections["2"].weight.shape)
        transfer_weights(model, self.path)
        self.assertTrue(torch.equal(initial, model.hatch_projections["2"].weight))

    def test_replacement_list_entries_use_defaults_not_source_entry_settings(self):
        self.config.model.transformers = [TransformerSettings(level=2, num_heads=4, num_blocks=3)]
        HatchFinder(self.config).save_weights(self.path)
        config = self.experiment({"transformers": [{"level": 2}]})
        self.assertEqual(config.model.transformers[0].num_heads, 8)
        self.assertEqual(config.model.transformers[0].num_blocks, 1)


if __name__ == "__main__":
    unittest.main()
