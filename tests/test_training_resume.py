import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from hatchfinder import HatchFinder, Train, load_config
from hatchfinder.config import ModelSettings, TransformerSettings


class FineTuningDropoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(
            Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml"
        )
        self.config.output.directory = self.root / "run"
        self.config.model.transformers = [
            TransformerSettings(level=0, dropout=0.3),
            TransformerSettings(level=2, dropout=0.4),
        ]
        self.config.model.hatch_transformers = [TransformerSettings(level=2, dropout=0.5)]
        self.source = HatchFinder(self.config)
        self.weights_path = self.root / "model.pt"
        self.source.save_weights(self.weights_path)
        self.config.training.load_model_path = self.weights_path

    def test_explicit_dropout_overrides_by_level_and_loads_weights(self):
        self.config.model.transformers = [
            TransformerSettings(level=2, dropout=0.0),
            TransformerSettings(level=0),
        ]
        self.config.model.hatch_transformers = [TransformerSettings(level=2, dropout=0.2)]
        trainer = Train(self.config)
        self.assertEqual([t.dropout for t in trainer.config.model.transformers], [0.3, 0.0])
        for encoder, level, expected in (
            (trainer.model.drawing_encoder, "0", 0.3),
            (trainer.model.drawing_encoder, "2", 0.0),
            (trainer.model.hatch_encoder, "2", 0.2),
        ):
            layer = encoder.transformers[level].transformer.layers[0]
            self.assertEqual(layer.self_attn.dropout, expected)
            self.assertEqual(layer.dropout.p, expected)
            self.assertEqual(layer.dropout1.p, expected)
            self.assertEqual(layer.dropout2.p, expected)
        for name, value in self.source.state_dict().items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[name]), name)
        saved_config = load_config(trainer.output_path / "config.yaml")
        self.assertEqual(saved_config.model, trainer.config.model)
        self.assertEqual(self.config.model.transformers[1].dropout, 0.0)

    def test_omitted_dropout_keeps_saved_values(self):
        for omit_lists in (False, True):
            with self.subTest(omit_lists=omit_lists):
                self.config.model = ModelSettings() if omit_lists else ModelSettings(
                    transformers=[{"level": 2}], hatch_transformers=[{"level": 2}]
                )
                trainer = Train(self.config)
                self.assertEqual(trainer.config.model, self.source.config.model)

    def test_unknown_level_with_explicit_dropout_is_rejected(self):
        self.config.model.transformers = [TransformerSettings(level=1, dropout=0.2)]
        with self.assertRaisesRegex(ValueError, "no transformer at this level"):
            Train(self.config)

    def test_resume_keeps_saved_dropout_and_strict_config_check(self):
        checkpoint_path = self.root / "last.pt"
        torch.save({
            "config": self.source.config.model_dump(mode="json"),
            "model_state_dict": self.source.state_dict(),
            "epoch": 0, "best_metric": 0.5, "patience_counter": 0,
        }, checkpoint_path)
        self.config.training.checkpoint_path = checkpoint_path
        self.config.model.transformers[0].dropout = 0.1
        trainer = Train(self.config)
        self.assertEqual(trainer.config.model, self.source.config.model)
        self.assertEqual(trainer.model.load_checkpoint(None, None, checkpoint_path), (1, 0.5, 0))
        changed_model = HatchFinder(self.config)
        with self.assertRaisesRegex(ValueError, "does not match"):
            changed_model.load_checkpoint(None, None, checkpoint_path)

    def test_weight_loading_still_rejects_other_config_changes(self):
        self.config.model.transformers[0].downsample = 2
        changed_model = HatchFinder(self.config)
        with self.assertRaisesRegex(ValueError, "does not match"):
            changed_model.load_model(self.weights_path)


class TrainingCheckpointTests(unittest.TestCase):
    def test_new_runs_get_unique_directories_but_resume_reuses_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "run"
            config = load_config(Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml")
            config.output.directory = base

            first = Train(config)
            second = Train(config)
            third = Train(config)
            self.assertEqual([first.output_path, second.output_path, third.output_path], [
                base, Path(f"{base}_0"), Path(f"{base}_1"),
            ])
            self.assertEqual(second.config.output.directory, second.output_path)

            config.output.unique_run_directory = False
            self.assertEqual(Train(config).output_path, base)

            config.output.unique_run_directory = True
            checkpoint_path = base / "last.pt"
            torch.save({"config": first.config.model_dump(mode="json")}, checkpoint_path)
            config.training.checkpoint_path = checkpoint_path
            self.assertEqual(Train(config).output_path, base)

    def test_best_exists_without_improvement_and_resume_extends_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for split in ("train", "valid"):
                split_path = dataset / split
                entries = {}
                for name, color in (
                    ("drawing", "white"),
                    ("search_mask", "white"),
                    ("hatch", "black"),
                    ("target", "black"),
                ):
                    image_path = split_path / name / "example.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L" if name in {"search_mask", "target"} else "RGB", (16, 16), color).save(image_path)
                    entries[name] = f"{split}/{name}/example.png"
                (split_path / "manifest.jsonl").write_text(json.dumps(entries) + "\n", encoding="utf-8")

            config = load_config(Path(__file__).resolve().parents[1] / "examples" / "smoke_test.yaml")
            config.data.dataset = dataset
            config.data.drawing_pad_multiple = 16
            config.augmentation.enabled = False
            config.output.directory = root / "first"
            trainer = Train(config)
            with patch.object(trainer, "get_valid_loss", side_effect=[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]):
                trainer.train()

            first_best = torch.load(root / "first" / "best.pt", weights_only=True)
            self.assertEqual(first_best["epoch"], -1)
            self.assertTrue((root / "first" / "last.pt").exists())

            config.training.checkpoint_path = root / "first" / "last.pt"
            config.training.epochs = 2
            config.output.directory = root / "second"
            resumed = Train(config)
            self.assertEqual(resumed.config.training.epochs, 2)
            self.assertEqual(resumed.output_path, root / "second")
            resumed.train()

            last = torch.load(root / "second" / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(last["scheduler_state_dict"]["T_max"], 2)
            self.assertTrue((root / "second" / "best.pt").exists())

            replacement_dataset = root / "replacement_dataset"
            shutil.copytree(dataset, replacement_dataset)
            config.data.dataset = replacement_dataset
            config.output.directory = root / "third"
            changed_dataset = Train(config)
            self.assertTrue(changed_dataset.dataset_changed)
            self.assertEqual(changed_dataset.config.data.dataset, replacement_dataset)
            with patch.object(changed_dataset, "get_valid_loss", side_effect=[(0.25, 0.25, 0.0), (0.5, 0.5, 0.0)]):
                changed_dataset.train()

            new_best = torch.load(root / "third" / "best.pt", weights_only=True)
            self.assertEqual(new_best["epoch"], 0)
            self.assertEqual(new_best["best_metric"], 0.25)


if __name__ == "__main__":
    unittest.main()
