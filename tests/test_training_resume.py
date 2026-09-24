import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from hatchfinder import Train, load_config


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
