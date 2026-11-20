from torch.utils.data import Dataset, get_worker_info
from PIL import Image
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from contextlib import ExitStack
import json
import random
import torch
import torch.nn.functional as F

from .augmentation import TrainingAugmentation
from .config import AugmentationSettings, DataSettings
from .search_windows import SampleKind, choose_window, sample_kind


class HatchDataset(Dataset):
    def __init__(
        self,
        path: Path,
        dataset_type: str,
        data_config: DataSettings,
        augmentation_config: AugmentationSettings,
        augment: bool = False,
    ):
        if dataset_type not in {"train", "valid"}:
            raise ValueError("dataset_type must be 'train' or 'valid'")

        self.items = []
        self.path = path
        self.data_config = data_config
        self.augmentation_config = augmentation_config
        self.split = dataset_type
        self.generation = data_config.search_mask_generation
        self.augment = augment
        self._augmentation = None
        self._augmentation_worker_id = None
        self._window_rng = None
        self._window_worker_id = None
        self.window_start_epoch = 0

        with open(path / dataset_type / "manifest.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                entry = {name: item[name] for name in ("drawing", "hatch", "target")}
                entry["search_mask"] = item.get("search_mask")
                if not entry["search_mask"] and not self.generation.enabled:
                    raise ValueError(f"Missing search_mask for {entry['drawing']}. Supply a saved mask or "
                                     "enable data.search_mask_generation with a whole-tile target.")
                self.items.append(entry)
        if self.split == "train" and any(not item["search_mask"] for item in self.items):
            self._index_generation_targets()

    def _index_generation_targets(self):
        self._empty_targets = {}
        self._absent_indices = []
        self._present_indices = []
        for index, item in enumerate(self.items):
            if item["search_mask"]:
                continue
            with Image.open(self.path / item["target"]) as im:
                empty = im.convert("L").getextrema()[1] <= 127
            self._empty_targets[index] = empty
            pool = self._absent_indices if empty else self._present_indices
            pool.append(index)
        cfg = self.generation
        if cfg.negative_fraction > 0 and cfg.outside_window_share_of_negatives < 1 and not self._absent_indices:
            raise ValueError("Search mask generation needs a pair with an empty whole-tile target for absent negatives")
        if (cfg.negative_fraction < 1 or cfg.outside_window_share_of_negatives > 0) and not self._present_indices:
            raise ValueError("Search mask generation needs a pair with a nonempty whole-tile target")

    def __len__(self):
        return len(self.items)

    def _get_augmentation(self) -> TrainingAugmentation:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1

        if self._augmentation is None or self._augmentation_worker_id != worker_id:
            augmentation_config = self.augmentation_config.model_copy(
                update={"seed": self.augmentation_config.seed + worker_id + 1}
            )
            self._augmentation = TrainingAugmentation(augmentation_config)
            self._augmentation_worker_id = worker_id

        return self._augmentation

    def __getitem__(self, index):
        index = range(len(self.items))[index]
        if self.items[index]["search_mask"]:
            return self._read_item(index)

        if self.split == "valid":
            rng = random.Random(f"{self.augmentation_config.seed}:valid:{index}")
            kind = None  # Keep this validation pair and its natural positive/absent type.
            pool = [index]
        else:
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else -1
            if self._window_worker_id != worker_id:
                seed = self.augmentation_config.seed + (worker.seed if worker is not None else 0)
                if self.window_start_epoch:
                    seed = f"{seed}:{self.window_start_epoch}"
                self._window_rng = random.Random(seed)
                self._window_worker_id = worker_id
            rng = self._window_rng
            kind = sample_kind(self.generation, rng)
            pool = self._absent_indices if kind == "absent" else self._present_indices

        full_tile = kind != "outside" and rng.random() < self.generation.full_tile_probability
        for attempt in range(17):
            # Try the same size on other pairs before sampling new dimensions.
            quantiles = (rng.random(), rng.random())
            exhaustive = attempt == 16
            candidates = rng.sample(pool, len(pool) if exhaustive else min(4, len(pool)))
            if self.split == "train" and attempt == 0 and self._empty_targets[index] == (kind == "absent"):
                candidates = [index] + [candidate for candidate in candidates if candidate != index]
            for candidate in candidates:
                result = self._read_item(candidate, kind, rng, quantiles, full_tile, exhaustive)
                if result is not None:
                    return result
        raise ValueError(
            f"No legal {kind or 'validation'} search window found in {self.split} "
            f"(requested row {index}). Check tile dimensions, window ranges, margins, "
            "min_positive_pixels and target coverage."
        )

    def _read_item(self, index, kind: SampleKind | None = None, rng=None, quantiles=None,
                   full_tile=False, exhaustive=False):
        item = self.items[index]
        with ExitStack() as images:
            def load(name: str, mode: str) -> Image.Image:
                with Image.open(self.path / item[name]) as source:
                    converted = source.convert(mode)
                images.callback(converted.close)
                return converted

            drawing = load("drawing", "RGB")
            hatch = load("hatch", "RGB")
            target = load("target", "L")
            if rng is None:
                mask = load("search_mask", "L")
            else:
                if drawing.size != target.size:
                    raise ValueError(f"Drawing and full target sizes differ for {item['drawing']}")
                mask = Image.new("L", drawing.size, 255)
                images.callback(mask.close)

            if self.augment and (rng is None or self.split == "train"):
                drawing, mask, hatch, target = self._get_augmentation()(
                    drawing, mask, hatch, target
                )
                for image in (drawing, mask, hatch, target):
                    images.callback(image.close)

            if rng is not None:
                if kind is None:
                    kind = "positive" if target.getextrema()[1] > 127 else "absent"
                bounds = choose_window(target, kind, self.generation, rng, quantiles,
                                       full_tile, exhaustive=exhaustive)
                if bounds is None:
                    return None
                mask = Image.new("L", drawing.size, 0)
                images.callback(mask.close)
                mask.paste(255, bounds)

            drawing_tensor = to_tensor(drawing)
            mask_tensor = (to_tensor(mask) > 0.5).float()
            hatch_tensor = to_tensor(hatch)
            target_tensor = (to_tensor(target) > 0.5).float()
            if rng is not None:
                target_tensor.mul_(mask_tensor)

        result = {
            "drawing": drawing_tensor,
            "search_mask": mask_tensor,
            "hatch": hatch_tensor,
            "target": target_tensor,
        }
        if rng is not None and self.split == "train":
            result["sample_kind"] = kind
        return result

    def pad_tensor(
        self,
        tensor: torch.Tensor,
        target_h: int,
        target_w: int,
        value: float,
    ):
        _, h, w = tensor.shape

        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return tensor

        return F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            value=value,
        )

    def collate_fn(self, batch):
        max_h = max(item["drawing"].shape[-2] for item in batch)
        max_w = max(item["drawing"].shape[-1] for item in batch)

        max_hatch_h = max(item["hatch"].shape[-2] for item in batch)
        max_hatch_w = max(item["hatch"].shape[-1] for item in batch)

        drawings = []
        masks = []
        hatches = []
        targets = []

        for item in batch:
            drawings.append(
                self.pad_tensor(
                    item["drawing"],
                    max_h,
                    max_w,
                    value=1.0,
                )
            )

            masks.append(
                self.pad_tensor(
                    item["search_mask"],
                    max_h,
                    max_w,
                    value=0.0,
                )
            )

            targets.append(
                self.pad_tensor(
                    item["target"],
                    max_h,
                    max_w,
                    value=0.0,
                )
            )

            hatches.append(
                self.pad_tensor(
                    item["hatch"],
                    max_hatch_h,
                    max_hatch_w,
                    value=1.0,
                )
            )

        result = {
            "drawing": torch.stack(drawings),
            "search_mask": torch.stack(masks),
            "hatch": torch.stack(hatches),
            "target": torch.stack(targets),
        }
        sample_kinds = [item["sample_kind"] for item in batch if "sample_kind" in item]
        if sample_kinds:
            result["sample_kind"] = sample_kinds
        return result
