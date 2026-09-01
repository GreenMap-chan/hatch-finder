from torch.utils.data import Dataset, get_worker_info
from PIL import Image
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F

from .augmentation import TrainingAugmentation
from .config import AugmentationSettings, DataSettings


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
        self.augment = augment
        self._augmentation = None
        self._augmentation_worker_id = None

        with open(path / dataset_type / "manifest.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.items.append({"drawing": item["drawing"], "search_mask": item["search_mask"], "hatch": item["hatch"], "target": item["target"]})

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
        item = self.items[index]

        with Image.open(self.path / item["drawing"]) as image:
            drawing = image.convert("RGB")
        with Image.open(self.path / item["search_mask"]) as image:
            mask = image.convert("L")
        with Image.open(self.path / item["hatch"]) as image:
            hatch = image.convert("RGB")
        with Image.open(self.path / item["target"]) as image:
            target = image.convert("L")

        if self.augment:
            augmented = self._get_augmentation()(drawing, mask, hatch, target)
            drawing.close()
            mask.close()
            hatch.close()
            target.close()
            drawing, mask, hatch, target = augmented

        drawing_tensor = to_tensor(drawing)
        mask_tensor = (to_tensor(mask) > 0.5).float()
        hatch_tensor = to_tensor(hatch)
        target_tensor = (to_tensor(target) > 0.5).float()

        drawing.close()
        mask.close()
        hatch.close()
        target.close()

        return {
            "drawing": drawing_tensor,
            "search_mask": mask_tensor,
            "hatch": hatch_tensor,
            "target": target_tensor,
        }

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

        return F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            value=value,
        )

    def collate_fn(self, batch):
        max_h = max(item["drawing"].shape[-2] for item in batch)
        max_w = max(item["drawing"].shape[-1] for item in batch)

        multiple = self.data_config.drawing_pad_multiple
        max_h = math.ceil(max_h / multiple) * multiple
        max_w = math.ceil(max_w / multiple) * multiple

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

        return {
            "drawing": torch.stack(drawings),
            "search_mask": torch.stack(masks),
            "hatch": torch.stack(hatches),
            "target": torch.stack(targets),
        }
