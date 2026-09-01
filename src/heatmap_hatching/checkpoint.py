from pathlib import Path

import torch

from .config import Config, PT_FORMAT_VERSION, config_from_pt_data


CHECKPOINT_PATH = Path("runs/best.pt")
MODEL_PATH = Path("runs/model.pt")


def convert_checkpoint_to_model(
    checkpoint_path: Path,
    model_path: Path,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Файл {checkpoint_path} не является чекпоинтом ожидаемого формата"
        )

    config = config_from_pt_data(checkpoint)
    if config is None:
        config = Config()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": PT_FORMAT_VERSION,
            "model_config": config.model.model_dump(mode="json"),
            "model_state_dict": checkpoint["model_state_dict"],
        },
        model_path,
    )
    print(f"Модель сохранена: {model_path}")


if __name__ == "__main__":
    convert_checkpoint_to_model(CHECKPOINT_PATH, MODEL_PATH)
