from pathlib import Path

import torch

from .config import (
    Config, ModelSettings, PT_FORMAT_VERSION, config_from_pt_data,
    model_config_from_pt_data,
)


CHECKPOINT_PATH = Path("runs/best.pt")
MODEL_PATH = Path("runs/model.pt")


def model_state_dict(data: object) -> dict[str, torch.Tensor]:
    if isinstance(data, dict) and "model_state_dict" in data:
        return data["model_state_dict"]
    if isinstance(data, dict):
        return data
    raise ValueError("The .pt file does not contain a model state_dict")


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, patience_counter, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "patience_counter": patience_counter,
            "config": model.config.model_dump(mode="json"),
            "format_version": PT_FORMAT_VERSION,
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler, path) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=model.device, weights_only=True)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        saved_model = ModelSettings.model_validate(checkpoint_config.get("model"))
        if saved_model != model.config.model:
            raise ValueError("Checkpoint model configuration does not match the current config")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_metric"],
        checkpoint["patience_counter"],
    )


def save_weights(model, path: Path):
    torch.save(
        {
            "format_version": PT_FORMAT_VERSION,
            "model_config": model.config.model.model_dump(mode="json"),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(model, path: Path):
    data = torch.load(path, map_location=model.device, weights_only=True)
    saved_model = model_config_from_pt_data(data)
    if saved_model is not None and saved_model != model.config.model:
        raise ValueError("Weights model configuration does not match the current model")
    model.load_state_dict(model._get_model_state_dict(data))


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
