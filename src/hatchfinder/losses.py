import torch
import torch.nn.functional as F


def get_dice(logits: torch.Tensor, target_tensor: torch.Tensor, mask: torch.Tensor):
    probabilities = torch.sigmoid(logits) * mask
    target = target_tensor * mask
    non_empty = target.sum(dim=(1, 2, 3)) > 0
    if not non_empty.any():
        return None

    probabilities = probabilities[non_empty]
    target = target[non_empty]
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    smooth = 1e-6
    dice = (2 * intersection + smooth) / (
        probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    )
    return 1 - dice.mean()


def get_loss(
    logits: torch.Tensor,
    target_tensor: torch.Tensor,
    mask: torch.Tensor,
    dice_fn=get_dice,
):
    if logits.shape != target_tensor.shape or logits.shape != mask.shape:
        raise ValueError(
            "logits, target_tensor and mask must have identical shapes, got "
            f"{tuple(logits.shape)}, {tuple(target_tensor.shape)} and {tuple(mask.shape)}"
        )

    target_tensor = target_tensor.to(logits.device, non_blocking=True)
    mask = mask.to(logits.device, non_blocking=True)
    loss_map = F.binary_cross_entropy_with_logits(
        logits, target_tensor, reduction="none"
    )
    bce_loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
    dice_loss = dice_fn(logits, target_tensor, mask)
    result_loss = bce_loss if dice_loss is None else bce_loss + dice_loss
    return result_loss, bce_loss, dice_loss
