"""Pixel template correlation used as a residual correction to decoder logits."""
import torch
from torch import nn
from torch.nn import functional as F

from .config import PixelCorrelationSettings


def _rectangular_max_pool2d(value: torch.Tensor, kernel_size: tuple[int, int]):
    """Exact rectangular max filter using two cheaper one-dimensional passes."""
    height, width = kernel_size
    horizontal = F.max_pool2d(value, (1, width), stride=1)
    return F.max_pool2d(horizontal, (height, 1), stride=1)


class PixelCorrelation(nn.Module):
    def __init__(self, settings: PixelCorrelationSettings):
        super().__init__()
        self.downsample = settings.downsample
        self.head = nn.Sequential(
            nn.Conv2d(3, settings.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(settings.hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def correlation_maps(self, drawing, hatch):
        if drawing.shape[0] != 1 or hatch.shape[0] != 1:
            raise ValueError("Pixel correlation requires batch_size=1; use gradient accumulation")
        if min(drawing.shape[-2:]) < self.downsample:
            raise ValueError("Drawing dimensions must be at least pixel_correlation.downsample")
        # Local variance needs FP32 even when the rest of the model uses BF16.
        with torch.autocast(device_type=drawing.device.type, enabled=False):
            gray = F.avg_pool2d(drawing.float().mean(1, keepdim=True), self.downsample)
            template = F.adaptive_avg_pool2d(
                hatch.float().mean(1, keepdim=True),
                tuple(max(2, size // self.downsample) for size in hatch.shape[-2:]),
            )
            scores, regions = [], []
            for rotation in range(4):
                kernel = torch.rot90(template, rotation, (-2, -1))
                height, width = kernel.shape[-2:]
                kernel = kernel - kernel.mean()
                padded = F.pad(gray, ((width-1)//2, width//2, (height-1)//2, height//2), value=1.0)
                mean = F.avg_pool2d(padded, (height, width), stride=1)
                variance = (F.avg_pool2d(padded.square(), (height, width), stride=1) - mean.square()).clamp_min(1e-7)
                numerator = F.conv2d(padded, kernel)
                denominator = (variance * height * width * kernel.square().sum()).clamp_min(1e-10).sqrt()
                score = (numerator / denominator).clamp(-1, 1)
                scores.append(score)
                # Inverse asymmetric padding preserves the crop extent for even sizes.
                padded_score = F.pad(score, (width//2, (width-1)//2, height//2, (height-1)//2), value=-1.0)
                regions.append(_rectangular_max_pool2d(padded_score, (height, width)))
            return torch.stack(scores).amax(0), torch.stack(regions).amax(0)

    def forward(self, drawing, hatch, logits):
        centers, regions = self.correlation_maps(drawing, hatch)
        base = F.interpolate(logits, size=centers.shape[-2:], mode="bilinear", align_corners=False)
        inputs = torch.cat((centers.to(logits.dtype), regions.to(logits.dtype), base), dim=1)
        correction = self.head(inputs)
        return F.interpolate(correction, size=logits.shape[-2:], mode="bilinear", align_corners=False)
