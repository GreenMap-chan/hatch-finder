import torch
import torch.nn as nn
from .residual_block import ResidualBlock

from .config import ModelSettings

class DrawingEncoder(nn.Module):
    def __init__(self, config: ModelSettings):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_channels = 4  # RGB + mask

        for i, out_channels in enumerate(config.drawing_channels):
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=1 if i == 0 else config.downsample_stride,
                    padding=config.kernel_size // 2,
                ),
                nn.GroupNorm(config.group_norm_groups_drawings, out_channels),
                nn.GELU(),

                # Вторая Conv не уменьшает разрешение
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=config.kernel_size // 2,
                ),
                nn.GroupNorm(config.group_norm_groups_drawings, out_channels),
                nn.GELU(),
            ]

            for _ in range(config.drawing_residual_blocks[i]):
                layers.append(
                    ResidualBlock(
                        channels=out_channels,
                        num_groups=config.group_norm_groups_drawings,
                    )
                )

            block = nn.Sequential(*layers)

            self.blocks.append(block)
            in_channels = out_channels

    def forward(self, drawing, mask):
        x = torch.cat([drawing, mask], dim=1)

        features = []

        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features
