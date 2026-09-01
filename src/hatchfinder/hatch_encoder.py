import torch
import torch.nn as nn

from .residual_block import ResidualBlock

from .config import ModelSettings



class HatchEncoder(nn.Module):
    def __init__(self, config: ModelSettings):
        super().__init__()

        self.blocks = nn.ModuleList()

        in_channels = 3

        for i, out_channels in enumerate(config.hatch_channels):
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=1 if i == 0 else config.downsample_stride,
                    padding=config.kernel_size // 2,
                ),
                nn.GroupNorm(
                    num_groups=config.group_norm_groups_hatchings,
                    num_channels=out_channels,
                ),
                nn.GELU(),

                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=config.kernel_size // 2,
                ),
                nn.GroupNorm(
                    num_groups=config.group_norm_groups_hatchings,
                    num_channels=out_channels,
                ),
                nn.GELU(),
            ]

            for _ in range(config.hatch_residual_blocks[i]):
                layers.append(
                    ResidualBlock(
                        channels=out_channels,
                        num_groups=config.group_norm_groups_hatchings,
                    )
                )

            self.blocks.append(nn.Sequential(*layers))
            in_channels = out_channels

    def forward(self, hatch: torch.Tensor):
        features = []

        x = hatch

        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features
