import torch
import torch.nn as nn

from .residual_block import ResidualBlock
from .spatial_transformer import SpatialTransformer
from .config import ModelSettings


class DrawingEncoder(nn.Module):
    def __init__(self, config: ModelSettings):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.transformers = nn.ModuleDict()

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
                        kernel_size=config.kernel_size,
                    )
                )

            self.blocks.append(nn.Sequential(*layers))
            in_channels = out_channels

        # Transformer-блоки для выбранных уровней
        for transformer_config in config.transformers:
            level = transformer_config.level

            self.transformers[str(level)] = SpatialTransformer(
                channels=config.drawing_channels[level],
                num_heads=transformer_config.num_heads,
                num_blocks=transformer_config.num_blocks,
                downsample=transformer_config.downsample,
                mlp_ratio=transformer_config.mlp_ratio,
                dropout=transformer_config.dropout,
                initial_gate=transformer_config.initial_gate,
            )

    def forward(
        self,
        drawing: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:

        x = torch.cat([drawing, mask], dim=1)

        features = []

        for level, block in enumerate(self.blocks):
            x = block(x)

            transformer_key = str(level)

            if transformer_key in self.transformers:
                x = self.transformers[transformer_key](x)

            features.append(x)

        return features