
import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=channels,
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=channels,
            ),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(
            x + self.block(x)
        )