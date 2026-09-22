import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_blocks: int = 1,
        downsample: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        initial_gate: float = 0.01,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if channels % 4 != 0:
            raise ValueError(
                "channels must be divisible by 4 for 2D positional encoding"
            )

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        if downsample < 1:
            raise ValueError("downsample must be >= 1")

        self.channels = channels
        self.downsample = downsample

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        # Transformer изначально не влияет на CNN features.
        self.gate = nn.Parameter(
            torch.tensor(float(initial_gate))
        )

    def _positional_encoding(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        quarter_dim = self.channels // 4

        y = torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        )

        x = torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )

        frequencies = torch.exp(
            torch.arange(
                quarter_dim,
                device=device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / max(quarter_dim - 1, 1))
        )

        y = y[:, None] * frequencies[None, :]
        x = x[:, None] * frequencies[None, :]

        y_encoding = torch.cat(
            [torch.sin(y), torch.cos(y)],
            dim=1,
        )

        x_encoding = torch.cat(
            [torch.sin(x), torch.cos(x)],
            dim=1,
        )

        y_encoding = y_encoding[:, None, :].expand(
            height,
            width,
            -1,
        )

        x_encoding = x_encoding[None, :, :].expand(
            height,
            width,
            -1,
        )

        encoding = torch.cat(
            [y_encoding, x_encoding],
            dim=-1,
        )

        encoding = encoding.reshape(
            1,
            height * width,
            self.channels,
        )

        return encoding.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        # [B, C, H, W] -> [B, C, h, w]
        if self.downsample > 1:
            x = F.avg_pool2d(
                x,
                kernel_size=self.downsample,
                stride=self.downsample,
            )

        batch_size, channels, height, width = x.shape

        # [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)

        positional_encoding = self._positional_encoding(
            height=height,
            width=width,
            device=x.device,
            dtype=x.dtype,
        )

        x = x + positional_encoding

        # [B, N, C] -> [B, N, C]
        x = self.transformer(x)

        # [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        # Возвращаем исходное spatial resolution.
        if self.downsample > 1:
            x = F.interpolate(
                x,
                size=identity.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return identity + self.gate * x