import torch
from torch import nn
from torch.nn import functional as F

from .config import CrossAttentionSettings


class CrossAttention(nn.Module):
    """Match drawing queries to pooled hatch tokens without a global collapse."""

    def __init__(
        self,
        hatch_channels: int,
        match_dim: int,
        output_dim: int,
        pool_size: int,
        settings: CrossAttentionSettings,
    ):
        super().__init__()
        self.downsample = settings.downsample
        self.hatch_projection = nn.Linear(hatch_channels, match_dim)
        self.hatch_positions = nn.Parameter(torch.empty(1, pool_size * pool_size, match_dim))
        nn.init.normal_(self.hatch_positions, std=0.02)
        self.query_norm = nn.LayerNorm(match_dim)
        self.hatch_norm = nn.LayerNorm(match_dim)
        self.attention = nn.MultiheadAttention(
            match_dim, settings.num_heads, dropout=settings.dropout, batch_first=True,
        )
        self.matcher = nn.Sequential(
            nn.Linear(match_dim * 4, match_dim),
            nn.GELU(),
            nn.Linear(match_dim, output_dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(settings.initial_gate)))

    def forward(self, drawing: torch.Tensor, pooled_hatch: torch.Tensor) -> torch.Tensor:
        spatial_size = drawing.shape[-2:]
        if self.downsample > 1:
            drawing = F.avg_pool2d(
                drawing,
                kernel_size=tuple(min(self.downsample, size) for size in spatial_size),
                stride=self.downsample,
            )
        batch, _, height, width = drawing.shape
        queries = drawing.flatten(2).transpose(1, 2)
        tokens = self.hatch_projection(pooled_hatch.flatten(2).transpose(1, 2))
        tokens = self.hatch_norm(tokens + self.hatch_positions.to(dtype=tokens.dtype))
        context, _ = self.attention(
            self.query_norm(queries), tokens, tokens, need_weights=False,
        )
        comparison = torch.cat(
            (queries, context, (queries - context).abs(), queries * context), dim=-1,
        )
        result = self.matcher(comparison).transpose(1, 2).reshape(batch, -1, height, width)
        if self.downsample > 1:
            result = F.interpolate(result, size=spatial_size, mode="bilinear", align_corners=False)
        return self.gate * result
