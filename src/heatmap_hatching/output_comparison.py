
from typing import Any

import torch
import torch.nn as nn

from .config import ModelSettings

class OutputComparison(nn.Module):
    def __init__(self, config: ModelSettings, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.matchers = nn.ModuleDict({
            str(level): self._create_matcher(dims[0], dims[1])
            for level, dims in enumerate(zip(
                config.match_dims,
                config.match_feature_dims,
            ))
            if dims[0] > 0
        })

    def _create_matcher(
        self,
        match_dim: int,
        output_dim: int,
    ):
        layers = []
        in_channels = match_dim * 4

        for out_channels in self.config.matcher_channels:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                ),
                nn.GELU(),
            ])
            in_channels = out_channels

        layers.append(
            nn.Conv2d(
                in_channels,
                output_dim,
                kernel_size=1,
            )
        )

        return nn.Sequential(*layers)

    def _compare(
        self,
        drawing_vectors: torch.Tensor,
        hatch_vector: torch.Tensor,
        matcher: nn.Module,
    ) -> torch.Tensor:
        hatch_map = hatch_vector[:, :, None, None].expand(
            -1,
            -1,
            drawing_vectors.shape[2],
            drawing_vectors.shape[3],
        )

        comparison = torch.cat(
            [
                drawing_vectors,
                hatch_map,
                torch.abs(drawing_vectors - hatch_map),
                drawing_vectors * hatch_map,
            ],
            dim=1,
        )

        return matcher(comparison)

    def forward(
        self,
        drawing_vectors: list[torch.Tensor | None],
        hatch_vectors: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        matching_maps: list[torch.Tensor | None] = [None] * len(drawing_vectors)

        for level, (drawing_vector, hatch_vector) in enumerate(zip(
            drawing_vectors,
            hatch_vectors,
        )):
            key = str(level)
            if key not in self.matchers:
                continue
            if drawing_vector is None or hatch_vector is None:
                raise RuntimeError(f"Missing vectors for enabled matching level {level}")

            matching_maps[level] = self._compare(
                drawing_vector,
                hatch_vector,
                self.matchers[key],
            )

        return matching_maps
