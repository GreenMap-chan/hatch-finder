import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM

from .config import ModelSettings

class HeatmapDecoder(nn.Module):
    def __init__(self, config: ModelSettings):
        super().__init__()
        encoder_channels = config.drawing_channels
        # Например encoder_channels = [32, 64, 128]

        self.input_projection = nn.Conv2d(
            config.match_feature_dims[-1],
            encoder_channels[-1],
            kernel_size=1,
        )

        self.blocks = nn.ModuleList()
        deepest_level = len(config.match_dims) - 1
        self.match_gates = nn.ParameterDict({
            str(level): nn.Parameter(torch.tensor(
                1.0 if level == deepest_level else 0.01,
            ))
            for level, match_dim in enumerate(config.match_dims)
            if match_dim > 0
        })

        current_channels = encoder_channels[-1]

        self.films = nn.ModuleDict()

        for skip_channels in reversed(encoder_channels[:-1]):
            level = len(encoder_channels) - len(self.blocks) - 2
            match_channels = (
                config.match_feature_dims[level]
                if config.match_dims[level] > 0
                else 0
            )
            block = nn.Sequential(
                nn.Conv2d(
                    current_channels + skip_channels + match_channels,
                    skip_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),

                nn.Conv2d(
                    skip_channels,
                    skip_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
            )

            self.blocks.append(block)

            if config.match_dims[level] > 0:
                self.films[str(level)] = FiLM(
                    condition_dim=config.match_dims[level],
                    channels=skip_channels,
                )

            current_channels = skip_channels

        self.output = nn.Conv2d(
            current_channels,
            1,
            kernel_size=1,
        )

        self.input_film = FiLM(
            condition_dim=config.match_dims[-1],
            channels=encoder_channels[-1],
        )

    def forward(
        self,
        matching_maps: list[torch.Tensor | None],
        encoder_features: list[torch.Tensor],
        hatch_vectors: list[torch.Tensor | None],
    ):
        # Самый глубокий уровень
        deepest_match = matching_maps[-1]
        if deepest_match is None:
            raise RuntimeError("The deepest matching level must be enabled")
        deepest_match = self.match_gates[str(len(matching_maps) - 1)] * deepest_match
        x = self.input_projection(deepest_match)

        x = self.input_film(
            x,
            hatch_vectors[-1],
        )

        skips = list(reversed(encoder_features[:-1]))
        match_skips = list(reversed(matching_maps[:-1]))
        hatch_skips = list(reversed(hatch_vectors[:-1]))
        match_levels = list(reversed(range(len(matching_maps) - 1)))

        for (
            block,
            skip,
            match,
            hatch_vector,
            level,
        ) in zip(
            self.blocks,
            skips,
            match_skips,
            hatch_skips,
            match_levels,
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if (
                match is not None
                and match.shape[-2:] != skip.shape[-2:]
            ):
                match = F.interpolate(
                    match,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            inputs = [x, skip]

            if match is not None:
                inputs.append(
                    self.match_gates[str(level)] * match
                )

            x = torch.cat(inputs, dim=1)
            x = block(x)

            film_key = str(level)
            if film_key in self.films:
                if hatch_vector is None:
                    raise RuntimeError(
                        f"Missing hatch vector for enabled FiLM level {level}"
                    )
                x = self.films[film_key](
                    x,
                    hatch_vector,
                )

        return self.output(x)
