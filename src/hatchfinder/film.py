import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        channels: int,
    ):
        super().__init__()

        self.projection = nn.Linear(
            condition_dim,
            channels * 2,
        )

        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ):
        gamma, beta = self.projection(condition).chunk(2, dim=1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return x * (1 + gamma) + beta