from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


class LatentRouter(ModuleBase):
    def __init__(self, d_model: int, position_dim: int, num_actions: int, hidden_dim: int | None = None) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for routing.")
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model + position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        return self.net(torch.cat([pooled, position], dim=-1))
