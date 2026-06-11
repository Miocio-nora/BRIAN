from __future__ import annotations

from brian_sphere_llm.model.llama_backbone import RMSNorm, require_torch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


class ExitBlock(ModuleBase):
    """Terminal operator before post blocks / LM head."""

    def __init__(self, d_model: int, position_dim: int) -> None:
        require_torch()
        super().__init__()
        self.position_adapter = nn.Linear(position_dim, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        bias = self.position_adapter(position).unsqueeze(1)
        return self.proj(self.norm(hidden + bias))
