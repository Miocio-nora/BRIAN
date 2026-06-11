from __future__ import annotations

from brian_sphere_llm.model.llama_backbone import BackboneConfig, TransformerBlock, require_torch

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

ModuleBase = nn.Module if nn is not None else object


class RouteBlock(ModuleBase):
    """Transformer block with a controlled block-position side channel."""

    def __init__(self, backbone: BackboneConfig, position_dim: int) -> None:
        require_torch()
        super().__init__()
        self.block = TransformerBlock(backbone)
        self.position_adapter = nn.Linear(position_dim, backbone.d_model, bias=False)
        nn.init.zeros_(self.position_adapter.weight)

    def forward(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        bias = self.position_adapter(position).unsqueeze(1)
        return self.block(hidden + bias)
