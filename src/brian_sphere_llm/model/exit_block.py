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

    def __init__(self, d_model: int, position_dim: int, position_injection: str = "adapter") -> None:
        require_torch()
        super().__init__()
        self.position_injection = position_injection
        if position_injection == "adapter":
            self.position_adapter = nn.Linear(position_dim, d_model, bias=False)
        elif position_injection == "direct_add":
            if position_dim != d_model:
                raise ValueError("direct_add position_dim must equal d_model")
            self.position_adapter = None
        else:
            raise ValueError(f"Unknown block position injection: {position_injection}")
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        bias = self._position_bias(position)
        return self.proj(self.norm(hidden + bias))

    def _position_bias(self, position: torch.Tensor) -> torch.Tensor:
        if self.position_injection == "direct_add":
            return position.unsqueeze(1)
        return self.position_adapter(position).unsqueeze(1)
