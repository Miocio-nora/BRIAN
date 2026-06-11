from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None

ModuleBase = nn.Module if nn is not None else object


class GlobalWriteAdapter(ModuleBase):
    """Compress local hidden states into canonical global memory codes."""

    def __init__(self, d_model: int, code_dim: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for GlobalWriteAdapter.")
        super().__init__()
        self.proj = nn.Linear(d_model, code_dim, bias=False)

    def forward(self, hidden: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(self.proj(hidden.mean(dim=1)), dim=-1)
