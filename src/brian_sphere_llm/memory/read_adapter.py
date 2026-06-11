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


class GlobalReadAdapter(ModuleBase):
    """Read canonical global codes back into local hidden space."""

    def __init__(self, d_model: int, code_dim: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for GlobalReadAdapter.")
        super().__init__()
        self.query = nn.Linear(d_model, code_dim, bias=False)
        self.out = nn.Linear(code_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, hidden: "torch.Tensor", codes: "torch.Tensor") -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        if codes.size(1) == 0:
            zero = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
            return hidden, {"global_attention_mass": zero, "global_read_gate": torch.sigmoid(self.gate).to(hidden.dtype)}
        pooled = hidden.mean(dim=1)
        query = self.query(pooled)
        scale = query.size(-1) ** -0.5
        attn = F.softmax(torch.einsum("bd,bnd->bn", query, codes) * scale, dim=-1)
        read_code = torch.einsum("bn,bnd->bd", attn, codes)
        gate = torch.sigmoid(self.gate).to(hidden.dtype)
        updated = hidden + gate * self.out(read_code).unsqueeze(1)
        return updated, {
            "global_attention_mass": attn.sum(dim=-1).mean(),
            "global_read_gate": gate,
        }
