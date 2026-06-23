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

    def __init__(
        self,
        d_model: int,
        code_dim: int,
        *,
        n_heads: int | None = None,
        head_delta_rank: int = 0,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for GlobalWriteAdapter.")
        super().__init__()
        self.d_model = d_model
        self.code_dim = code_dim
        self.n_heads = int(n_heads or 1)
        self.head_delta_rank = int(head_delta_rank)
        if self.head_delta_rank < 0:
            raise ValueError("head_delta_rank must be non-negative.")
        if self.head_delta_rank and d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for per-head global write deltas.")
        self.head_dim = d_model // self.n_heads if self.head_delta_rank else d_model
        self.proj = nn.Linear(d_model, code_dim, bias=False)
        if self.head_delta_rank:
            self.head_delta_down = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.head_delta_rank))
            self.head_delta_up = nn.Parameter(torch.empty(self.n_heads, self.head_delta_rank, code_dim))
            nn.init.normal_(self.head_delta_down, std=0.02)
            nn.init.zeros_(self.head_delta_up)

    def forward(self, hidden: "torch.Tensor", *, tokenwise: bool = True) -> "torch.Tensor":
        source = hidden if tokenwise else hidden.mean(dim=1)
        code = self.proj(source)
        if self.head_delta_rank:
            if tokenwise:
                heads = source.reshape(source.size(0), source.size(1), self.n_heads, self.head_dim)
                delta = torch.einsum("bthd,hdr->bthr", heads, self.head_delta_down)
                delta = torch.einsum("bthr,hrc->bthc", delta, self.head_delta_up).sum(dim=2)
            else:
                heads = source.reshape(source.size(0), self.n_heads, self.head_dim)
                delta = torch.einsum("bhd,hdr->bhr", heads, self.head_delta_down)
                delta = torch.einsum("bhr,hrc->bhc", delta, self.head_delta_up).sum(dim=1)
            code = code + delta / (self.n_heads**0.5)
        return F.normalize(code, dim=-1)
