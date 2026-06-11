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

    def __init__(
        self,
        d_model: int,
        code_dim: int,
        *,
        n_heads: int | None = None,
        head_delta_rank: int = 0,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for GlobalReadAdapter.")
        super().__init__()
        self.d_model = d_model
        self.code_dim = code_dim
        self.n_heads = int(n_heads or 1)
        self.head_delta_rank = int(head_delta_rank)
        if self.head_delta_rank < 0:
            raise ValueError("head_delta_rank must be non-negative.")
        if self.head_delta_rank and d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for per-head global read deltas.")
        self.head_dim = d_model // self.n_heads if self.head_delta_rank else d_model
        self.query = nn.Linear(d_model, code_dim, bias=False)
        self.out = nn.Linear(code_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(-4.0))
        if self.head_delta_rank:
            self.head_delta_down = nn.Parameter(torch.empty(self.n_heads, code_dim, self.head_delta_rank))
            self.head_delta_up = nn.Parameter(torch.empty(self.n_heads, self.head_delta_rank, self.head_dim))
            nn.init.normal_(self.head_delta_down, std=0.02)
            nn.init.zeros_(self.head_delta_up)

    def forward(
        self,
        hidden: "torch.Tensor",
        codes: "torch.Tensor",
        *,
        sink_slots: int = 0,
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        if codes.size(1) == 0:
            zero = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
            return hidden, {
                "global_attention_mass": zero,
                "global_sink_attention_mass": zero,
                "global_window_attention_mass": zero,
                "global_read_gate": torch.sigmoid(self.gate).to(hidden.dtype),
            }
        pooled = hidden.mean(dim=1)
        query = self.query(pooled)
        scale = query.size(-1) ** -0.5
        attn = F.softmax(torch.einsum("bd,bnd->bn", query, codes) * scale, dim=-1)
        read_code = torch.einsum("bn,bnd->bd", attn, codes)
        gate = torch.sigmoid(self.gate).to(hidden.dtype)
        read_hidden = self.out(read_code)
        if self.head_delta_rank:
            delta = torch.einsum("bd,hdr->bhr", read_code, self.head_delta_down)
            delta = torch.einsum("bhr,hrm->bhm", delta, self.head_delta_up)
            read_hidden = read_hidden + delta.reshape(read_code.size(0), self.d_model)
        updated = hidden + gate * read_hidden.unsqueeze(1)
        sink_count = max(0, min(int(sink_slots), codes.size(1)))
        sink_mass = (
            attn[:, :sink_count].sum(dim=-1).mean()
            if sink_count
            else torch.zeros((), device=hidden.device, dtype=attn.dtype)
        )
        window_mass = attn[:, sink_count:].sum(dim=-1).mean()
        return updated, {
            "global_attention_mass": attn.sum(dim=-1).mean(),
            "global_sink_attention_mass": sink_mass,
            "global_window_attention_mass": window_mass,
            "global_read_gate": gate,
        }
