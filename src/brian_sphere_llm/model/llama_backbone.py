from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - imported in environments without torch
    torch = None
    nn = None
    F = None

ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model code. Install the B200/cu128 environment first.")


@dataclass(frozen=True)
class BackboneConfig:
    vocab_size: int
    context_length: int
    layers: int
    d_model: int
    n_heads: int
    dropout: float = 0.0
    ffn_multiplier: float = 4.0


class RMSNorm(ModuleBase):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        require_torch()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class RotaryEmbedding(ModuleBase):
    def __init__(self, dim: int, max_position: int, base: float = 10000.0) -> None:
        require_torch()
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(ModuleBase):
    def __init__(self, config: BackboneConfig) -> None:
        require_torch()
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)


class SwiGLU(ModuleBase):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        require_torch()
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(ModuleBase):
    def __init__(self, config: BackboneConfig) -> None:
        require_torch()
        super().__init__()
        hidden_dim = int(math.ceil(config.ffn_multiplier * config.d_model / 256) * 256)
        if config.d_model < 256:
            hidden_dim = int(config.ffn_multiplier * config.d_model)
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


def build_causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=ignore_index)


def count_parameters(module: nn.Module) -> int:
    require_torch()
    return sum(parameter.numel() for parameter in module.parameters())
