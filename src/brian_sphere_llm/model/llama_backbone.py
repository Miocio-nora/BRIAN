from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - imported in environments without torch
    torch = None
    nn = None
    F = None

if torch is not None:  # pragma: no cover - import path is exercised through model tests
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
else:  # pragma: no cover
    torch_checkpoint = None

ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model code. Install the B200/cu128 environment first.")


def checkpoint_if_enabled(owner: nn.Module, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    if bool(getattr(owner, "activation_checkpointing", False)) and owner.training and torch.is_grad_enabled():
        if torch_checkpoint is None:  # pragma: no cover
            raise ModuleNotFoundError("PyTorch checkpointing is unavailable.")
        return torch_checkpoint(module, *args, use_reentrant=False)
    return module(*args)


@dataclass(frozen=True)
class BackboneConfig:
    vocab_size: int
    context_length: int
    layers: int
    d_model: int
    n_heads: int
    dropout: float = 0.0
    ffn_multiplier: float = 4.0
    attention_global_logit_bias_init: float | None = None
    attention_global_sink_slots: int = 0
    attention_global_kv_mode: str = "summary"
    attention_global_code_dim: int | None = None


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
        self.attention_global_sink_slots = int(config.attention_global_sink_slots)
        self.attention_global_kv_mode = str(config.attention_global_kv_mode)
        if self.attention_global_kv_mode not in {"summary", "token_compressed"}:
            raise ValueError(f"Unknown attention_global_kv_mode: {self.attention_global_kv_mode}")
        self.attention_global_code_dim = int(config.attention_global_code_dim or self.head_dim)
        self.global_logit_bias = (
            nn.Parameter(torch.tensor(float(config.attention_global_logit_bias_init)))
            if config.attention_global_logit_bias_init is not None
            else None
        )
        if self.attention_global_kv_mode == "token_compressed":
            self.global_key_write = nn.Linear(self.head_dim, self.attention_global_code_dim, bias=False)
            self.global_value_write = nn.Linear(self.head_dim, self.attention_global_code_dim, bias=False)
            self.global_key_read = nn.Linear(self.attention_global_code_dim, self.head_dim, bias=False)
            self.global_value_read = nn.Linear(self.attention_global_code_dim, self.head_dim, bias=False)
        else:
            self.global_key_write = None
            self.global_value_write = None
            self.global_key_read = None
            self.global_value_read = None

    def forward(
        self,
        x: torch.Tensor,
        attention_global_state: Any | None = None,
        *,
        return_attention_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        write_key, write_value = self._attention_global_write_tensors(k, v)
        global_keys = getattr(attention_global_state, "keys", None)
        global_values = getattr(attention_global_state, "values", None)
        global_valid = getattr(attention_global_state, "valid", None)
        has_global = global_keys is not None and int(global_keys.size(2)) > 0
        if has_global:
            global_k, global_v = self._attention_global_read_tensors(global_keys, global_values, dtype=k.dtype)
            all_k = torch.cat([global_k, k], dim=2)
            all_v = torch.cat([global_v, v], dim=2)
            attn_mask = self._global_prefix_mask(global_valid, seq_len, dtype=q.dtype, device=q.device)
            y = F.scaled_dot_product_attention(
                q,
                all_k,
                all_v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
            metrics = self._global_attention_metrics(q, k, global_k, global_valid)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            metrics = self._empty_global_attention_metrics(x)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        output = self.out(y)
        if return_attention_kv:
            return output, write_key, write_value, metrics
        return output

    def _attention_global_write_tensors(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_global_kv_mode == "token_compressed":
            assert self.global_key_write is not None
            assert self.global_value_write is not None
            return self.global_key_write(k), self.global_value_write(v)
        return k.mean(dim=2), v.mean(dim=2)

    def _attention_global_read_tensors(
        self,
        global_keys: torch.Tensor,
        global_values: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_global_kv_mode == "token_compressed":
            assert self.global_key_read is not None
            assert self.global_value_read is not None
            keys = self.global_key_read(global_keys.to(dtype=dtype))
            values = self.global_value_read(global_values.to(dtype=dtype))
            return keys, values
        return global_keys.to(dtype=dtype), global_values.to(dtype=dtype)

    def _global_prefix_mask(
        self,
        global_valid: torch.Tensor,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        slots = int(global_valid.size(1))
        local = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        allowed = torch.cat(
            [
                global_valid.to(device=device, dtype=torch.bool)[:, None, None, :].expand(-1, 1, seq_len, slots),
                local[None, None, :, :].expand(global_valid.size(0), 1, -1, -1),
            ],
            dim=-1,
        )
        mask = torch.zeros(global_valid.size(0), 1, seq_len, slots + seq_len, dtype=dtype, device=device)
        if slots:
            mask[:, :, :, :slots] = self._global_logit_bias(dtype=dtype, device=device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _global_attention_metrics(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        global_keys: torch.Tensor,
        global_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        slots = int(global_keys.size(2))
        if slots <= 0:
            return self._empty_global_attention_metrics(q)
        with torch.no_grad():
            query = q[:, :, -1:, :].detach()
            global_scores = torch.matmul(query, global_keys.detach().transpose(-2, -1)) * (self.head_dim**-0.5)
            global_scores = global_scores + self._global_logit_bias(dtype=global_scores.dtype, device=global_scores.device)
            invalid = ~global_valid.to(device=global_scores.device, dtype=torch.bool)[:, None, None, :]
            global_scores = global_scores.masked_fill(invalid, torch.finfo(global_scores.dtype).min)
            local_scores = torch.matmul(query, local_k.detach().transpose(-2, -1)) * (self.head_dim**-0.5)
            weights = torch.softmax(torch.cat([global_scores, local_scores], dim=-1), dim=-1)
            global_weights = weights[..., :slots]
            sink_slots = int(getattr(self, "attention_global_sink_slots", 0))
            sink_count = max(0, min(sink_slots, slots))
            sink_mass = global_weights[..., :sink_count].sum(dim=-1).mean() if sink_count else global_weights.new_zeros(())
            window_mass = global_weights[..., sink_count:].sum(dim=-1).mean()
            return {
                "attention_global_kv_last_token_mass": global_weights.sum(dim=-1).mean(),
                "attention_global_kv_sink_last_token_mass": sink_mass,
                "attention_global_kv_window_last_token_mass": window_mass,
                "attention_global_kv_logit_bias": self._global_logit_bias(
                    dtype=global_weights.dtype,
                    device=global_weights.device,
                ),
            }

    def _empty_global_attention_metrics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return {
            "attention_global_kv_last_token_mass": zero,
            "attention_global_kv_sink_last_token_mass": zero,
            "attention_global_kv_window_last_token_mass": zero,
            "attention_global_kv_logit_bias": self._global_logit_bias(dtype=x.dtype, device=x.device),
        }

    def _global_logit_bias(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.global_logit_bias is None:
            return torch.zeros((), device=device, dtype=dtype)
        return self.global_logit_bias.to(device=device, dtype=dtype)


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

    def forward(
        self,
        x: torch.Tensor,
        attention_global_state: Any | None = None,
        *,
        return_attention_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        attn_input = self.attn_norm(x)
        if return_attention_kv:
            attn_out, key_summary, value_summary, metrics = self.attn(
                attn_input,
                attention_global_state,
                return_attention_kv=True,
            )
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))
            return x, key_summary, value_summary, metrics
        x = x + self.attn(attn_input, attention_global_state)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def build_causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=ignore_index)


def count_parameters(module: nn.Module) -> int:
    require_torch()
    return sum(parameter.numel() for parameter in module.parameters())
