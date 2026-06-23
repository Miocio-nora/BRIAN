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
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except ImportError:  # pragma: no cover - depends on PyTorch build
        create_block_mask = None
        flex_attention = None
else:  # pragma: no cover
    torch_checkpoint = None
    create_block_mask = None
    flex_attention = None

ModuleBase = nn.Module if nn is not None else object
_compiled_flex_attention = None


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model code. Install the B200/cu128 environment first.")


def checkpoint_if_enabled(owner: nn.Module, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    if bool(getattr(owner, "activation_checkpointing", False)) and owner.training and torch.is_grad_enabled():
        if torch_checkpoint is None:  # pragma: no cover
            raise ModuleNotFoundError("PyTorch checkpointing is unavailable.")
        return torch_checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def flex_attention_if_available(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_mask: Any,
) -> torch.Tensor:
    if flex_attention is None:
        raise ModuleNotFoundError("PyTorch FlexAttention is required for sparse_varlen route block execution.")
    if query.is_cuda:
        global _compiled_flex_attention
        if _compiled_flex_attention is None:
            _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
        return _compiled_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={"ROWS_GUARANTEED_SAFE": True},
        )
    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        kernel_options={"ROWS_GUARANTEED_SAFE": True},
    )


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

    def apply_to_positions(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[0, 0, positions, :].to(device=x.device)
        sin = self.sin[0, 0, positions, :].to(device=x.device)
        return apply_rotary(x, cos.unsqueeze(1), sin.unsqueeze(1))


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
        if self.attention_global_kv_mode not in {"summary", "token_compressed", "pure_factorized"}:
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
        elif self.attention_global_kv_mode == "pure_factorized":
            self.global_key_write = nn.Linear(config.d_model, self.attention_global_code_dim, bias=False)
            self.global_value_write = nn.Linear(config.d_model, self.attention_global_code_dim, bias=False)
            self.global_key_read = None
            self.global_value_read = None
            self.global_key_head_read = nn.Parameter(
                torch.empty(self.n_heads, self.head_dim, self.attention_global_code_dim)
            )
            self.global_value_head_read = nn.Parameter(
                torch.empty(self.n_heads, self.attention_global_code_dim, self.head_dim)
            )
            nn.init.xavier_uniform_(self.global_key_head_read)
            nn.init.xavier_uniform_(self.global_value_head_read)
        else:
            self.global_key_write = None
            self.global_value_write = None
            self.global_key_read = None
            self.global_value_read = None
            self.global_key_head_read = None
            self.global_value_head_read = None

    def forward(
        self,
        x: torch.Tensor,
        attention_global_state: Any | None = None,
        *,
        return_attention_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, dim = x.shape
        if self.attention_global_kv_mode == "pure_factorized":
            return self._forward_pure_factorized_global(
                x,
                attention_global_state,
                return_attention_kv=return_attention_kv,
            )
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
            if global_k.dim() == 5:
                global_k_for_attention = self._flatten_token_global_kv(global_k)
                global_v_for_attention = self._flatten_token_global_kv(global_v)
            else:
                global_k_for_attention = global_k
                global_v_for_attention = global_v
            all_k = torch.cat([global_k_for_attention, k], dim=2)
            all_v = torch.cat([global_v_for_attention, v], dim=2)
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

    def tie_pure_factorized_writers_from(self, other: "CausalSelfAttention") -> None:
        if self.attention_global_kv_mode != "pure_factorized":
            raise ValueError("pure factorized writer sharing is only valid for pure_factorized mode.")
        if other.attention_global_kv_mode != "pure_factorized":
            raise ValueError("shared writer source must also use pure_factorized mode.")
        self.global_key_write = other.global_key_write
        self.global_value_write = other.global_value_write

    def _forward_pure_factorized_global(
        self,
        x: torch.Tensor,
        attention_global_state: Any | None = None,
        *,
        return_attention_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, dim = x.shape
        assert self.global_key_write is not None
        assert self.global_value_write is not None
        assert self.global_key_head_read is not None
        assert self.global_value_head_read is not None

        q_weight = self.qkv.weight[:dim, :]
        q = F.linear(x, q_weight)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_code = torch.einsum("bhqd,hdc->bhqc", q, self.global_key_head_read.to(dtype=q.dtype))

        current_key_code = self.global_key_write(x)
        current_value_code = self.global_value_write(x)
        global_keys = getattr(attention_global_state, "keys", None)
        global_values = getattr(attention_global_state, "values", None)
        global_valid = getattr(attention_global_state, "valid", None)

        previous_key_code, previous_value_code, previous_allowed = self._pure_factorized_previous_pool(
            global_keys,
            global_values,
            global_valid,
            batch=batch,
            seq_len=seq_len,
            dtype=x.dtype,
            device=x.device,
        )
        current_allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        current_allowed = current_allowed.unsqueeze(0).expand(batch, -1, -1)
        key_code = torch.cat([previous_key_code, current_key_code], dim=1)
        value_code = torch.cat([previous_value_code, current_value_code], dim=1)
        allowed = torch.cat([previous_allowed, current_allowed], dim=-1)

        previous_key_count = previous_key_code.size(1)
        attn_mask = torch.zeros(batch, 1, seq_len, key_code.size(1), dtype=q_code.dtype, device=x.device)
        if previous_key_count:
            attn_mask[..., :previous_key_count] = self._global_logit_bias(dtype=q_code.dtype, device=x.device)
        attn_mask = attn_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(q_code.dtype).min)
        key_for_attention = key_code[:, None, :, :].expand(-1, self.n_heads, -1, -1)
        value_for_attention = value_code[:, None, :, :].expand(-1, self.n_heads, -1, -1)
        value_code_out = F.scaled_dot_product_attention(
            q_code,
            key_for_attention,
            value_for_attention,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.attention_global_code_dim**-0.5,
        )
        y = torch.einsum("bhqc,hcd->bhqd", value_code_out, self.global_value_head_read.to(dtype=value_code_out.dtype))
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        output = self.out(y)

        metrics = self._pure_factorized_metrics(q_code, key_code, allowed, previous_key_count, x)
        if return_attention_kv:
            return output, current_key_code.unsqueeze(1), current_value_code.unsqueeze(1), metrics
        return output

    def _pure_factorized_previous_pool(
        self,
        global_keys: torch.Tensor | None,
        global_values: torch.Tensor | None,
        global_valid: torch.Tensor | None,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_keys is None or global_values is None or global_valid is None or int(global_keys.size(2)) == 0:
            return (
                torch.empty(batch, 0, self.attention_global_code_dim, device=device, dtype=dtype),
                torch.empty(batch, 0, self.attention_global_code_dim, device=device, dtype=dtype),
                torch.empty(batch, seq_len, 0, device=device, dtype=torch.bool),
            )
        if global_keys.dim() != 5 or global_values.dim() != 5:
            raise ValueError("pure_factorized Attention Global KV requires token-shaped rank-5 cache state.")
        batch, heads, slots, global_seq_len, code_dim = global_keys.shape
        if heads != 1:
            raise ValueError("pure_factorized global pool stores shared headless key/value codes.")
        key_code = global_keys.to(device=device, dtype=dtype).reshape(batch, slots * global_seq_len, code_dim)
        value_code = global_values.to(device=device, dtype=dtype).reshape(batch, slots * global_seq_len, code_dim)
        valid = global_valid.to(device=device, dtype=torch.bool)
        query_pos = torch.arange(seq_len, device=device)
        global_pos = torch.arange(global_seq_len, device=device)
        causal = global_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        allowed = (valid[:, None, :, :] & causal[None, :, None, :]).reshape(batch, seq_len, slots * global_seq_len)
        return key_code, value_code, allowed

    def _pure_factorized_metrics(
        self,
        q_code: torch.Tensor,
        key_code: torch.Tensor,
        allowed: torch.Tensor,
        previous_key_count: int,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if previous_key_count <= 0:
            return self._empty_global_attention_metrics(x)
        with torch.no_grad():
            scores = torch.einsum("bhc,bkc->bhk", q_code[:, :, -1, :], key_code) * (
                self.attention_global_code_dim**-0.5
            )
            scores[:, :, :previous_key_count] = scores[:, :, :previous_key_count] + self._global_logit_bias(
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = scores.masked_fill(~allowed[:, None, -1, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            global_weights = weights[..., :previous_key_count]
            total_mass = global_weights.sum(dim=-1).mean()
            zero = total_mass.new_zeros(())
            return {
                "attention_global_kv_last_token_mass": total_mass,
                "attention_global_kv_sink_last_token_mass": zero,
                "attention_global_kv_window_last_token_mass": total_mass,
                "attention_global_kv_logit_bias": self._global_logit_bias(dtype=weights.dtype, device=weights.device),
            }

    def forward_selected(self, x: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if query_mask.shape != (batch, seq_len):
            raise ValueError("query_mask must have shape [batch, seq_len].")
        batch_indices, query_positions = torch.where(query_mask)
        if batch_indices.numel() == 0:
            return x.new_empty((0, dim))

        q_weight = self.qkv.weight[:dim, :]
        kv_weight = self.qkv.weight[dim:, :]
        q = F.linear(x[query_mask], q_weight)
        kv = F.linear(x, kv_weight)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope.cos[:, :, :seq_len, :].to(device=k.device)
        sin = self.rope.sin[:, :, :seq_len, :].to(device=k.device)
        k = apply_rotary(k, cos, sin)
        q = self.rope.apply_to_positions(q, query_positions)

        counts = query_mask.sum(dim=1)
        max_selected = int(counts.max().item())
        offsets = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        local_indices = torch.arange(q.size(0), device=x.device) - offsets

        q_padded = q.new_zeros((batch, max_selected, self.n_heads, self.head_dim))
        query_positions_padded = torch.zeros(batch, max_selected, dtype=torch.long, device=x.device)
        query_valid = torch.zeros(batch, max_selected, dtype=torch.bool, device=x.device)
        q_padded[batch_indices, local_indices] = q
        query_positions_padded[batch_indices, local_indices] = query_positions
        query_valid[batch_indices, local_indices] = True

        key_positions = torch.arange(seq_len, device=x.device)
        attn_mask = (key_positions.view(1, 1, seq_len) <= query_positions_padded.unsqueeze(-1)).unsqueeze(1)
        y = F.scaled_dot_product_attention(
            q_padded.transpose(1, 2),
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        selected = y.transpose(1, 2)[query_valid].to(dtype=q.dtype)
        return self.out(selected.reshape(-1, dim))

    def forward_selected_varlen(self, x: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if query_mask.shape != (batch, seq_len):
            raise ValueError("query_mask must have shape [batch, seq_len].")
        if self.training and self.dropout != 0.0:
            raise ValueError("sparse_varlen route block execution currently requires attention dropout == 0.")
        if create_block_mask is None:
            raise ModuleNotFoundError("PyTorch FlexAttention block masks are required for sparse_varlen.")
        batch_indices, query_positions = torch.where(query_mask)
        if batch_indices.numel() == 0:
            return x.new_empty((0, dim))

        q_weight = self.qkv.weight[:dim, :]
        kv_weight = self.qkv.weight[dim:, :]
        q = F.linear(x[query_mask], q_weight)
        kv = F.linear(x, kv_weight)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope.cos[:, :, :seq_len, :].to(device=k.device)
        sin = self.rope.sin[:, :, :seq_len, :].to(device=k.device)
        k = apply_rotary(k, cos, sin)
        q = self.rope.apply_to_positions(q, query_positions)

        query = q.transpose(0, 1).unsqueeze(0).contiguous()
        key = k.permute(1, 0, 2, 3).reshape(1, self.n_heads, batch * seq_len, self.head_dim).contiguous()
        value = v.permute(1, 0, 2, 3).reshape(1, self.n_heads, batch * seq_len, self.head_dim).contiguous()

        key_indices = torch.arange(batch * seq_len, device=x.device)
        key_batches = torch.div(key_indices, seq_len, rounding_mode="floor")
        key_positions = key_indices % seq_len
        query_batches = batch_indices
        selected_query_positions = query_positions

        def mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            return (query_batches[q_idx] == key_batches[kv_idx]) & (
                key_positions[kv_idx] <= selected_query_positions[q_idx]
            )

        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=query.size(2),
            KV_LEN=batch * seq_len,
            device=x.device,
            BLOCK_SIZE=(128, 128),
        )
        y = flex_attention_if_available(query, key, value, block_mask=block_mask)
        selected = y.squeeze(0).transpose(0, 1).to(dtype=q.dtype)
        return self.out(selected.reshape(-1, dim))

    def _attention_global_write_tensors(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_global_kv_mode == "token_compressed":
            assert self.global_key_write is not None
            assert self.global_value_write is not None
            return self.global_key_write(k), self.global_value_write(v)
        return k, v

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

    def _flatten_token_global_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 5:
            return tensor
        batch, heads, slots, seq_len, dim = tensor.shape
        return tensor.reshape(batch, heads, slots * seq_len, dim)

    def _global_prefix_mask(
        self,
        global_valid: torch.Tensor,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if global_valid.dim() == 3:
            return self._token_global_prefix_mask(global_valid, seq_len, dtype=dtype, device=device)
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

    def _token_global_prefix_mask(
        self,
        global_valid: torch.Tensor,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        batch = global_valid.size(0)
        slots = int(global_valid.size(1))
        global_seq_len = int(global_valid.size(2))
        local = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        query_pos = torch.arange(seq_len, device=device)
        global_pos = torch.arange(global_seq_len, device=device)
        global_causal = global_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        allowed_global = (
            global_valid.to(device=device, dtype=torch.bool)[:, None, :, :]
            & global_causal[None, :, None, :]
        ).reshape(batch, seq_len, slots * global_seq_len)
        allowed = torch.cat(
            [
                allowed_global[:, None, :, :],
                local[None, None, :, :].expand(batch, 1, -1, -1),
            ],
            dim=-1,
        )
        mask = torch.zeros(batch, 1, seq_len, slots * global_seq_len + seq_len, dtype=dtype, device=device)
        if slots:
            mask[:, :, :, : slots * global_seq_len] = self._global_logit_bias(dtype=dtype, device=device)
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
            if global_keys.dim() == 5:
                global_scores = self._token_global_last_query_scores(query, global_keys.detach(), global_valid)
                global_weights_shape = global_scores.shape
                flat_global_scores = global_scores.reshape(
                    global_scores.size(0),
                    global_scores.size(1),
                    global_scores.size(2),
                    -1,
                )
            else:
                global_scores = torch.matmul(query, global_keys.detach().transpose(-2, -1)) * (self.head_dim**-0.5)
                global_scores = global_scores + self._global_logit_bias(dtype=global_scores.dtype, device=global_scores.device)
                invalid = ~global_valid.to(device=global_scores.device, dtype=torch.bool)[:, None, None, :]
                flat_global_scores = global_scores.masked_fill(invalid, torch.finfo(global_scores.dtype).min)
                global_weights_shape = global_scores.shape
            local_scores = torch.matmul(query, local_k.detach().transpose(-2, -1)) * (self.head_dim**-0.5)
            weights = torch.softmax(torch.cat([flat_global_scores, local_scores], dim=-1), dim=-1)
            global_weights = weights[..., : flat_global_scores.size(-1)].reshape(global_weights_shape)
            sink_slots = int(getattr(self, "attention_global_sink_slots", 0))
            sink_count = max(0, min(sink_slots, slots))
            if global_weights.dim() == 5:
                sink_mass = (
                    global_weights[:, :, :, :sink_count, :].sum(dim=(-1, -2)).mean()
                    if sink_count
                    else global_weights.new_zeros(())
                )
                window_mass = global_weights[:, :, :, sink_count:, :].sum(dim=(-1, -2)).mean()
                total_mass = global_weights.sum(dim=(-1, -2)).mean()
            else:
                sink_mass = global_weights[..., :sink_count].sum(dim=-1).mean() if sink_count else global_weights.new_zeros(())
                window_mass = global_weights[..., sink_count:].sum(dim=-1).mean()
                total_mass = global_weights.sum(dim=-1).mean()
            return {
                "attention_global_kv_last_token_mass": total_mass,
                "attention_global_kv_sink_last_token_mass": sink_mass,
                "attention_global_kv_window_last_token_mass": window_mass,
                "attention_global_kv_logit_bias": self._global_logit_bias(
                    dtype=global_weights.dtype,
                    device=global_weights.device,
                ),
            }

    def _token_global_last_query_scores(
        self,
        query: torch.Tensor,
        global_keys: torch.Tensor,
        global_valid: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("bhqd,bhsud->bhqsu", query, global_keys) * (self.head_dim**-0.5)
        scores = scores + self._global_logit_bias(dtype=scores.dtype, device=scores.device)
        seq_len = global_keys.size(3)
        valid = global_valid.to(device=scores.device, dtype=torch.bool)[:, None, None, :, :]
        query_pos = torch.full((1,), seq_len - 1, device=scores.device, dtype=torch.long)
        global_pos = torch.arange(seq_len, device=scores.device)
        causal = global_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        allowed = valid & causal[None, None, :, None, :]
        return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

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

    def forward_selected(self, x: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        selected = x[query_mask] + self.attn.forward_selected(attn_input, query_mask)
        return selected + self.ffn(self.ffn_norm(selected))

    def forward_selected_varlen(self, x: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        selected = x[query_mask] + self.attn.forward_selected_varlen(attn_input, query_mask)
        return selected + self.ffn(self.ffn_norm(selected))


def build_causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=ignore_index)


def count_parameters(module: nn.Module) -> int:
    require_torch()
    return sum(parameter.numel() for parameter in module.parameters())
