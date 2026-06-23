from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass
class AttentionGlobalKVState:
    keys: "torch.Tensor"
    values: "torch.Tensor"
    valid: "torch.Tensor"

    @property
    def slots(self) -> int:
        return int(self.keys.size(2))

    def index_select(self, indices: "torch.Tensor") -> "AttentionGlobalKVState":
        return AttentionGlobalKVState(
            keys=self.keys.index_select(0, indices),
            values=self.values.index_select(0, indices),
            valid=self.valid.index_select(0, indices),
        )


class CanonicalAttentionGlobalKVCache:
    """Per-forward attention-level K/V cache with sink + sliding-window retention."""

    def __init__(self, sink_slots: int, window_slots: int) -> None:
        if sink_slots < 0 or window_slots < 0:
            raise ValueError("sink_slots and window_slots must be non-negative")
        if sink_slots + window_slots <= 0:
            raise ValueError("Attention Global KV cache requires at least one retained slot")
        self.sink_slots = sink_slots
        self.window_slots = window_slots

    def empty(
        self,
        *,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        device: "torch.device",
        dtype: "torch.dtype",
        sequence_length: int | None = None,
    ) -> AttentionGlobalKVState:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Attention Global KV cache.")
        if sequence_length is not None:
            return AttentionGlobalKVState(
                keys=torch.empty(batch_size, n_heads, 0, sequence_length, head_dim, device=device, dtype=dtype),
                values=torch.empty(batch_size, n_heads, 0, sequence_length, head_dim, device=device, dtype=dtype),
                valid=torch.empty(batch_size, 0, sequence_length, device=device, dtype=torch.bool),
            )
        return AttentionGlobalKVState(
            keys=torch.empty(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype),
            values=torch.empty(batch_size, n_heads, 0, head_dim, device=device, dtype=dtype),
            valid=torch.empty(batch_size, 0, device=device, dtype=torch.bool),
        )

    def write(
        self,
        state: AttentionGlobalKVState,
        key: "torch.Tensor",
        value: "torch.Tensor",
        valid: "torch.Tensor",
    ) -> AttentionGlobalKVState:
        state_rank = state.keys.dim()
        if state_rank not in {4, 5}:
            raise ValueError("Attention Global KV state must be rank 4 or rank 5.")
        if key.dim() == 3:
            key = key.unsqueeze(2)
        if value.dim() == 3:
            value = value.unsqueeze(2)
        if state_rank == 5 and key.dim() == 4:
            key = key.unsqueeze(2)
        if state_rank == 5 and value.dim() == 4:
            value = value.unsqueeze(2)
        if valid.dim() == 1:
            valid = valid.unsqueeze(1)
        if state_rank == 5 and valid.dim() == 2:
            valid = valid.unsqueeze(1)
        if key.dim() != state_rank or value.dim() != state_rank:
            raise ValueError(
                "Attention Global KV writes must match cache rank: "
                "[batch, heads, slots, head_dim] or [batch, heads, slots, seq, head_dim]"
            )
        if key.shape != value.shape:
            raise ValueError("Attention Global KV key/value write shapes must match")
        if key.size(0) != state.keys.size(0) or key.size(1) != state.keys.size(1) or key.size(-1) != state.keys.size(-1):
            raise ValueError("Attention Global KV write shape does not match cache state")
        if state_rank == 5 and key.size(3) != state.keys.size(3):
            raise ValueError("Attention Global KV write sequence length does not match cache state")
        expected_valid = (key.size(0), key.size(2)) if state_rank == 4 else (key.size(0), key.size(2), key.size(3))
        if valid.shape != expected_valid:
            raise ValueError(f"Attention Global KV valid mask must have shape {expected_valid}")
        appended = AttentionGlobalKVState(
            keys=torch.cat([state.keys, key], dim=2),
            values=torch.cat([state.values, value], dim=2),
            valid=torch.cat([state.valid, valid.to(dtype=torch.bool, device=state.valid.device)], dim=1),
        )
        return self._retain(appended)

    def _retain(self, state: AttentionGlobalKVState) -> AttentionGlobalKVState:
        sink = state.keys[:, :, : self.sink_slots, ...] if self.sink_slots else state.keys[:, :, :0, ...]
        sink_values = state.values[:, :, : self.sink_slots, ...] if self.sink_slots else state.values[:, :, :0, ...]
        sink_valid = state.valid[:, : self.sink_slots, ...] if self.sink_slots else state.valid[:, :0, ...]
        tail_keys = state.keys[:, :, self.sink_slots :, ...]
        tail_values = state.values[:, :, self.sink_slots :, ...]
        tail_valid = state.valid[:, self.sink_slots :, ...]
        if self.window_slots:
            window = tail_keys[:, :, -self.window_slots :, ...]
            window_values = tail_values[:, :, -self.window_slots :, ...]
            window_valid = tail_valid[:, -self.window_slots :, ...]
        else:
            window = tail_keys[:, :, :0, ...]
            window_values = tail_values[:, :, :0, ...]
            window_valid = tail_valid[:, :0, ...]
        return AttentionGlobalKVState(
            keys=torch.cat([sink, window], dim=2),
            values=torch.cat([sink_values, window_values], dim=2),
            valid=torch.cat([sink_valid, window_valid], dim=1),
        )
