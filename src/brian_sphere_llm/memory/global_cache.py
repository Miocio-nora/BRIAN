from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass
class GlobalCacheState:
    codes: "torch.Tensor"

    @property
    def slots(self) -> int:
        return int(self.codes.size(1))


class CanonicalGlobalCache:
    """Per-forward canonical code cache with sink + sliding-window retention."""

    def __init__(self, sink_slots: int, window_slots: int) -> None:
        if sink_slots < 0 or window_slots < 0:
            raise ValueError("sink_slots and window_slots must be non-negative")
        if sink_slots + window_slots <= 0:
            raise ValueError("Global cache requires at least one retained slot")
        self.sink_slots = sink_slots
        self.window_slots = window_slots

    def empty(self, *, batch_size: int, code_dim: int, device: "torch.device", dtype: "torch.dtype") -> GlobalCacheState:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for Global KV cache.")
        return GlobalCacheState(codes=torch.empty(batch_size, 0, code_dim, device=device, dtype=dtype))

    def write(self, state: GlobalCacheState, code: "torch.Tensor") -> GlobalCacheState:
        if code.dim() not in {2, 3}:
            raise ValueError("Global cache write code must have shape [batch, code_dim] or [batch, seq, code_dim]")
        write = code.unsqueeze(1)
        codes = state.codes
        if codes.size(1) == 0 and codes.dim() != write.dim():
            codes = write[:, :0]
        if codes.dim() != write.dim():
            raise ValueError("Global cache write rank does not match cache state")
        appended = torch.cat([codes, write], dim=1)
        sink = appended[:, : self.sink_slots, :] if self.sink_slots else appended[:, :0, :]
        tail = appended[:, self.sink_slots :, :]
        if self.window_slots:
            window = tail[:, -self.window_slots :, :]
        else:
            window = tail[:, :0, :]
        return GlobalCacheState(codes=torch.cat([sink, window], dim=1))
