from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None

ModuleBase = nn.Module if nn is not None else object


@dataclass
class BDRECacheState:
    """Persistent reader-compiled cache, stored one tensor per completed token."""

    block_keys: tuple[torch.Tensor, ...] = ()
    block_values: tuple[torch.Tensor, ...] = ()
    step_keys: tuple[torch.Tensor, ...] = ()
    step_values: tuple[torch.Tensor, ...] = ()
    writer_keys: tuple[torch.Tensor, ...] = ()
    writer_values: tuple[torch.Tensor, ...] = ()
    writer_blocks: tuple[torch.Tensor, ...] = ()
    writer_valid: tuple[torch.Tensor, ...] = ()
    lazy_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return len(self.block_keys)

    def append(
        self,
        block_key: torch.Tensor,
        block_value: torch.Tensor,
        *,
        step_key: torch.Tensor | None = None,
        step_value: torch.Tensor | None = None,
        writer_key: torch.Tensor | None = None,
        writer_value: torch.Tensor | None = None,
        writer_block: torch.Tensor | None = None,
        writer_valid: torch.Tensor | None = None,
        lazy_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> "BDRECacheState":
        has_step = step_key is not None or step_value is not None
        if has_step and (step_key is None or step_value is None):
            raise ValueError("BDRE step Key and Value caches must be appended together.")
        has_writer = any(value is not None for value in (writer_key, writer_value, writer_block, writer_valid))
        if has_writer and any(value is None for value in (writer_key, writer_value, writer_block, writer_valid)):
            raise ValueError("BDRE writer history requires Key, Value, block IDs, and validity.")
        return BDRECacheState(
            block_keys=self.block_keys + (block_key,),
            block_values=self.block_values + (block_value,),
            step_keys=self.step_keys + ((step_key,) if step_key is not None else ()),
            step_values=self.step_values + ((step_value,) if step_value is not None else ()),
            writer_keys=self.writer_keys + ((writer_key,) if writer_key is not None else ()),
            writer_values=self.writer_values + ((writer_value,) if writer_value is not None else ()),
            writer_blocks=self.writer_blocks + ((writer_block,) if writer_block is not None else ()),
            writer_valid=self.writer_valid + ((writer_valid,) if writer_valid is not None else ()),
            lazy_cache={} if lazy_cache is None else lazy_cache,
        )

    def stacked_blocks(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.block_keys:
            raise ValueError("BDRE block cache is empty.")
        return torch.stack(self.block_keys, dim=1), torch.stack(self.block_values, dim=1)

    def stacked_steps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.step_keys:
            raise ValueError("BDRE eager reader-step cache is empty.")
        return torch.stack(self.step_keys, dim=1), torch.stack(self.step_values, dim=1)

    def stacked_writers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.writer_keys:
            raise ValueError("BDRE writer history is unavailable for reader-step compilation.")
        return (
            torch.stack(self.writer_keys, dim=1),
            torch.stack(self.writer_values, dim=1),
            torch.stack(self.writer_blocks, dim=1),
            torch.stack(self.writer_valid, dim=1),
        )

    def memory_bytes(self) -> int:
        lazy_tensors = tuple(tensor for pair in self.lazy_cache.values() for tensor in pair)
        tensors = (
            self.block_keys
            + self.block_values
            + self.step_keys
            + self.step_values
            + self.writer_keys
            + self.writer_values
            + self.writer_blocks
            + self.writer_valid
            + lazy_tensors
        )
        return sum(int(value.numel() * value.element_size()) for value in tensors)


@dataclass(frozen=True)
class BDRECompileOutput:
    keys: torch.Tensor
    values: torch.Tensor
    key_weights: torch.Tensor
    value_weights: torch.Tensor
    metrics: dict[str, torch.Tensor]


class BDRECompiler(ModuleBase):
    """Position-conditioned compiler from writer-step codes to reader caches."""

    def __init__(
        self,
        *,
        max_route_steps: int,
        key_temperature: float,
        value_temperature: float,
        position_tau: float = 1.0,
        step_lambda: float = 0.0,
        late_step_weight: float = 0.0,
        compile_top_k: int | None = None,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for BDRE compilation.")
        super().__init__()
        if max_route_steps < 1:
            raise ValueError("max_route_steps must be positive.")
        if key_temperature <= 0.0 or value_temperature <= 0.0:
            raise ValueError("BDRE temperatures must be positive.")
        if compile_top_k is not None and compile_top_k < 1:
            raise ValueError("bdre_compile_top_k must be null or positive.")
        self.max_route_steps = int(max_route_steps)
        self.key_temperature = float(key_temperature)
        self.value_temperature = float(value_temperature)
        self.position_tau = float(position_tau)
        self.step_lambda = float(step_lambda)
        self.late_step_weight = float(late_step_weight)
        self.compile_top_k = int(compile_top_k) if compile_top_k is not None else None

    def compile(
        self,
        writer_keys: torch.Tensor,
        writer_values: torch.Tensor,
        writer_blocks: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        reader_step: int | torch.Tensor | None = None,
        reader_actions: torch.Tensor | None = None,
    ) -> BDRECompileOutput:
        """Compile `[B,S,r]` writer codes for all readers or selected readers."""

        if writer_keys.dim() != 3 or writer_values.dim() != 3:
            raise ValueError("BDRE writer codes must have shape [batch, steps, code_dim].")
        if writer_blocks.shape != writer_valid.shape or writer_blocks.shape != writer_keys.shape[:2]:
            raise ValueError("BDRE writer block and validity tensors must match [batch, steps].")
        if writer_keys.size(1) > self.max_route_steps:
            raise ValueError("BDRE writer history exceeds max_route_steps.")
        normalized_positions = F.normalize(block_positions, dim=-1)
        safe_blocks = writer_blocks.clamp(min=0, max=normalized_positions.size(0) - 1)
        writer_positions = normalized_positions[safe_blocks]
        if reader_actions is None:
            reader_positions = normalized_positions.unsqueeze(0).expand(writer_keys.size(0), -1, -1)
        else:
            if reader_actions.shape != (writer_keys.size(0),):
                raise ValueError("reader_actions must have shape [batch].")
            reader_positions = normalized_positions[reader_actions].unsqueeze(1)
        scores = self.position_tau * torch.einsum("brp,bsp->brs", reader_positions, writer_positions)
        scores = self._add_step_terms(scores, reader_step)
        support = self._support_mask(scores, writer_valid)
        key_weights = self._weights(scores, support, self.key_temperature)
        value_weights = self._weights(scores, support, self.value_temperature)
        keys = torch.einsum("brs,bsk->brk", key_weights.to(writer_keys.dtype), writer_keys)
        values = torch.einsum("brs,bsv->brv", value_weights.to(writer_values.dtype), writer_values)
        metrics = self._metrics(key_weights, value_weights, writer_valid, support)
        return BDRECompileOutput(keys, values, key_weights, value_weights, metrics)

    def compile_all_reader_steps(
        self,
        writer_keys: torch.Tensor,
        writer_values: torch.Tensor,
        writer_blocks: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [
            self.compile(
                writer_keys,
                writer_values,
                writer_blocks,
                writer_valid,
                block_positions,
                reader_step=step,
            )
            for step in range(self.max_route_steps)
        ]
        return (
            torch.stack([output.keys for output in outputs], dim=1),
            torch.stack([output.values for output in outputs], dim=1),
        )

    def compile_history_for_reader(
        self,
        writer_keys: torch.Tensor,
        writer_values: torch.Tensor,
        writer_blocks: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        reader_action: int,
        reader_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamically compile `[B,T,S,r]` histories for one reader state."""

        if writer_keys.dim() != 4:
            raise ValueError("BDRE writer history must have shape [batch, tokens, steps, code_dim].")
        batch, tokens, steps, key_dim = writer_keys.shape
        value_dim = writer_values.size(-1)
        flat_actions = torch.full(
            (batch * tokens,),
            int(reader_action),
            dtype=torch.long,
            device=writer_keys.device,
        )
        output = self.compile(
            writer_keys.reshape(batch * tokens, steps, key_dim),
            writer_values.reshape(batch * tokens, steps, value_dim),
            writer_blocks.reshape(batch * tokens, steps),
            writer_valid.reshape(batch * tokens, steps),
            block_positions,
            reader_step=reader_step,
            reader_actions=flat_actions,
        )
        return (
            output.keys[:, 0].reshape(batch, tokens, key_dim),
            output.values[:, 0].reshape(batch, tokens, value_dim),
        )

    def _add_step_terms(self, scores: torch.Tensor, reader_step: int | torch.Tensor | None) -> torch.Tensor:
        step_indexes = torch.arange(scores.size(-1), device=scores.device, dtype=scores.dtype)
        if reader_step is not None and self.step_lambda != 0.0:
            if isinstance(reader_step, int):
                distance = (step_indexes - float(reader_step)).abs().view(1, 1, -1)
            else:
                reader = reader_step.to(device=scores.device, dtype=scores.dtype).reshape(-1, 1, 1)
                distance = (step_indexes.view(1, 1, -1) - reader).abs()
            scores = scores - self.step_lambda * distance
        if self.late_step_weight != 0.0:
            late = (step_indexes + 1.0) / float(self.max_route_steps)
            scores = scores + self.late_step_weight * late.view(1, 1, -1)
        return scores

    def _support_mask(self, scores: torch.Tensor, writer_valid: torch.Tensor) -> torch.Tensor:
        support = writer_valid.unsqueeze(1).expand(-1, scores.size(1), -1)
        if self.compile_top_k is None or self.compile_top_k >= scores.size(-1):
            return support
        k = min(self.compile_top_k, scores.size(-1))
        masked_scores = scores.masked_fill(~support, torch.finfo(scores.dtype).min)
        indexes = masked_scores.topk(k, dim=-1).indices
        top_mask = torch.zeros_like(support)
        top_mask.scatter_(-1, indexes, True)
        return support & top_mask

    @staticmethod
    def _weights(scores: torch.Tensor, support: torch.Tensor, temperature: float) -> torch.Tensor:
        masked = (scores / temperature).masked_fill(~support, torch.finfo(scores.dtype).min)
        return F.softmax(masked.float(), dim=-1).to(dtype=scores.dtype)

    @staticmethod
    def _metrics(
        key_weights: torch.Tensor,
        value_weights: torch.Tensor,
        writer_valid: torch.Tensor,
        support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        key_entropy = -(key_weights.float().clamp_min(1e-9) * key_weights.float().clamp_min(1e-9).log()).sum(-1)
        value_entropy = -(
            value_weights.float().clamp_min(1e-9) * value_weights.float().clamp_min(1e-9).log()
        ).sum(-1)
        step_indexes = torch.arange(writer_valid.size(-1), device=writer_valid.device)
        last_indexes = step_indexes.unsqueeze(0).masked_fill(~writer_valid, -1).max(dim=-1).values.clamp_min(0)
        gather_index = last_indexes[:, None, None].expand(-1, key_weights.size(1), 1)
        last_mass = 0.5 * (
            key_weights.gather(-1, gather_index).squeeze(-1) + value_weights.gather(-1, gather_index).squeeze(-1)
        )
        return {
            "key_weight_entropy": key_entropy.mean(),
            "value_weight_entropy": value_entropy.mean(),
            "selected_writer_steps": support.to(key_weights.dtype).sum(dim=-1).mean(),
            "last_step_mass": last_mass.mean(),
        }
