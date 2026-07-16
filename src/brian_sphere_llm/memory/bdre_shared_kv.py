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


if torch is not None:

    @torch.compile(fullgraph=True, dynamic=True)
    def _compiled_incremental_prefix_step(
        writer_key: torch.Tensor,
        writer_value: torch.Tensor,
        writer_block: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        writer_step: torch.Tensor,
        key_max: torch.Tensor,
        key_denom: torch.Tensor,
        key_numerator: torch.Tensor,
        value_max: torch.Tensor,
        value_denom: torch.Tensor,
        value_numerator: torch.Tensor,
        position_tau: float,
        step_lambda: float,
        step_scale: float,
        late_step_weight: float,
        max_route_steps: int,
        key_temperature: float,
        value_temperature: float,
    ) -> tuple[torch.Tensor, ...]:
        normalized_positions = F.normalize(block_positions, dim=-1)
        safe_blocks = writer_block.clamp(min=0, max=normalized_positions.size(0) - 1)
        writer_position = F.embedding(safe_blocks, normalized_positions)
        scores = position_tau * torch.einsum(
            "rp,np->nr",
            normalized_positions,
            writer_position,
        )
        step = writer_step.to(device=scores.device, dtype=scores.dtype)
        if step_lambda != 0.0:
            scores = scores + step_lambda * step / step_scale
        if late_step_weight != 0.0:
            scores = scores + late_step_weight * (step + 1.0) / float(max_route_steps)
        valid = writer_valid.unsqueeze(1)

        def update(
            logits: torch.Tensor,
            previous_max: torch.Tensor,
            previous_denom: torch.Tensor,
            previous_numerator: torch.Tensor,
            writer: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            candidate = torch.where(valid, logits, torch.full_like(logits, -float("inf")))
            next_max = torch.maximum(previous_max, candidate)
            previous_scale = torch.where(
                torch.isfinite(previous_max),
                torch.exp(previous_max - next_max),
                torch.zeros_like(previous_max),
            )
            writer_scale = torch.where(
                valid,
                torch.exp(logits - next_max),
                torch.zeros_like(logits),
            )
            next_denom = previous_denom * previous_scale + writer_scale
            next_numerator = (
                previous_numerator * previous_scale.unsqueeze(-1)
                + writer.float().unsqueeze(1) * writer_scale.unsqueeze(-1)
            )
            compiled = (next_numerator / next_denom.clamp_min(1e-20).unsqueeze(-1)).to(
                dtype=writer.dtype
            )
            return next_max, next_denom, next_numerator, compiled

        key_max, key_denom, key_numerator, keys = update(
            scores / key_temperature,
            key_max,
            key_denom,
            key_numerator,
            writer_key,
        )
        value_max, value_denom, value_numerator, values = update(
            scores / value_temperature,
            value_max,
            value_denom,
            value_numerator,
            writer_value,
        )
        return (
            key_max,
            key_denom,
            key_numerator,
            value_max,
            value_denom,
            value_numerator,
            keys,
            values,
        )

else:  # pragma: no cover - exercised only without PyTorch.
    _compiled_incremental_prefix_step = None


@dataclass
class BDRECacheState:
    """Persistent reader-compiled cache with a contiguous token dimension."""

    block_keys: torch.Tensor | None = None
    block_values: torch.Tensor | None = None
    step_keys: torch.Tensor | None = None
    step_values: torch.Tensor | None = None
    writer_keys: torch.Tensor | None = None
    writer_values: torch.Tensor | None = None
    writer_blocks: torch.Tensor | None = None
    writer_valid: torch.Tensor | None = None
    lazy_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return 0 if self.block_keys is None else int(self.block_keys.size(1))

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
        return self.append_tokens(
            block_key.unsqueeze(1),
            block_value.unsqueeze(1),
            step_key=None if step_key is None else step_key.unsqueeze(1),
            step_value=None if step_value is None else step_value.unsqueeze(1),
            writer_key=None if writer_key is None else writer_key.unsqueeze(1),
            writer_value=None if writer_value is None else writer_value.unsqueeze(1),
            writer_block=None if writer_block is None else writer_block.unsqueeze(1),
            writer_valid=None if writer_valid is None else writer_valid.unsqueeze(1),
            lazy_cache=lazy_cache,
        )

    def append_tokens(
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
        """Append an already contiguous token chunk along the cache token axis."""

        has_step = step_key is not None or step_value is not None
        if has_step and (step_key is None or step_value is None):
            raise ValueError("BDRE step Key and Value caches must be appended together.")
        has_writer = any(value is not None for value in (writer_key, writer_value, writer_block, writer_valid))
        if has_writer and any(value is None for value in (writer_key, writer_value, writer_block, writer_valid)):
            raise ValueError("BDRE writer history requires Key, Value, block IDs, and validity.")
        return BDRECacheState(
            block_keys=_append_tokens(self.block_keys, block_key),
            block_values=_append_tokens(self.block_values, block_value),
            step_keys=_append_optional_tokens(self.step_keys, step_key),
            step_values=_append_optional_tokens(self.step_values, step_value),
            writer_keys=_append_optional_tokens(self.writer_keys, writer_key),
            writer_values=_append_optional_tokens(self.writer_values, writer_value),
            writer_blocks=_append_optional_tokens(self.writer_blocks, writer_block),
            writer_valid=_append_optional_tokens(self.writer_valid, writer_valid),
            lazy_cache={} if lazy_cache is None else lazy_cache,
        )

    def detached(self) -> "BDRECacheState":
        return BDRECacheState(
            block_keys=_detach_optional(self.block_keys),
            block_values=_detach_optional(self.block_values),
            step_keys=_detach_optional(self.step_keys),
            step_values=_detach_optional(self.step_values),
            writer_keys=_detach_optional(self.writer_keys),
            writer_values=_detach_optional(self.writer_values),
            writer_blocks=_detach_optional(self.writer_blocks),
            writer_valid=_detach_optional(self.writer_valid),
            lazy_cache={key: (value[0].detach(), value[1].detach()) for key, value in self.lazy_cache.items()},
        )

    def stacked_blocks(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.block_keys is None or self.block_values is None:
            raise ValueError("BDRE block cache is empty.")
        return self.block_keys, self.block_values

    def stacked_steps(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.step_keys is None or self.step_values is None:
            raise ValueError("BDRE eager reader-step cache is empty.")
        return self.step_keys, self.step_values

    def stacked_writers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(
            value is None
            for value in (self.writer_keys, self.writer_values, self.writer_blocks, self.writer_valid)
        ):
            raise ValueError("BDRE writer history is unavailable for reader-step compilation.")
        return (
            self.writer_keys,
            self.writer_values,
            self.writer_blocks,
            self.writer_valid,
        )

    def memory_bytes(self) -> int:
        lazy_tensors = tuple(tensor for pair in self.lazy_cache.values() for tensor in pair)
        tensors = tuple(
            value
            for value in (
                self.block_keys,
                self.block_values,
                self.step_keys,
                self.step_values,
                self.writer_keys,
                self.writer_values,
                self.writer_blocks,
                self.writer_valid,
            )
            if value is not None
        ) + lazy_tensors
        return sum(int(value.numel() * value.element_size()) for value in tensors)


def _append_tokens(history: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    if value.dim() < 2:
        raise ValueError("BDRE cache chunks must include batch and token dimensions.")
    return value if history is None else torch.cat((history, value), dim=1)


def _append_optional_tokens(history: torch.Tensor | None, value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return history
    return _append_tokens(history, value)


def _detach_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach()


@dataclass(frozen=True)
class BDRECompileOutput:
    keys: torch.Tensor
    values: torch.Tensor
    key_weights: torch.Tensor
    value_weights: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BDREIncrementalCompileState:
    key_max: torch.Tensor
    key_denom: torch.Tensor
    key_numerator: torch.Tensor
    value_max: torch.Tensor
    value_denom: torch.Tensor
    value_numerator: torch.Tensor


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
        normalize_step_distance: bool = False,
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
        self.normalize_step_distance = bool(normalize_step_distance)
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
        collect_metrics: bool = True,
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
        writer_positions = F.embedding(safe_blocks, normalized_positions)
        if reader_actions is None:
            reader_positions = normalized_positions.unsqueeze(0).expand(writer_keys.size(0), -1, -1)
        else:
            if reader_actions.shape != (writer_keys.size(0),):
                raise ValueError("reader_actions must have shape [batch].")
            reader_positions = F.embedding(reader_actions, normalized_positions).unsqueeze(1)
        scores = self.position_tau * torch.einsum("brp,bsp->brs", reader_positions, writer_positions)
        scores = self._add_step_terms(scores, reader_step)
        support = self._support_mask(scores, writer_valid)
        key_weights = self._weights(scores, support, self.key_temperature)
        value_weights = self._weights(scores, support, self.value_temperature)
        keys = torch.einsum("brs,bsk->brk", key_weights.to(writer_keys.dtype), writer_keys)
        values = torch.einsum("brs,bsv->brv", value_weights.to(writer_values.dtype), writer_values)
        metrics = (
            self._metrics(key_weights, value_weights, writer_valid, support)
            if collect_metrics
            else {}
        )
        return BDRECompileOutput(keys, values, key_weights, value_weights, metrics)

    def compile_prefix(
        self,
        writer_keys: torch.Tensor,
        writer_values: torch.Tensor,
        writer_blocks: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        reader_step: int,
        reader_actions: torch.Tensor | None = None,
        collect_metrics: bool = True,
    ) -> BDRECompileOutput:
        """Compile the writer prefix visible at one synchronous route step."""

        if reader_step < 0 or reader_step >= self.max_route_steps:
            raise ValueError("reader_step must be within max_route_steps.")
        step_indexes = torch.arange(writer_valid.size(1), device=writer_valid.device)
        prefix_valid = writer_valid & (step_indexes.unsqueeze(0) <= int(reader_step))
        if not bool(prefix_valid.any(dim=-1).all()):
            raise ValueError("Every synchronous-prefix token must have a valid writer by reader_step.")
        return self.compile(
            writer_keys,
            writer_values,
            writer_blocks,
            prefix_valid,
            block_positions,
            reader_step=reader_step,
            reader_actions=reader_actions,
            collect_metrics=collect_metrics,
        )

    def compile_prefix_incremental(
        self,
        writer_key: torch.Tensor,
        writer_value: torch.Tensor,
        writer_block: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        writer_step: int,
        state: BDREIncrementalCompileState | None,
    ) -> tuple[BDRECompileOutput, BDREIncrementalCompileState]:
        """Exactly update an unrestricted synchronous-prefix softmax online."""

        if _compiled_incremental_prefix_step is None:
            raise RuntimeError("Incremental BDRE compilation requires PyTorch.")
        if self.compile_top_k is not None:
            raise ValueError("Incremental prefix compilation does not support compile_top_k.")
        if writer_key.dim() != 2 or writer_value.dim() != 2:
            raise ValueError("Incremental writer codes must have shape [batch, code_dim].")
        if writer_block.shape != writer_valid.shape or writer_block.shape != writer_key.shape[:1]:
            raise ValueError("Incremental writer block and validity tensors must match [batch].")
        if writer_step < 0 or writer_step >= self.max_route_steps:
            raise ValueError("writer_step must be within max_route_steps.")

        batch = writer_key.size(0)
        readers = block_positions.size(0)
        if state is None:
            score_shape = (batch, readers)
            state = BDREIncrementalCompileState(
                key_max=torch.full(
                    score_shape,
                    -float("inf"),
                    device=writer_key.device,
                    dtype=torch.float32,
                ),
                key_denom=torch.zeros(score_shape, device=writer_key.device, dtype=torch.float32),
                key_numerator=torch.zeros(
                    (*score_shape, writer_key.size(-1)),
                    device=writer_key.device,
                    dtype=torch.float32,
                ),
                value_max=torch.full(
                    score_shape,
                    -float("inf"),
                    device=writer_value.device,
                    dtype=torch.float32,
                ),
                value_denom=torch.zeros(
                    score_shape,
                    device=writer_value.device,
                    dtype=torch.float32,
                ),
                value_numerator=torch.zeros(
                    (*score_shape, writer_value.size(-1)),
                    device=writer_value.device,
                    dtype=torch.float32,
                ),
            )
        step_scale = (
            float(self.max_route_steps - 1)
            if self.normalize_step_distance and self.max_route_steps > 1
            else 1.0
        )
        result = _compiled_incremental_prefix_step(
            writer_key,
            writer_value,
            writer_block,
            writer_valid,
            block_positions,
            torch.tensor(writer_step, device=writer_key.device),
            state.key_max,
            state.key_denom,
            state.key_numerator,
            state.value_max,
            state.value_denom,
            state.value_numerator,
            self.position_tau,
            self.step_lambda,
            step_scale,
            self.late_step_weight,
            self.max_route_steps,
            self.key_temperature,
            self.value_temperature,
        )
        next_state = BDREIncrementalCompileState(*result[:6])
        empty_weights = result[6].new_empty((batch, readers, 0))
        output = BDRECompileOutput(
            keys=result[6],
            values=result[7],
            key_weights=empty_weights,
            value_weights=empty_weights,
            metrics={},
        )
        return output, next_state

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

    def compile_all_reader_steps_vectorized(
        self,
        writer_keys: torch.Tensor,
        writer_values: torch.Tensor,
        writer_blocks: torch.Tensor,
        writer_valid: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        collect_metrics: bool = True,
    ) -> tuple[BDRECompileOutput, ...]:
        """Compile every reader depth in one batched, full-bank operation."""

        if writer_keys.dim() != 3 or writer_values.dim() != 3:
            raise ValueError("BDRE writer codes must have shape [batch, steps, code_dim].")
        if writer_blocks.shape != writer_valid.shape or writer_blocks.shape != writer_keys.shape[:2]:
            raise ValueError("BDRE writer block and validity tensors must match [batch, steps].")
        if writer_keys.size(1) > self.max_route_steps:
            raise ValueError("BDRE writer history exceeds max_route_steps.")
        if not bool(writer_valid.any(dim=-1).all()):
            raise ValueError("Every full-bank token must have at least one valid writer.")

        normalized_positions = F.normalize(block_positions, dim=-1)
        safe_blocks = writer_blocks.clamp(min=0, max=normalized_positions.size(0) - 1)
        writer_positions = F.embedding(safe_blocks, normalized_positions)
        base_scores = self.position_tau * torch.einsum(
            "rp,bsp->brs",
            normalized_positions,
            writer_positions,
        )
        reader_steps = torch.arange(
            self.max_route_steps,
            device=writer_keys.device,
            dtype=base_scores.dtype,
        )
        writer_steps = torch.arange(
            writer_keys.size(1),
            device=writer_keys.device,
            dtype=base_scores.dtype,
        )
        scores = base_scores.unsqueeze(1).expand(-1, self.max_route_steps, -1, -1)
        if self.step_lambda != 0.0:
            distance = (reader_steps[:, None] - writer_steps[None, :]).abs()
            if self.normalize_step_distance and self.max_route_steps > 1:
                distance = distance / float(self.max_route_steps - 1)
            scores = scores - self.step_lambda * distance.view(
                1,
                self.max_route_steps,
                1,
                writer_keys.size(1),
            )
        if self.late_step_weight != 0.0:
            late = (writer_steps + 1.0) / float(self.max_route_steps)
            scores = scores + self.late_step_weight * late.view(
                1,
                1,
                1,
                writer_keys.size(1),
            )

        batch = writer_keys.size(0)
        readers = normalized_positions.size(0)
        flat_scores = scores.reshape(batch * self.max_route_steps, readers, writer_keys.size(1))
        flat_valid = writer_valid[:, None, :].expand(
            -1,
            self.max_route_steps,
            -1,
        ).reshape(batch * self.max_route_steps, writer_keys.size(1))
        flat_support = self._support_mask(flat_scores, flat_valid)
        key_weights = self._weights(
            flat_scores,
            flat_support,
            self.key_temperature,
        ).view(batch, self.max_route_steps, readers, writer_keys.size(1))
        value_weights = self._weights(
            flat_scores,
            flat_support,
            self.value_temperature,
        ).view(batch, self.max_route_steps, readers, writer_keys.size(1))
        support = flat_support.view(batch, self.max_route_steps, readers, writer_keys.size(1))
        keys = torch.einsum("blrs,bsk->blrk", key_weights.to(writer_keys.dtype), writer_keys)
        values = torch.einsum("blrs,bsv->blrv", value_weights.to(writer_values.dtype), writer_values)

        outputs: list[BDRECompileOutput] = []
        for reader_step in range(self.max_route_steps):
            metrics = (
                self._metrics(
                    key_weights[:, reader_step],
                    value_weights[:, reader_step],
                    writer_valid,
                    support[:, reader_step],
                )
                if collect_metrics
                else {}
            )
            outputs.append(
                BDRECompileOutput(
                    keys[:, reader_step],
                    values[:, reader_step],
                    key_weights[:, reader_step],
                    value_weights[:, reader_step],
                    metrics,
                )
            )
        return tuple(outputs)

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
            if self.normalize_step_distance and self.max_route_steps > 1:
                distance = distance / float(self.max_route_steps - 1)
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
