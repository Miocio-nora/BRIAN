from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brian_sphere_llm.losses.balance_loss import block_balance_loss
from brian_sphere_llm.losses.coverage_floor_loss import block_coverage_floor_loss
from brian_sphere_llm.losses.cost_loss import route_cost_loss
from brian_sphere_llm.losses.exit_boundary_loss import exit_boundary_loss
from brian_sphere_llm.losses.location_loss import location_loss
from brian_sphere_llm.losses.route_loss import route_imitation_loss
from brian_sphere_llm.losses.selected_balance_loss import selected_block_balance_loss
from brian_sphere_llm.losses.transition_diversity_loss import transition_diversity_loss
from brian_sphere_llm.memory.bdre_shared_kv import BDRECacheState, BDRECompileOutput, BDRECompiler
from brian_sphere_llm.model.baseline import _float_value, _int_value
from brian_sphere_llm.model.brian_model import (
    BrianRouteConfig,
    BrianRouteCore,
    _bool_value,
    _coverage_floor_min,
    _loss_weight,
    _loss_weights_mapping,
    _routing_constraints_mapping,
    _routing_options_mapping,
    _zero_loss_like,
)
from brian_sphere_llm.model.llama_backbone import apply_rotary, build_causal_lm_loss, count_parameters
from brian_sphere_llm.routing.metrics import summarize_routes

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    F = None

ModuleBase = nn.Module if nn is not None else object


@dataclass(frozen=True)
class BDREConfig:
    route: BrianRouteConfig
    key_dim: int = 32
    value_dim: int = 32
    depth_mode: str = "none"
    step_lambda: float = 0.0
    position_tau: float = 1.0
    late_step_weight: float = 0.0
    compile_top_k: int | None = None
    key_temperature: float = 0.5
    value_temperature: float = 1.0
    reader_step_cache: str = "lazy"
    self_kv_mode: str = "bdre_prefix"
    key_read_mode: str = "explicit_decode_rope"
    value_read_mode: str = "latent_aggregate"
    position_geometry_enabled: bool = True
    position_geometry_weight: float = 0.001
    position_geometry_normalize: bool = True
    execution_mode: str = "token_by_token"
    dispatch_mode: str = "legacy_cuda_scan"
    chunk_size: int = 1
    normalize_step_distance: bool = False
    synchronous_attention_backend: str = "per_query_reference"
    depth_visibility_policy: str = "depth_prefix"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, config_dir: str | Path | None = None) -> "BDREConfig":
        if not _bool_value(data.get("bdre_shared_kv", False), "bdre_shared_kv"):
            raise ValueError("BDRE model configuration requires bdre_shared_kv=true.")
        route = BrianRouteConfig.from_dict(data, config_dir=config_dir)
        geometry = data.get("position_geometry", {})
        if not isinstance(geometry, Mapping):
            raise ValueError("position_geometry must be a mapping.")
        execution = data.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ValueError("execution must be a mapping.")
        compile_top_k = _optional_positive_int(data.get("bdre_compile_top_k"), "bdre_compile_top_k")
        config = cls(
            route=route,
            key_dim=_int_value(data.get("bdre_key_dim", 32), "bdre_key_dim", minimum=1),
            value_dim=_int_value(data.get("bdre_value_dim", 32), "bdre_value_dim", minimum=1),
            depth_mode=str(data.get("bdre_depth_mode", "none")),
            step_lambda=_float_value(data.get("bdre_step_lambda", 0.0), "bdre_step_lambda", minimum=0.0),
            position_tau=_float_value(data.get("bdre_position_tau", 1.0), "bdre_position_tau", minimum=0.0),
            late_step_weight=_float_value(
                data.get("bdre_late_step_weight", 0.0),
                "bdre_late_step_weight",
                minimum=0.0,
            ),
            compile_top_k=compile_top_k,
            key_temperature=_float_value(
                data.get("bdre_key_temperature", 0.5),
                "bdre_key_temperature",
                minimum=1e-6,
            ),
            value_temperature=_float_value(
                data.get("bdre_value_temperature", 1.0),
                "bdre_value_temperature",
                minimum=1e-6,
            ),
            reader_step_cache=str(data.get("bdre_reader_step_cache", "lazy")),
            self_kv_mode=str(data.get("bdre_self_kv_mode", "bdre_prefix")),
            key_read_mode=str(data.get("bdre_key_read_mode", "explicit_decode_rope")),
            value_read_mode=str(data.get("bdre_value_read_mode", "latent_aggregate")),
            position_geometry_enabled=_bool_value(
                geometry.get("enabled", True),
                "position_geometry.enabled",
            ),
            position_geometry_weight=_float_value(
                geometry.get("weight", 0.001),
                "position_geometry.weight",
                minimum=0.0,
            ),
            position_geometry_normalize=_bool_value(
                geometry.get("normalize", True),
                "position_geometry.normalize",
            ),
            execution_mode=str(execution.get("mode", "token_by_token")),
            dispatch_mode=str(execution.get("dispatch", "legacy_cuda_scan")),
            chunk_size=_int_value(execution.get("chunk_size", 1), "execution.chunk_size", minimum=1),
            normalize_step_distance=_bool_value(
                data.get("bdre_normalize_step_distance", False),
                "bdre_normalize_step_distance",
            ),
            synchronous_attention_backend=str(
                execution.get("attention_backend", "per_query_reference")
            ),
            depth_visibility_policy=str(
                execution.get("depth_visibility_policy", "depth_prefix")
            ),
        )
        if str(geometry.get("internal_target", "regular_simplex")) != "regular_simplex":
            raise ValueError("BDRE v1 position_geometry.internal_target must be regular_simplex.")
        if str(geometry.get("in_out_target", "antipodal_orthogonal")) != "antipodal_orthogonal":
            raise ValueError("BDRE v1 position_geometry.in_out_target must be antipodal_orthogonal.")
        config.validate()
        return config

    def validate(self) -> None:
        if self.depth_mode not in {"none", "reader_step", "synchronous_prefix"}:
            raise ValueError("bdre_depth_mode must be 'none', 'reader_step', or 'synchronous_prefix'.")
        if self.reader_step_cache not in {"eager", "lazy", "dynamic"}:
            raise ValueError("bdre_reader_step_cache must be 'eager', 'lazy', or 'dynamic'.")
        if self.self_kv_mode not in {"bdre_prefix", "current_step", "none"}:
            raise ValueError("bdre_self_kv_mode must be 'bdre_prefix', 'current_step', or 'none'.")
        if self.key_read_mode != "explicit_decode_rope":
            raise ValueError("BDRE v1 supports only explicit_decode_rope Key reads.")
        if self.value_read_mode != "latent_aggregate":
            raise ValueError("BDRE v1 supports only latent_aggregate Value reads.")
        if self.execution_mode not in {"token_by_token", "synchronous_prefix"}:
            raise ValueError("BDRE execution.mode must be 'token_by_token' or 'synchronous_prefix'.")
        if self.dispatch_mode not in {
            "legacy_cuda_scan",
            "grouped_host",
            "grouped_mm",
        }:
            raise ValueError(
                "BDRE execution.dispatch must be 'legacy_cuda_scan', 'grouped_host', "
                "or 'grouped_mm'."
            )
        if self.synchronous_attention_backend not in {"per_query_reference", "shared_padded_explicit"}:
            raise ValueError(
                "BDRE execution.attention_backend must be 'per_query_reference' or 'shared_padded_explicit'."
            )
        if self.depth_visibility_policy not in {"depth_prefix", "full_bank"}:
            raise ValueError(
                "BDRE execution.depth_visibility_policy must be 'depth_prefix' or 'full_bank'."
            )
        if self.execution_mode == "synchronous_prefix":
            if self.depth_mode != "synchronous_prefix":
                raise ValueError("synchronous_prefix execution requires bdre_depth_mode=synchronous_prefix.")
            if self.reader_step_cache != "eager":
                raise ValueError("synchronous_prefix execution requires bdre_reader_step_cache=eager.")
            if self.self_kv_mode != "bdre_prefix":
                raise ValueError("synchronous_prefix execution requires bdre_self_kv_mode=bdre_prefix.")
            if not self.route.hard_exit:
                raise ValueError("synchronous_prefix execution requires hard_exit=true.")
        elif self.depth_mode == "synchronous_prefix":
            raise ValueError("bdre_depth_mode=synchronous_prefix requires synchronous_prefix execution.")
        elif self.depth_visibility_policy != "depth_prefix":
            raise ValueError("full_bank depth visibility is available only for synchronous_prefix execution.")
        elif self.synchronous_attention_backend != "per_query_reference":
            raise ValueError("Shared padded attention is available only for synchronous_prefix execution.")
        if self.dispatch_mode == "grouped_mm" and (
            self.execution_mode != "synchronous_prefix"
            or self.synchronous_attention_backend != "shared_padded_explicit"
        ):
            raise ValueError(
                "Grouped MM dispatch requires synchronous_prefix execution with "
                "shared_padded_explicit attention."
            )
        if not self.position_geometry_normalize:
            raise ValueError("BDRE v1 requires normalized position geometry.")
        if self.route.top_k != 1 or self.route.later_top_k != 1:
            raise ValueError("BDRE v1 requires top_k=later_top_k=1.")
        if self.route.global_kv or self.route.attention_global_kv or self.route.parallel_passing:
            raise ValueError("BDRE shared KV cannot be combined with legacy Global KV or parallel passing.")
        if self.route.block_position_mode != "spherical_code" or not self.route.independent_input_position:
            raise ValueError("BDRE v1 requires spherical_code with independent_input_position=true.")
        if self.compile_top_k is not None and self.compile_top_k > self.route.max_route_steps:
            raise ValueError("bdre_compile_top_k cannot exceed max_route_steps.")


@dataclass(frozen=True)
class _SynchronousActionGroup:
    action: int
    indexes: torch.Tensor
    active_batches: torch.Tensor
    flat_slots: torch.Tensor
    max_selected: int


@dataclass(frozen=True)
class _SynchronousGroupedMMDispatch:
    indexes: torch.Tensor
    actions: torch.Tensor
    offsets: torch.Tensor
    groups: tuple[_SynchronousActionGroup, ...]


@dataclass(frozen=True)
class _SynchronousGroupedMMWeights:
    position_adapter: torch.Tensor | None
    attn_norm: torch.Tensor
    qkv: torch.Tensor
    key_write: torch.Tensor
    value_write: torch.Tensor
    attention_out: torch.Tensor
    ffn_norm: torch.Tensor
    ffn_gate_up: torch.Tensor
    ffn_down: torch.Tensor


class BDREBlockProjection(ModuleBase):
    """Independent writer canonicalizers and per-head reader decoders for one free block."""

    def __init__(self, d_model: int, n_heads: int, key_dim: int, value_dim: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for BDRE projections.")
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        head_dim = d_model // n_heads
        self.key_write = nn.Linear(d_model, key_dim, bias=False)
        self.value_write = nn.Linear(d_model, value_dim, bias=False)
        self.key_read = nn.Parameter(torch.empty(n_heads, key_dim, head_dim))
        self.value_read = nn.Parameter(torch.empty(n_heads, value_dim, head_dim))
        nn.init.xavier_uniform_(self.key_write.weight)
        nn.init.xavier_uniform_(self.value_write.weight)
        nn.init.xavier_uniform_(self.key_read)
        nn.init.xavier_uniform_(self.value_read)


@dataclass(frozen=True)
class IncrementalAttentionState:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> "IncrementalAttentionState":
        keys = key if self.keys is None else torch.cat((self.keys, key), dim=2)
        values = value if self.values is None else torch.cat((self.values, value), dim=2)
        return IncrementalAttentionState(keys, values)

    def detached(self) -> "IncrementalAttentionState":
        return IncrementalAttentionState(
            None if self.keys is None else self.keys.detach(),
            None if self.values is None else self.values.detach(),
        )


@dataclass(frozen=True)
class BDREIncrementalState:
    pre: tuple[IncrementalAttentionState, ...]
    post: tuple[IncrementalAttentionState, ...]
    cache: BDRECacheState

    def detached(self) -> "BDREIncrementalState":
        return BDREIncrementalState(
            pre=tuple(state.detached() for state in self.pre),
            post=tuple(state.detached() for state in self.post),
            cache=self.cache.detached(),
        )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, name, minimum=1)


def _incremental_transformer_block(
    block: Any,
    hidden: torch.Tensor,
    state: IncrementalAttentionState,
    token_position: int,
) -> tuple[torch.Tensor, IncrementalAttentionState]:
    attn_input = block.attn_norm(hidden)
    attention = block.attn
    batch, one, dim = attn_input.shape
    if one != 1:
        raise ValueError("Incremental transformer execution requires one token.")
    q, key, value = attention.qkv(attn_input).chunk(3, dim=-1)
    q = q.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
    cosine = attention.rope.cos[:, :, token_position : token_position + 1, :].to(
        device=q.device,
        dtype=q.dtype,
    )
    sine = attention.rope.sin[:, :, token_position : token_position + 1, :].to(
        device=q.device,
        dtype=q.dtype,
    )
    q = apply_rotary(q, cosine, sine)
    key = apply_rotary(key, cosine, sine)
    all_keys = torch.cat((state.keys, key), dim=2) if state.keys is not None else key
    all_values = torch.cat((state.values, value), dim=2) if state.values is not None else value
    attended = F.scaled_dot_product_attention(
        q,
        all_keys,
        all_values,
        is_causal=False,
        dropout_p=attention.dropout if block.training else 0.0,
    )
    attended = attended.transpose(1, 2).contiguous().view(batch, 1, dim)
    hidden = hidden + attention.out(attended)
    hidden = hidden + block.ffn(block.ffn_norm(hidden))
    return hidden, IncrementalAttentionState(all_keys, all_values)


def _chunk_transformer_block(
    block: Any,
    hidden: torch.Tensor,
    state: IncrementalAttentionState,
    start_position: int,
) -> tuple[torch.Tensor, IncrementalAttentionState]:
    """Advance a standard causal block over several contiguous token positions."""

    attn_input = block.attn_norm(hidden)
    attention = block.attn
    batch, chunk, dim = attn_input.shape
    q, key, value = attention.qkv(attn_input).chunk(3, dim=-1)
    q = q.view(batch, chunk, attention.n_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, chunk, attention.n_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch, chunk, attention.n_heads, attention.head_dim).transpose(1, 2)
    positions = torch.arange(start_position, start_position + chunk, device=hidden.device)
    cosine = attention.rope.cos[:, :, positions, :].to(device=q.device, dtype=q.dtype)
    sine = attention.rope.sin[:, :, positions, :].to(device=q.device, dtype=q.dtype)
    q = apply_rotary(q, cosine, sine)
    key = apply_rotary(key, cosine, sine)
    all_keys = torch.cat((state.keys, key), dim=2) if state.keys is not None else key
    all_values = torch.cat((state.values, value), dim=2) if state.values is not None else value
    if all_keys.size(2) != start_position + chunk:
        raise ValueError("Chunk attention state length does not match the BDRE token position.")
    key_positions = torch.arange(all_keys.size(2), device=hidden.device)
    allowed = key_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
    attended = F.scaled_dot_product_attention(
        q,
        all_keys,
        all_values,
        attn_mask=allowed,
        is_causal=False,
        dropout_p=attention.dropout if block.training else 0.0,
    )
    attended = attended.transpose(1, 2).contiguous().view(batch, chunk, dim)
    hidden = hidden + attention.out(attended)
    hidden = hidden + block.ffn(block.ffn_norm(hidden))
    return hidden, IncrementalAttentionState(all_keys, all_values)


class _BDREDiagnostics:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started = time.perf_counter()
        self.compile_seconds = 0.0
        self.self_prefix_seconds = 0.0
        self.compile_metric_sums: dict[str, torch.Tensor] = {}
        self.compile_output_count = 0
        self.writer_counts: list[torch.Tensor] = []
        self.lazy_requests = 0
        self.lazy_hits = 0
        self.last_visualization: dict[str, torch.Tensor] | None = None

    def record_compile(self, output: BDRECompileOutput, elapsed: float) -> None:
        self.compile_seconds += elapsed
        if not output.metrics:
            return
        self.compile_output_count += 1
        for name, value in output.metrics.items():
            detached = value.detach().float()
            self.compile_metric_sums[name] = self.compile_metric_sums.get(name, torch.zeros_like(detached)) + detached

    def metrics(self, state: BDRECacheState, position_metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
        elapsed = max(1e-12, time.perf_counter() - self.started)
        counts = torch.cat([value.detach().float().reshape(-1) for value in self.writer_counts])
        if self.compile_output_count:
            metric_names = ("key_weight_entropy", "value_weight_entropy", "selected_writer_steps", "last_step_mass")
            compile_metrics = {
                name: float((self.compile_metric_sums[name] / self.compile_output_count).cpu())
                for name in metric_names
            }
        else:
            compile_metrics = {
                "key_weight_entropy": 0.0,
                "value_weight_entropy": 0.0,
                "selected_writer_steps": 0.0,
                "last_step_mass": 0.0,
            }
        compiled_state_count = 0
        if state.step_keys is not None:
            compiled_state_count = (
                state.tokens
                * state.step_keys.size(0)
                * state.step_keys.size(2)
                * state.step_keys.size(3)
            )
        elif state.block_keys is not None:
            compiled_state_count = state.tokens * state.block_keys.size(0) * state.block_keys.size(2)
        return {
            "bdre_compile_time_ms": self.compile_seconds * 1000.0,
            "bdre_compile_fraction": self.compile_seconds / elapsed,
            "bdre_self_prefix_time_ms": self.self_prefix_seconds * 1000.0,
            "bdre_self_prefix_fraction": self.self_prefix_seconds / elapsed,
            "bdre_compiled_state_count": float(compiled_state_count),
            "bdre_lazy_cache_hit_rate": self.lazy_hits / max(1, self.lazy_requests),
            "bdre_writer_steps_mean": float(counts.mean().cpu()),
            "bdre_writer_steps_min": float(counts.min().cpu()),
            "bdre_writer_steps_max": float(counts.max().cpu()),
            "bdre_key_weight_entropy": compile_metrics["key_weight_entropy"],
            "bdre_value_weight_entropy": compile_metrics["value_weight_entropy"],
            "bdre_selected_writer_steps": compile_metrics["selected_writer_steps"],
            "bdre_last_step_mass": compile_metrics["last_step_mass"],
            "bdre_position_gram_error": float(position_metrics["position_gram_error"].detach().float().cpu()),
            "bdre_internal_position_min_angle": float(
                position_metrics["internal_position_min_angle"].detach().float().cpu()
            ),
            "bdre_cache_memory_mb": state.memory_bytes() / (1024.0 * 1024.0),
        }


class BrianBDRERouteCore(BrianRouteCore):
    """Exact token-serial BRIAN route core with reader-compiled shared KV."""

    def __init__(self, bdre_config: BDREConfig) -> None:
        bdre_config.validate()
        super().__init__(bdre_config.route)
        self.bdre_config = bdre_config
        self.bdre_projections = nn.ModuleList(
            [
                BDREBlockProjection(
                    self.config.base.d_model,
                    self.config.base.n_heads,
                    bdre_config.key_dim,
                    bdre_config.value_dim,
                )
                for _ in range(self.config.route_pool_blocks)
            ]
        )
        self.bdre_compiler = BDRECompiler(
            max_route_steps=self.config.max_route_steps,
            key_temperature=bdre_config.key_temperature,
            value_temperature=bdre_config.value_temperature,
            position_tau=bdre_config.position_tau,
            step_lambda=bdre_config.step_lambda,
            normalize_step_distance=bdre_config.normalize_step_distance,
            late_step_weight=bdre_config.late_step_weight,
            compile_top_k=bdre_config.compile_top_k,
        )

    def empty_incremental_state(self) -> BDREIncrementalState:
        return BDREIncrementalState(
            pre=tuple(IncrementalAttentionState() for _ in self.pre_blocks),
            post=tuple(IncrementalAttentionState() for _ in self.post_blocks),
            cache=BDRECacheState(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        route_mode: str = "free",
        pseudo_policy: str = "sequential",
        loss_weights: Mapping[str, Any] | None = None,
        routing_constraints: Mapping[str, Any] | None = None,
        routing_options: Mapping[str, Any] | None = None,
        hard_exit: bool | None = None,
        log_path_counts: bool = False,
        router_probability: float | None = None,
        global_step: int = 0,
        collect_router_space: bool = False,
        collect_bdre_visualization: bool = False,
        summarize_routing: bool = True,
        stream_chunk_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stream_chunk_options is not None:
            if targets is not None:
                raise ValueError("Streaming chunk forward does not accept full-sequence targets.")
            return self.forward_stream_chunk(input_ids, **dict(stream_chunk_options))
        if self.bdre_config.execution_mode == "synchronous_prefix":
            return self._forward_synchronous_prefix(
                input_ids,
                targets,
                route_mode=route_mode,
                pseudo_policy=pseudo_policy,
                loss_weights=loss_weights,
                routing_constraints=routing_constraints,
                routing_options=routing_options,
                hard_exit=hard_exit,
                log_path_counts=log_path_counts,
                router_probability=router_probability,
                global_step=global_step,
                collect_router_space=collect_router_space,
                collect_bdre_visualization=collect_bdre_visualization,
                summarize_routing=summarize_routing,
            )
        if input_ids.dim() != 2 or input_ids.size(1) < 1:
            raise ValueError("BDRE exact forward requires input_ids with shape [batch, sequence].")
        if route_mode == "parallel":
            raise ValueError("BDRE v1 does not support parallel route passing.")
        loss_weights = _loss_weights_mapping(loss_weights)
        routing_constraints = _routing_constraints_mapping(routing_constraints)
        routing_options = _routing_options_mapping(routing_options)
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        route_targets = self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        state = self.empty_incremental_state()
        diagnostics = _BDREDiagnostics(enabled=summarize_routing)
        logits_by_token: list[torch.Tensor] = []
        last_route_info: dict[str, Any] | None = None
        last_router_records: list[dict[str, Any]] | None = None

        for token_position in range(input_ids.size(1)):
            record = token_position == input_ids.size(1) - 1
            logits, state, route_info, router_records = self._forward_one_token(
                input_ids[:, token_position : token_position + 1],
                state,
                token_position=token_position,
                route_mode=route_mode,
                route_targets=route_targets,
                routing_constraints=routing_constraints,
                routing_options=routing_options,
                hard_exit=bool(hard_exit),
                router_probability=router_probability,
                global_step=global_step,
                diagnostics=diagnostics,
                record=record,
                collect_router_space=collect_router_space and record,
                collect_bdre_visualization=collect_bdre_visualization and record,
            )
            logits_by_token.append(logits)
            if record:
                last_route_info = route_info
                last_router_records = router_records

        assert last_route_info is not None
        logits = torch.cat(logits_by_token, dim=1)
        bdre_metrics = (
            diagnostics.metrics(state.cache, self.position_table.geometry_metrics())
            if summarize_routing
            else {}
        )
        output = self._build_output(
            logits,
            last_route_info,
            targets=targets,
            loss_weights=loss_weights,
            routing_constraints=routing_constraints,
            max_steps=len(last_route_info["route_logits"]),
            summarize_routing=summarize_routing,
            log_path_counts=log_path_counts,
            bdre_metrics=bdre_metrics,
        )
        if last_router_records is not None:
            output["router_space"] = {
                "records": last_router_records,
                "num_actions": self.config.route_pool_blocks + 1,
                "out_action": self.out_action,
            }
        if diagnostics.last_visualization is not None:
            output["bdre_visualization"] = diagnostics.last_visualization
        return output

    def forward_incremental(
        self,
        input_ids: torch.Tensor,
        state: BDREIncrementalState | None = None,
        *,
        route_mode: str = "free",
        pseudo_policy: str = "sequential",
        routing_constraints: Mapping[str, Any] | None = None,
        routing_options: Mapping[str, Any] | None = None,
        hard_exit: bool | None = None,
        router_probability: float | None = None,
        global_step: int = 0,
        collect_bdre_visualization: bool = False,
    ) -> dict[str, Any]:
        """Advance one token without recomputing the prefix."""

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        if input_ids.dim() != 2 or input_ids.size(1) != 1:
            raise ValueError("forward_incremental accepts exactly one token per batch item.")
        if self.bdre_config.execution_mode == "synchronous_prefix":
            output = self._forward_synchronous_prefix_stream_chunk(
                input_ids,
                state,
                route_mode=route_mode,
                pseudo_policy=pseudo_policy,
                routing_constraints=routing_constraints,
                routing_options=routing_options,
                hard_exit=hard_exit,
                router_probability=router_probability,
                global_step=global_step,
                collect_bdre_visualization=collect_bdre_visualization,
            )
            return output
        state = state or self.empty_incremental_state()
        constraints = _routing_constraints_mapping(routing_constraints)
        options = _routing_options_mapping(routing_options)
        diagnostics = _BDREDiagnostics()
        route_targets = self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        logits, next_state, route_info, _ = self._forward_one_token(
            input_ids,
            state,
            token_position=state.cache.tokens,
            route_mode=route_mode,
            route_targets=route_targets,
            routing_constraints=constraints,
            routing_options=options,
            hard_exit=self.config.hard_exit if hard_exit is None else hard_exit,
            router_probability=router_probability,
            global_step=global_step,
            diagnostics=diagnostics,
            record=True,
            collect_router_space=False,
            collect_bdre_visualization=collect_bdre_visualization,
        )
        metrics = diagnostics.metrics(next_state.cache, self.position_table.geometry_metrics())
        summary = summarize_routes(route_info, self.config.route_pool_blocks)
        summary.update(metrics)
        output = {
            "logits": logits,
            "route_info": route_info,
            "routing_summary": summary,
            "incremental_state": next_state,
        }
        if diagnostics.last_visualization is not None:
            output["bdre_visualization"] = diagnostics.last_visualization
        return output

    def prepare_stream_route_targets(
        self,
        input_ids: torch.Tensor,
        *,
        route_mode: str,
        pseudo_policy: str,
    ) -> list[torch.Tensor]:
        """Prepare one route-target schedule shared by every token chunk."""

        return self._targets_for_mode(route_mode, pseudo_policy, input_ids)

    def forward_stream_chunk(
        self,
        input_ids: torch.Tensor,
        state: BDREIncrementalState | None = None,
        *,
        next_token_targets: torch.Tensor | None = None,
        loss_token_count: int | None = None,
        include_auxiliary_losses: bool = False,
        route_targets: list[torch.Tensor] | None = None,
        route_mode: str = "free",
        pseudo_policy: str = "sequential",
        loss_weights: Mapping[str, Any] | None = None,
        routing_constraints: Mapping[str, Any] | None = None,
        routing_options: Mapping[str, Any] | None = None,
        hard_exit: bool | None = None,
        log_path_counts: bool = False,
        router_probability: float | None = None,
        global_step: int = 0,
        collect_router_space: bool = False,
        collect_bdre_visualization: bool = False,
        summarize_routing: bool = True,
    ) -> dict[str, Any]:
        """Advance an exact token-serial chunk while preserving its persistent KV state."""

        if self.bdre_config.execution_mode == "synchronous_prefix":
            return self._forward_synchronous_prefix_stream_chunk(
                input_ids,
                state,
                next_token_targets=next_token_targets,
                loss_token_count=loss_token_count,
                include_auxiliary_losses=include_auxiliary_losses,
                route_targets=route_targets,
                route_mode=route_mode,
                pseudo_policy=pseudo_policy,
                loss_weights=loss_weights,
                routing_constraints=routing_constraints,
                routing_options=routing_options,
                hard_exit=hard_exit,
                log_path_counts=log_path_counts,
                router_probability=router_probability,
                global_step=global_step,
                collect_router_space=collect_router_space,
                collect_bdre_visualization=collect_bdre_visualization,
                summarize_routing=summarize_routing,
            )

        if input_ids.dim() != 2 or input_ids.size(1) < 1:
            raise ValueError("BDRE stream chunks require input_ids with shape [batch, chunk].")
        if route_mode == "parallel":
            raise ValueError("BDRE v1 does not support parallel route passing.")
        if next_token_targets is not None and next_token_targets.shape != input_ids.shape:
            raise ValueError("next_token_targets must match the stream chunk shape.")
        if next_token_targets is not None and (loss_token_count is None or loss_token_count < 1):
            raise ValueError("loss_token_count must be positive when next_token_targets are provided.")

        state = state or self.empty_incremental_state()
        if (
            state.cache.tokens
            and state.cache.block_keys is not None
            and state.cache.block_keys.size(0) != input_ids.size(0)
        ):
            raise ValueError("BDRE stream state batch size does not match input_ids.")
        loss_weights = _loss_weights_mapping(loss_weights)
        constraints = _routing_constraints_mapping(routing_constraints)
        options = _routing_options_mapping(routing_options)
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        route_targets = (
            route_targets
            if route_targets is not None
            else self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        )
        diagnostics = _BDREDiagnostics(enabled=summarize_routing)
        start_position = state.cache.tokens
        logits_by_token: list[torch.Tensor] = []
        last_route_info: dict[str, Any] | None = None
        last_router_records: list[dict[str, Any]] | None = None

        for chunk_position in range(input_ids.size(1)):
            record = chunk_position == input_ids.size(1) - 1
            logits, state, route_info, router_records = self._forward_one_token(
                input_ids[:, chunk_position : chunk_position + 1],
                state,
                token_position=start_position + chunk_position,
                route_mode=route_mode,
                route_targets=route_targets,
                routing_constraints=constraints,
                routing_options=options,
                hard_exit=bool(hard_exit),
                router_probability=router_probability,
                global_step=global_step,
                diagnostics=diagnostics,
                record=record,
                collect_router_space=collect_router_space and record,
                collect_bdre_visualization=collect_bdre_visualization and record,
            )
            logits_by_token.append(logits)
            if record:
                last_route_info = route_info
                last_router_records = router_records

        assert last_route_info is not None
        logits = torch.cat(logits_by_token, dim=1)
        lm_loss = None
        if next_token_targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                next_token_targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / int(loss_token_count)
        bdre_metrics = (
            diagnostics.metrics(state.cache, self.position_table.geometry_metrics())
            if summarize_routing
            else {}
        )
        output = self._build_output(
            logits,
            last_route_info,
            targets=None,
            lm_loss=lm_loss,
            include_auxiliary_losses=include_auxiliary_losses,
            loss_weights=loss_weights,
            routing_constraints=constraints,
            max_steps=len(last_route_info["route_logits"]),
            summarize_routing=summarize_routing,
            log_path_counts=log_path_counts,
            bdre_metrics=bdre_metrics,
        )
        output["incremental_state"] = state
        if last_router_records is not None:
            output["router_space"] = {
                "records": last_router_records,
                "num_actions": self.config.route_pool_blocks + 1,
                "out_action": self.out_action,
            }
        if diagnostics.last_visualization is not None:
            output["bdre_visualization"] = diagnostics.last_visualization
        return output

    def _forward_synchronous_prefix(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None,
        *,
        route_mode: str,
        pseudo_policy: str,
        loss_weights: Mapping[str, Any] | None,
        routing_constraints: Mapping[str, Any] | None,
        routing_options: Mapping[str, Any] | None,
        hard_exit: bool | None,
        log_path_counts: bool,
        router_probability: float | None,
        global_step: int,
        collect_router_space: bool,
        collect_bdre_visualization: bool,
        summarize_routing: bool,
    ) -> dict[str, Any]:
        if input_ids.dim() != 2 or input_ids.size(1) < 1:
            raise ValueError("BDRE synchronous-prefix forward requires [batch, sequence] input_ids.")
        if route_mode == "parallel":
            raise ValueError("BDRE synchronous-prefix execution does not support parallel route passing.")
        mapped_loss_weights = _loss_weights_mapping(loss_weights)
        constraints = _routing_constraints_mapping(routing_constraints)
        options = _routing_options_mapping(routing_options)
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        if not hard_exit:
            raise ValueError("BDRE synchronous-prefix execution requires hard exit.")
        route_targets = self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        state = self.empty_incremental_state()
        diagnostics = _BDREDiagnostics(enabled=summarize_routing)
        logits_by_chunk: list[torch.Tensor] = []
        last_route_info: dict[str, Any] | None = None
        last_router_records: list[dict[str, Any]] | None = None
        chunk_size = min(self.bdre_config.chunk_size, input_ids.size(1))

        for start in range(0, input_ids.size(1), chunk_size):
            end = min(input_ids.size(1), start + chunk_size)
            final_chunk = end == input_ids.size(1)
            logits, state, route_info, router_records = self._run_synchronous_prefix_chunk(
                input_ids[:, start:end],
                state,
                route_mode=route_mode,
                route_targets=route_targets,
                routing_constraints=constraints,
                routing_options=options,
                hard_exit=True,
                router_probability=router_probability,
                global_step=global_step,
                diagnostics=diagnostics,
                record=final_chunk,
                collect_router_space=collect_router_space and final_chunk,
                collect_bdre_visualization=collect_bdre_visualization and final_chunk,
            )
            logits_by_chunk.append(logits)
            if final_chunk:
                last_route_info = route_info
                last_router_records = router_records

        assert last_route_info is not None
        logits = torch.cat(logits_by_chunk, dim=1)
        bdre_metrics = (
            diagnostics.metrics(state.cache, self.position_table.geometry_metrics())
            if summarize_routing
            else {}
        )
        output = self._build_output(
            logits,
            last_route_info,
            targets=targets,
            loss_weights=mapped_loss_weights,
            routing_constraints=constraints,
            max_steps=len(last_route_info["route_logits"]),
            summarize_routing=summarize_routing,
            log_path_counts=log_path_counts,
            bdre_metrics=bdre_metrics,
        )
        if last_router_records is not None:
            output["router_space"] = {
                "records": last_router_records,
                "num_actions": self.config.route_pool_blocks + 1,
                "out_action": self.out_action,
            }
        if diagnostics.last_visualization is not None:
            output["bdre_visualization"] = diagnostics.last_visualization
        return output

    def _forward_synchronous_prefix_stream_chunk(
        self,
        input_ids: torch.Tensor,
        state: BDREIncrementalState | None = None,
        *,
        next_token_targets: torch.Tensor | None = None,
        loss_token_count: int | None = None,
        include_auxiliary_losses: bool = False,
        route_targets: list[torch.Tensor] | None = None,
        route_mode: str = "free",
        pseudo_policy: str = "sequential",
        loss_weights: Mapping[str, Any] | None = None,
        routing_constraints: Mapping[str, Any] | None = None,
        routing_options: Mapping[str, Any] | None = None,
        hard_exit: bool | None = None,
        log_path_counts: bool = False,
        router_probability: float | None = None,
        global_step: int = 0,
        collect_router_space: bool = False,
        collect_bdre_visualization: bool = False,
        summarize_routing: bool = True,
    ) -> dict[str, Any]:
        if input_ids.dim() != 2 or input_ids.size(1) < 1:
            raise ValueError("BDRE synchronous-prefix chunks require [batch, chunk] input_ids.")
        if input_ids.size(1) > self.bdre_config.chunk_size:
            raise ValueError("Synchronous-prefix chunk exceeds execution.chunk_size.")
        if route_mode == "parallel":
            raise ValueError("BDRE synchronous-prefix execution does not support parallel route passing.")
        if next_token_targets is not None and next_token_targets.shape != input_ids.shape:
            raise ValueError("next_token_targets must match the synchronous-prefix chunk shape.")
        if next_token_targets is not None and (loss_token_count is None or loss_token_count < 1):
            raise ValueError("loss_token_count must be positive when next_token_targets are provided.")
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        if not hard_exit:
            raise ValueError("BDRE synchronous-prefix execution requires hard exit.")
        state = state or self.empty_incremental_state()
        if state.cache.tokens and state.cache.step_keys is None:
            raise ValueError("Synchronous-prefix history requires step-indexed cache tensors.")
        if (
            state.cache.tokens
            and state.cache.block_keys is not None
            and state.cache.block_keys.size(0) != input_ids.size(0)
        ):
            raise ValueError("BDRE synchronous-prefix state batch size does not match input_ids.")

        mapped_loss_weights = _loss_weights_mapping(loss_weights)
        constraints = _routing_constraints_mapping(routing_constraints)
        options = _routing_options_mapping(routing_options)
        route_targets = (
            route_targets
            if route_targets is not None
            else self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        )
        diagnostics = _BDREDiagnostics(enabled=summarize_routing)
        logits, next_state, route_info, router_records = self._run_synchronous_prefix_chunk(
            input_ids,
            state,
            route_mode=route_mode,
            route_targets=route_targets,
            routing_constraints=constraints,
            routing_options=options,
            hard_exit=True,
            router_probability=router_probability,
            global_step=global_step,
            diagnostics=diagnostics,
            record=True,
            collect_router_space=collect_router_space,
            collect_bdre_visualization=collect_bdre_visualization,
        )
        lm_loss = None
        if next_token_targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                next_token_targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / int(loss_token_count)
        bdre_metrics = (
            diagnostics.metrics(next_state.cache, self.position_table.geometry_metrics())
            if summarize_routing
            else {}
        )
        output = self._build_output(
            logits,
            route_info,
            targets=None,
            lm_loss=lm_loss,
            include_auxiliary_losses=include_auxiliary_losses,
            loss_weights=mapped_loss_weights,
            routing_constraints=constraints,
            max_steps=len(route_info["route_logits"]),
            summarize_routing=summarize_routing,
            log_path_counts=log_path_counts,
            bdre_metrics=bdre_metrics,
        )
        output["incremental_state"] = next_state
        if router_records is not None:
            output["router_space"] = {
                "records": router_records,
                "num_actions": self.config.route_pool_blocks + 1,
                "out_action": self.out_action,
            }
        if diagnostics.last_visualization is not None:
            output["bdre_visualization"] = diagnostics.last_visualization
        return output

    def _run_synchronous_prefix_chunk(
        self,
        input_ids: torch.Tensor,
        state: BDREIncrementalState,
        *,
        route_mode: str,
        route_targets: list[torch.Tensor],
        routing_constraints: Mapping[str, Any],
        routing_options: Mapping[str, Any],
        hard_exit: bool,
        router_probability: float | None,
        global_step: int,
        diagnostics: _BDREDiagnostics,
        record: bool,
        collect_router_space: bool,
        collect_bdre_visualization: bool,
    ) -> tuple[torch.Tensor, BDREIncrementalState, dict[str, Any], list[dict[str, Any]] | None]:
        batch, chunk = input_ids.shape
        start_position = state.cache.tokens
        hidden = self.token_embedding(input_ids)
        pre_states: list[IncrementalAttentionState] = []
        for block, block_state in zip(self.pre_blocks, state.pre):
            hidden, next_block_state = _chunk_transformer_block(block, hidden, block_state, start_position)
            pre_states.append(next_block_state)

        initial_position = self.position_table.initial(batch, input_ids.device)
        position = initial_position.unsqueeze(1).expand(-1, chunk, -1)
        route_info = self._empty_route_info(hidden, hard_exit, global_step, routing_options)
        router_records: list[dict[str, Any]] | None = [] if collect_router_space else None
        max_steps = len(route_targets) if route_mode in {"fixed", "pseudo"} else self.config.max_route_steps
        if max_steps > self.config.max_route_steps:
            raise ValueError("Synchronous-prefix route target exceeds max_route_steps.")
        route_shape = (batch, chunk)
        exited = torch.zeros(route_shape, dtype=torch.bool, device=input_ids.device)
        last_internal = torch.full(route_shape, -1, dtype=torch.long, device=input_ids.device)
        recur_length = torch.zeros(route_shape, dtype=torch.long, device=input_ids.device)
        has_writer = torch.zeros(route_shape, dtype=torch.bool, device=input_ids.device)
        writer_keys: list[torch.Tensor] = []
        writer_values: list[torch.Tensor] = []
        writer_blocks: list[torch.Tensor] = []
        writer_valid: list[torch.Tensor] = []
        step_compile_outputs: list[BDRECompileOutput] = []
        grouped_mm_weights = (
            self._synchronous_grouped_mm_weights()
            if self.bdre_config.dispatch_mode == "grouped_mm"
            else None
        )

        for step in range(max_steps):
            exited_before = exited
            router_position = self._router_position(position)
            router_embedding = (
                self.router.token_embedding(hidden, router_position) if router_records is not None else None
            )
            raw_logits = (
                self.router.logits_from_embedding(router_embedding)
                if router_embedding is not None
                else self.router.token_logits(hidden, router_position)
            )
            logits = self._apply_location_bias(raw_logits, position)
            logits = self._apply_route_logit_noise(logits, global_step, routing_options)
            logits = self._apply_route_constraints(logits, step, max_steps, routing_constraints)
            logits, recur_cap_mask = self._apply_self_recur_cap(
                logits,
                last_internal,
                recur_length,
                routing_constraints,
            )
            probs = F.softmax(logits, dim=-1)
            top_actions, top_weights = self._topk_actions(probs, 1)
            if route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets):
                target = route_targets[step]
                if target.dim() == 1:
                    target = target.unsqueeze(1).expand(-1, chunk)
                elif target.shape != route_shape:
                    raise ValueError("Synchronous-prefix route targets must have shape [batch] or [batch, chunk].")
            else:
                target = torch.full(route_shape, self.out_action, dtype=torch.long, device=input_ids.device)

            if route_mode in {"fixed", "pseudo"}:
                selected = target
            elif route_mode == "scheduled":
                selected, _ = self._scheduled_select(
                    logits,
                    target,
                    global_step,
                    router_probability,
                    routing_options,
                )
            elif route_mode == "free":
                selected = self._router_action(logits, routing_options)
            else:
                raise ValueError(f"Unknown route_mode: {route_mode}")

            if route_mode in {"free", "scheduled"}:
                selected, random_mask = self._apply_random_route_override(
                    selected,
                    global_step,
                    routing_options,
                    last_internal,
                    recur_length,
                    routing_constraints,
                )
            else:
                random_mask = torch.zeros_like(selected, dtype=torch.bool)
            selected, selected_cap_mask = self._enforce_self_recur_cap_on_selected(
                selected,
                logits,
                last_internal,
                recur_length,
                routing_constraints,
            )
            if self._force_final_exit(step, max_steps, routing_constraints):
                selected = torch.full_like(selected, self.out_action)
            selected = torch.where(exited, torch.full_like(selected, self.out_action), selected)
            missing_writer = ~has_writer & (selected == self.out_action) & ~exited
            internal_choice = logits[..., : self.config.route_pool_blocks].argmax(dim=-1)
            selected = torch.where(missing_writer, internal_choice, selected)

            exit_now = selected == self.out_action
            valid = (selected != self.out_action) & ~exited
            has_writer = has_writer | valid
            flat_hidden = hidden.reshape(batch * chunk, -1)
            flat_position = position.reshape(batch * chunk, -1)
            flat_selected = selected.reshape(-1)
            flat_valid = valid.reshape(-1)
            step_key: torch.Tensor | None = None
            step_value: torch.Tensor | None = None
            prepared: list[
                tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, _SynchronousActionGroup | None]
            ] = []
            grouped_mm_prepared: tuple[
                _SynchronousGroupedMMDispatch,
                torch.Tensor,
                torch.Tensor,
            ] | None = None
            if grouped_mm_weights is not None:
                dispatch = self._synchronous_grouped_mm_dispatch(
                    flat_selected,
                    chunk=chunk,
                )
                if dispatch is not None:
                    routed_input, query, canonical_key, canonical_value = self._prepare_synchronous_grouped_mm_step(
                        flat_hidden,
                        flat_position,
                        flat_selected,
                        dispatch,
                        grouped_mm_weights,
                    )
                    step_key = canonical_key.new_zeros((batch * chunk, self.bdre_config.key_dim))
                    step_value = canonical_value.new_zeros((batch * chunk, self.bdre_config.value_dim))
                    step_key = step_key.index_copy(0, dispatch.indexes, canonical_key)
                    step_value = step_value.index_copy(0, dispatch.indexes, canonical_value)
                    grouped_mm_prepared = (dispatch, routed_input, query)
            else:
                if self.bdre_config.synchronous_attention_backend == "shared_padded_explicit":
                    packed_groups = self._active_synchronous_action_groups(
                        flat_selected,
                        flat_valid,
                        chunk=chunk,
                    )
                    action_groups = [
                        (group.action, group.indexes, group)
                        for group in packed_groups
                    ]
                else:
                    reference_groups, _ = self._active_action_groups(flat_selected, flat_valid)
                    action_groups = [
                        (action, indexes, None)
                        for action, indexes in reference_groups
                    ]

                for action, indexes, pack_group in action_groups:
                    block = self.route_blocks[action]
                    projection = self.bdre_projections[action]
                    selected_hidden = flat_hidden.index_select(0, indexes).unsqueeze(1)
                    selected_position = flat_position.index_select(0, indexes)
                    routed_input = selected_hidden + block._position_bias(self._block_position(selected_position))
                    attn_input = block.block.attn_norm(routed_input)
                    attention = block.block.attn
                    dim = attn_input.size(-1)
                    query, key, value = attention.qkv(attn_input).chunk(3, dim=-1)
                    query = query.view(-1, 1, attention.n_heads, attention.head_dim).transpose(1, 2).squeeze(2)
                    key = key.view(-1, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
                    value = value.view(-1, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
                    canonical_key = projection.key_write(key.transpose(1, 2).reshape(-1, 1, dim)).squeeze(1)
                    canonical_value = projection.value_write(value.transpose(1, 2).reshape(-1, 1, dim)).squeeze(1)
                    if step_key is None or step_value is None:
                        step_key = canonical_key.new_zeros((batch * chunk, self.bdre_config.key_dim))
                        step_value = canonical_value.new_zeros((batch * chunk, self.bdre_config.value_dim))
                    step_key = step_key.index_copy(0, indexes, canonical_key)
                    step_value = step_value.index_copy(0, indexes, canonical_value)
                    prepared.append((action, indexes, routed_input, query, pack_group))

            if step_key is None or step_value is None:
                if not writer_keys or not writer_values:
                    raise RuntimeError("The first synchronous-prefix route step must produce a writer.")
                step_key = writer_keys[0].new_zeros((batch * chunk, self.bdre_config.key_dim))
                step_value = writer_values[0].new_zeros((batch * chunk, self.bdre_config.value_dim))
            writer_keys.append(step_key.view(batch, chunk, -1))
            writer_values.append(step_value.view(batch, chunk, -1))
            writer_blocks.append(selected.clamp(min=0, max=self.config.route_pool_blocks - 1))
            writer_valid.append(valid)
            compile_output, flat_writers = self._compile_synchronous_prefix_step(
                writer_keys,
                writer_values,
                writer_blocks,
                writer_valid,
                reader_step=step,
                diagnostics=diagnostics,
            )
            step_compile_outputs.append(compile_output)
            current_keys = compile_output.keys.view(
                batch,
                chunk,
                self.config.route_pool_blocks,
                self.bdre_config.key_dim,
            )
            current_values = compile_output.values.view(
                batch,
                chunk,
                self.config.route_pool_blocks,
                self.bdre_config.value_dim,
            )

            next_flat_hidden = flat_hidden
            if grouped_mm_prepared is not None:
                dispatch, routed_input, query = grouped_mm_prepared
                block_output = self._finish_synchronous_prefix_grouped_mm(
                    routed_input,
                    query,
                    dispatch=dispatch,
                    weights=grouped_mm_weights,
                    chunk=chunk,
                    start_position=start_position,
                    reader_step=step,
                    historical_state=state.cache,
                    current_keys=current_keys,
                    current_values=current_values,
                )
                next_flat_hidden = next_flat_hidden.index_copy(0, dispatch.indexes, block_output)
            else:
                for action, indexes, routed_input, query, pack_group in prepared:
                    block_output = self._finish_synchronous_prefix_block(
                        self.route_blocks[action],
                        self.bdre_projections[action],
                        routed_input,
                        query,
                        indexes=indexes,
                        chunk=chunk,
                        start_position=start_position,
                        reader_action=action,
                        reader_step=step,
                        historical_state=state.cache,
                        current_keys=current_keys,
                        current_values=current_values,
                        pack_group=pack_group,
                    )
                    next_flat_hidden = next_flat_hidden.index_copy(0, indexes, block_output.squeeze(1))
            hidden = next_flat_hidden.view(batch, chunk, -1)

            if hard_exit:
                exited = exited | exit_now
            position = self.position_table.by_action(selected)
            if record:
                record_logits = self._last_token_view(logits)
                record_probs = self._last_token_view(probs)
                record_selected = self._last_token_view(selected)
                record_target = self._last_token_view(target)
                record_position = self._last_token_view(position)
                route_info["route_logits"].append(record_logits)
                route_info["route_probs"].append(record_probs)
                route_info["selected_actions"].append(record_selected)
                route_info["topk_actions"].append(self._last_token_view(top_actions))
                route_info["topk_weights"].append(self._last_token_view(top_weights))
                route_info["used_weighted_fusion"].append(torch.zeros_like(record_selected, dtype=torch.bool))
                route_info["exit_flags"].append(self._last_token_view(exit_now))
                if route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets):
                    route_info["route_targets"].append(record_target)
                route_info["location_distance"].append(
                    self.position_table.location_distance(record_position, record_probs)
                )
                route_info["position_norms"].append(record_position.norm(dim=-1).mean())
                route_info["random_route_override_count"].append(
                    self._last_token_view(random_mask).to(hidden.dtype).sum()
                )
                route_info["self_recur_cap_count"].append(
                    (
                        self._last_token_view(recur_cap_mask)
                        | self._last_token_view(selected_cap_mask)
                    ).to(hidden.dtype).sum()
                )
            if router_records is not None and router_embedding is not None:
                router_records.append(
                    {
                        "step": int(step),
                        "embedding": self._last_token_view(router_embedding).detach(),
                        "raw_logits": self._last_token_view(raw_logits).detach(),
                        "effective_logits": self._last_token_view(logits).detach(),
                        "probs": self._last_token_view(probs).detach(),
                        "selected_actions": self._last_token_view(selected).detach(),
                        "top_actions": self._last_token_view(top_actions).detach(),
                        "top_weights": self._last_token_view(top_weights).detach(),
                        "random_route_override": self._last_token_view(random_mask).detach(),
                        "self_recur_cap_active": self._last_token_view(recur_cap_mask).detach(),
                        "exited_before": self._last_token_view(exited_before).detach(),
                        "exit_now": self._last_token_view(exit_now).detach(),
                    }
                )
            last_internal, recur_length = self._update_self_recur_state(
                selected,
                last_internal,
                recur_length,
            )
            if hard_exit and bool(exited.detach().all().cpu()):
                break

        if not writer_keys:
            raise RuntimeError("Synchronous-prefix routing produced no writer states.")
        for reader_step in range(len(step_compile_outputs), self.config.max_route_steps):
            compile_output, flat_writers = self._compile_synchronous_prefix_step(
                writer_keys,
                writer_values,
                writer_blocks,
                writer_valid,
                reader_step=reader_step,
                diagnostics=diagnostics,
            )
            step_compile_outputs.append(compile_output)

        flat_key, flat_value, flat_block, flat_valid = flat_writers
        started = time.perf_counter()
        final_compile = self.bdre_compiler.compile(
            flat_key,
            flat_value,
            flat_block,
            flat_valid,
            self._internal_block_positions(),
            collect_metrics=diagnostics.enabled,
        )
        diagnostics.record_compile(final_compile, time.perf_counter() - started)
        diagnostics.writer_counts.append(flat_valid.sum(dim=-1))
        block_keys = final_compile.keys.view(
            batch,
            chunk,
            self.config.route_pool_blocks,
            self.bdre_config.key_dim,
        )
        block_values = final_compile.values.view(
            batch,
            chunk,
            self.config.route_pool_blocks,
            self.bdre_config.value_dim,
        )
        historical_step_outputs = step_compile_outputs
        if self.bdre_config.depth_visibility_policy == "full_bank":
            historical_step_outputs = []
            for reader_step in range(self.config.max_route_steps):
                started = time.perf_counter()
                output = self.bdre_compiler.compile(
                    flat_key,
                    flat_value,
                    flat_block,
                    flat_valid,
                    self._internal_block_positions(),
                    reader_step=reader_step,
                    collect_metrics=diagnostics.enabled,
                )
                diagnostics.record_compile(output, time.perf_counter() - started)
                historical_step_outputs.append(output)
        step_keys = torch.stack(
            [
                output.keys.view(
                    batch,
                    chunk,
                    self.config.route_pool_blocks,
                    self.bdre_config.key_dim,
                )
                for output in historical_step_outputs
            ],
            dim=2,
        )
        step_values = torch.stack(
            [
                output.values.view(
                    batch,
                    chunk,
                    self.config.route_pool_blocks,
                    self.bdre_config.value_dim,
                )
                for output in historical_step_outputs
            ],
            dim=2,
        )
        next_cache = state.cache.append_tokens(
            block_keys,
            block_values,
            step_key=step_keys,
            step_value=step_values,
        )
        if collect_bdre_visualization:
            last_rows = torch.arange(batch, device=input_ids.device) * chunk + (chunk - 1)
            diagnostics.last_visualization = {
                "writer_blocks": flat_block[last_rows].detach().cpu(),
                "writer_valid": flat_valid[last_rows].detach().cpu(),
                "key_weights": final_compile.key_weights[last_rows].detach().cpu(),
                "value_weights": final_compile.value_weights[last_rows].detach().cpu(),
                "reader_step_key_weights": torch.stack(
                    [output.key_weights[last_rows] for output in step_compile_outputs],
                    dim=1,
                ).detach().cpu(),
                "reader_step_value_weights": torch.stack(
                    [output.value_weights[last_rows] for output in step_compile_outputs],
                    dim=1,
                ).detach().cpu(),
                "historical_reader_step_key_weights": torch.stack(
                    [output.key_weights[last_rows] for output in historical_step_outputs],
                    dim=1,
                ).detach().cpu(),
                "historical_reader_step_value_weights": torch.stack(
                    [output.value_weights[last_rows] for output in historical_step_outputs],
                    dim=1,
                ).detach().cpu(),
                "depth_visibility_policy": self.bdre_config.depth_visibility_policy,
                "block_positions": self._internal_block_positions().detach().cpu(),
            }

        out_actions = torch.full(route_shape, self.out_action, dtype=torch.long, device=input_ids.device)
        out_position = self._block_position(self.position_table.by_action(out_actions))
        hidden = self.exit_block(hidden, out_position)
        post_states: list[IncrementalAttentionState] = []
        for block, block_state in zip(self.post_blocks, state.post):
            hidden, next_block_state = _chunk_transformer_block(block, hidden, block_state, start_position)
            post_states.append(next_block_state)
        logits_out = self.lm_head(self.norm(hidden))
        next_state = BDREIncrementalState(tuple(pre_states), tuple(post_states), next_cache)
        return logits_out, next_state, route_info, router_records

    def _synchronous_grouped_mm_weights(self) -> _SynchronousGroupedMMWeights:
        if not self.route_blocks:
            raise ValueError("grouped_mm dispatch requires at least one route block.")
        first = self.route_blocks[0]
        if any(block.position_injection != first.position_injection for block in self.route_blocks):
            raise ValueError("grouped_mm dispatch requires a common position injection mode.")

        first_parameter = next(self.route_blocks[0].parameters())
        grouped_dtype = first_parameter.dtype
        if first_parameter.is_cuda and torch.is_autocast_enabled("cuda"):
            grouped_dtype = torch.get_autocast_dtype("cuda")

        def grouped_matrix(parameters: list[torch.Tensor]) -> torch.Tensor:
            matrix = torch.stack(parameters).transpose(-1, -2).contiguous()
            return matrix.to(dtype=grouped_dtype)

        position_adapter = None
        if first.position_injection != "direct_add":
            position_weights = []
            for block in self.route_blocks:
                if block.position_adapter is None:
                    raise ValueError("grouped_mm dispatch is missing a position adapter.")
                position_weights.append(block.position_adapter.weight)
            position_adapter = grouped_matrix(position_weights)

        return _SynchronousGroupedMMWeights(
            position_adapter=position_adapter,
            attn_norm=torch.stack([block.block.attn_norm.weight for block in self.route_blocks]),
            qkv=grouped_matrix([block.block.attn.qkv.weight for block in self.route_blocks]),
            key_write=grouped_matrix([projection.key_write.weight for projection in self.bdre_projections]),
            value_write=grouped_matrix([projection.value_write.weight for projection in self.bdre_projections]),
            attention_out=grouped_matrix([block.block.attn.out.weight for block in self.route_blocks]),
            ffn_norm=torch.stack([block.block.ffn_norm.weight for block in self.route_blocks]),
            ffn_gate_up=grouped_matrix(
                [
                    torch.cat((block.block.ffn.w1.weight, block.block.ffn.w2.weight), dim=0)
                    for block in self.route_blocks
                ]
            ),
            ffn_down=grouped_matrix([block.block.ffn.w3.weight for block in self.route_blocks]),
        )

    def _synchronous_grouped_mm_dispatch(
        self,
        selected: torch.Tensor,
        *,
        chunk: int,
    ) -> _SynchronousGroupedMMDispatch | None:
        selected_host = selected.detach().to(device="cpu")
        index_parts: list[torch.Tensor] = []
        group_metadata: list[
            tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []
        counts: list[int] = []
        cursor = 0
        for action in range(self.config.route_pool_blocks):
            indexes_host = torch.nonzero(
                selected_host == action,
                as_tuple=False,
            ).flatten()
            count = int(indexes_host.numel())
            counts.append(count)
            if not count:
                continue
            index_parts.append(indexes_host)
            batch_indexes = torch.div(indexes_host, chunk, rounding_mode="floor")
            active_batches, batch_counts = torch.unique_consecutive(batch_indexes, return_counts=True)
            active_rows = torch.repeat_interleave(torch.arange(active_batches.numel()), batch_counts)
            row_starts = torch.cumsum(batch_counts, dim=0) - batch_counts
            local_slots = torch.arange(count) - row_starts[active_rows]
            max_selected = int(batch_counts.max().item())
            group_metadata.append(
                (action, cursor, active_batches, active_rows, local_slots, max_selected)
            )
            cursor += count
        if not index_parts:
            return None

        indexes = torch.cat(index_parts).to(device=selected.device)
        actions = selected.index_select(0, indexes)
        offsets = torch.tensor(counts, dtype=torch.int32, device=selected.device).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        groups: list[_SynchronousActionGroup] = []
        for action, start, active_batches, active_rows, local_slots, max_selected in group_metadata:
            end = start + counts[action]
            flat_slots = active_rows * max_selected + local_slots
            groups.append(
                _SynchronousActionGroup(
                    action=action,
                    indexes=indexes[start:end],
                    active_batches=active_batches.to(device=selected.device),
                    flat_slots=flat_slots.to(device=selected.device),
                    max_selected=max_selected,
                )
            )
        return _SynchronousGroupedMMDispatch(
            indexes=indexes,
            actions=actions,
            offsets=offsets,
            groups=tuple(groups),
        )

    @staticmethod
    def _grouped_mm_linear(
        inputs: torch.Tensor,
        matrices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        grouped_mm = getattr(F, "grouped_mm", None)
        if inputs.is_cuda and matrices.dtype == torch.bfloat16 and grouped_mm is not None:
            return grouped_mm(inputs.to(dtype=matrices.dtype), matrices, offs=offsets)

        ends = offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
        outputs: list[torch.Tensor] = []
        start = 0
        for action, end in enumerate(ends):
            outputs.append(inputs[start:end] @ matrices[action])
            start = end
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _grouped_mm_rms_norm(
        inputs: torch.Tensor,
        norm_weights: torch.Tensor,
        actions: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        selected_weights = F.embedding(actions, norm_weights).to(dtype=inputs.dtype)
        scale = torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + eps)
        return selected_weights * inputs * scale

    def _prepare_synchronous_grouped_mm_step(
        self,
        flat_hidden: torch.Tensor,
        flat_position: torch.Tensor,
        flat_selected: torch.Tensor,
        dispatch: _SynchronousGroupedMMDispatch,
        weights: _SynchronousGroupedMMWeights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del flat_selected
        selected_hidden = flat_hidden.index_select(0, dispatch.indexes)
        selected_position = self._block_position(flat_position.index_select(0, dispatch.indexes))
        if weights.position_adapter is None:
            position_bias = selected_position
        else:
            position_bias = self._grouped_mm_linear(
                selected_position,
                weights.position_adapter,
                dispatch.offsets,
            )
        routed_input = selected_hidden + position_bias
        attn_input = self._grouped_mm_rms_norm(
            routed_input,
            weights.attn_norm,
            dispatch.actions,
            self.route_blocks[0].block.attn_norm.eps,
        )
        qkv = self._grouped_mm_linear(attn_input, weights.qkv, dispatch.offsets)
        dim = attn_input.size(-1)
        query, key, value = qkv.chunk(3, dim=-1)
        attention = self.route_blocks[0].block.attn
        query = query.view(-1, attention.n_heads, attention.head_dim)
        canonical_key = self._grouped_mm_linear(key, weights.key_write, dispatch.offsets)
        canonical_value = self._grouped_mm_linear(value, weights.value_write, dispatch.offsets)
        return routed_input, query, canonical_key, canonical_value

    def _finish_synchronous_prefix_grouped_mm(
        self,
        routed_input: torch.Tensor,
        query: torch.Tensor,
        *,
        dispatch: _SynchronousGroupedMMDispatch,
        weights: _SynchronousGroupedMMWeights,
        chunk: int,
        start_position: int,
        reader_step: int,
        historical_state: BDRECacheState,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
    ) -> torch.Tensor:
        attended_all = self._read_synchronous_prefix_grouped_mm_per_action(
            routed_input,
            query,
            dispatch=dispatch,
            chunk=chunk,
            start_position=start_position,
            reader_step=reader_step,
            historical_state=historical_state,
            current_keys=current_keys,
            current_values=current_values,
        )
        attention_output = self._grouped_mm_linear(
            attended_all,
            weights.attention_out,
            dispatch.offsets,
        )
        routed = routed_input + attention_output
        ffn_input = self._grouped_mm_rms_norm(
            routed,
            weights.ffn_norm,
            dispatch.actions,
            self.route_blocks[0].block.ffn_norm.eps,
        )
        gate, up = self._grouped_mm_linear(
            ffn_input,
            weights.ffn_gate_up,
            dispatch.offsets,
        ).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return routed + self._grouped_mm_linear(hidden, weights.ffn_down, dispatch.offsets)

    def _read_synchronous_prefix_grouped_mm_per_action(
        self,
        routed_input: torch.Tensor,
        query: torch.Tensor,
        *,
        dispatch: _SynchronousGroupedMMDispatch,
        chunk: int,
        start_position: int,
        reader_step: int,
        historical_state: BDRECacheState,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
    ) -> torch.Tensor:
        attended_parts: list[torch.Tensor] = []
        cursor = 0
        for group in dispatch.groups:
            end = cursor + group.indexes.numel()
            attended = self._read_synchronous_prefix_block_shared_explicit(
                self.route_blocks[group.action],
                self.bdre_projections[group.action],
                routed_input[cursor:end].unsqueeze(1),
                query[cursor:end],
                indexes=group.indexes,
                chunk=chunk,
                start_position=start_position,
                reader_action=group.action,
                reader_step=reader_step,
                historical_state=historical_state,
                current_keys=current_keys,
                current_values=current_values,
                pack_group=group,
            )
            attended_parts.append(attended.squeeze(1))
            cursor = end
        return torch.cat(attended_parts, dim=0)

    def _compile_synchronous_prefix_step(
        self,
        writer_keys: list[torch.Tensor],
        writer_values: list[torch.Tensor],
        writer_blocks: list[torch.Tensor],
        writer_valid: list[torch.Tensor],
        *,
        reader_step: int,
        diagnostics: _BDREDiagnostics,
    ) -> tuple[BDRECompileOutput, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch, chunk = writer_valid[0].shape
        steps = len(writer_keys)
        key = torch.stack(writer_keys, dim=2)
        value = torch.stack(writer_values, dim=2)
        block = torch.stack(writer_blocks, dim=2)
        valid = torch.stack(writer_valid, dim=2)
        padding = self.config.max_route_steps - steps
        if padding < 0:
            raise ValueError("Synchronous-prefix writer history exceeds max_route_steps.")
        if padding:
            key = F.pad(key, (0, 0, 0, padding))
            value = F.pad(value, (0, 0, 0, padding))
            block = F.pad(block, (0, padding))
            valid = F.pad(valid, (0, padding), value=False)
        flat_key = key.reshape(batch * chunk, self.config.max_route_steps, self.bdre_config.key_dim)
        flat_value = value.reshape(batch * chunk, self.config.max_route_steps, self.bdre_config.value_dim)
        flat_block = block.reshape(batch * chunk, self.config.max_route_steps)
        flat_valid = valid.reshape(batch * chunk, self.config.max_route_steps)
        started = time.perf_counter()
        output = self.bdre_compiler.compile_prefix(
            flat_key,
            flat_value,
            flat_block,
            flat_valid,
            self._internal_block_positions(),
            reader_step=reader_step,
            collect_metrics=diagnostics.enabled,
        )
        diagnostics.record_compile(output, time.perf_counter() - started)
        return output, (flat_key, flat_value, flat_block, flat_valid)

    def _finish_synchronous_prefix_block(
        self,
        block: Any,
        projection: BDREBlockProjection,
        routed_input: torch.Tensor,
        query: torch.Tensor,
        *,
        indexes: torch.Tensor,
        chunk: int,
        start_position: int,
        reader_action: int,
        reader_step: int,
        historical_state: BDRECacheState,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        pack_group: _SynchronousActionGroup | None = None,
    ) -> torch.Tensor:
        if self.bdre_config.synchronous_attention_backend == "shared_padded_explicit":
            if pack_group is None:
                raise ValueError("Shared padded attention requires synchronous action packing metadata.")
            return self._finish_synchronous_prefix_block_shared_explicit(
                block,
                projection,
                routed_input,
                query,
                indexes=indexes,
                chunk=chunk,
                start_position=start_position,
                reader_action=reader_action,
                reader_step=reader_step,
                historical_state=historical_state,
                current_keys=current_keys,
                current_values=current_values,
                pack_group=pack_group,
            )
        batch_indexes = torch.div(indexes, chunk, rounding_mode="floor")
        local_positions = indexes % chunk
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if start_position:
            if historical_state.step_keys is None or historical_state.step_values is None:
                raise ValueError("Synchronous-prefix history is missing step cache tensors.")
            key_parts.append(historical_state.step_keys[batch_indexes, :, reader_step, reader_action, :])
            value_parts.append(historical_state.step_values[batch_indexes, :, reader_step, reader_action, :])
        key_parts.append(current_keys[batch_indexes, :, reader_action, :])
        value_parts.append(current_values[batch_indexes, :, reader_action, :])
        key_codes = torch.cat(key_parts, dim=1)
        value_codes = torch.cat(value_parts, dim=1)
        attention = block.block.attn
        decoded_key = torch.einsum(
            "nkr,hrd->nhkd",
            key_codes.to(dtype=projection.key_read.dtype),
            projection.key_read,
        ).to(dtype=query.dtype)
        key_positions = torch.arange(key_codes.size(1), device=query.device)
        key_cosine = attention.rope.cos[:, :, key_positions, :].to(device=query.device, dtype=query.dtype)
        key_sine = attention.rope.sin[:, :, key_positions, :].to(device=query.device, dtype=query.dtype)
        decoded_key = apply_rotary(decoded_key, key_cosine, key_sine)
        query_positions = start_position + local_positions
        query_cosine = attention.rope.cos[0, 0, query_positions, :].to(device=query.device, dtype=query.dtype)
        query_sine = attention.rope.sin[0, 0, query_positions, :].to(device=query.device, dtype=query.dtype)
        query = apply_rotary(query, query_cosine.unsqueeze(1), query_sine.unsqueeze(1))
        scores = torch.einsum("nhd,nhkd->nhk", query, decoded_key) * (attention.head_dim**-0.5)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        weights = F.dropout(weights, p=attention.dropout, training=block.training)
        latent_value = torch.einsum(
            "nhk,nkr->nhr",
            weights,
            value_codes.to(dtype=weights.dtype),
        )
        attended = torch.einsum(
            "nhr,hrd->nhd",
            latent_value.to(dtype=projection.value_read.dtype),
            projection.value_read,
        ).to(dtype=routed_input.dtype)
        attended = attended.reshape(routed_input.size(0), 1, -1)
        routed = routed_input + attention.out(attended)
        return routed + block.block.ffn(block.block.ffn_norm(routed))

    def _finish_synchronous_prefix_block_shared_explicit(
        self,
        block: Any,
        projection: BDREBlockProjection,
        routed_input: torch.Tensor,
        query: torch.Tensor,
        *,
        indexes: torch.Tensor,
        chunk: int,
        start_position: int,
        reader_action: int,
        reader_step: int,
        historical_state: BDRECacheState,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        pack_group: _SynchronousActionGroup,
    ) -> torch.Tensor:
        attended = self._read_synchronous_prefix_block_shared_explicit(
            block,
            projection,
            routed_input,
            query,
            indexes=indexes,
            chunk=chunk,
            start_position=start_position,
            reader_action=reader_action,
            reader_step=reader_step,
            historical_state=historical_state,
            current_keys=current_keys,
            current_values=current_values,
            pack_group=pack_group,
        )
        attention = block.block.attn
        routed = routed_input + attention.out(attended)
        return routed + block.block.ffn(block.block.ffn_norm(routed))

    def _read_synchronous_prefix_block_shared_explicit(
        self,
        block: Any,
        projection: BDREBlockProjection,
        routed_input: torch.Tensor,
        query: torch.Tensor,
        *,
        indexes: torch.Tensor,
        chunk: int,
        start_position: int,
        reader_action: int,
        reader_step: int,
        historical_state: BDRECacheState,
        current_keys: torch.Tensor,
        current_values: torch.Tensor,
        pack_group: _SynchronousActionGroup,
    ) -> torch.Tensor:
        local_positions = indexes % chunk
        active_batches = pack_group.active_batches
        flat_slots = pack_group.flat_slots
        max_selected = pack_group.max_selected

        attention = block.block.attn
        query_positions = start_position + local_positions
        query_cosine = attention.rope.cos[0, 0, query_positions, :].to(device=query.device, dtype=query.dtype)
        query_sine = attention.rope.sin[0, 0, query_positions, :].to(device=query.device, dtype=query.dtype)
        query = apply_rotary(query, query_cosine.unsqueeze(1), query_sine.unsqueeze(1))
        padded_query = query.new_zeros(
            (active_batches.numel() * max_selected, attention.n_heads, attention.head_dim)
        ).index_copy(0, flat_slots, query)
        padded_query = padded_query.view(
            active_batches.numel(),
            max_selected,
            attention.n_heads,
            attention.head_dim,
        ).transpose(1, 2)
        padded_positions = torch.zeros(
            active_batches.numel() * max_selected,
            dtype=torch.long,
            device=query.device,
        ).index_copy(0, flat_slots, query_positions)
        padded_positions = padded_positions.view(active_batches.numel(), max_selected)

        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if start_position:
            if historical_state.step_keys is None or historical_state.step_values is None:
                raise ValueError("Synchronous-prefix history is missing step cache tensors.")
            key_parts.append(
                historical_state.step_keys[
                    active_batches,
                    :,
                    reader_step,
                    reader_action,
                    :,
                ]
            )
            value_parts.append(
                historical_state.step_values[
                    active_batches,
                    :,
                    reader_step,
                    reader_action,
                    :,
                ]
            )
        key_parts.append(current_keys[active_batches, :, reader_action, :])
        value_parts.append(current_values[active_batches, :, reader_action, :])
        key_codes = torch.cat(key_parts, dim=1)
        value_codes = torch.cat(value_parts, dim=1)

        key_read_weight = projection.key_read.permute(0, 2, 1).reshape(
            attention.n_heads * attention.head_dim,
            self.bdre_config.key_dim,
        )
        decoded_key = F.linear(
            key_codes.to(dtype=projection.key_read.dtype),
            key_read_weight,
        ).view(
            active_batches.numel(),
            key_codes.size(1),
            attention.n_heads,
            attention.head_dim,
        ).permute(0, 2, 1, 3).to(dtype=query.dtype)
        key_positions = torch.arange(key_codes.size(1), device=query.device)
        key_cosine = attention.rope.cos[:, :, key_positions, :].to(device=query.device, dtype=query.dtype)
        key_sine = attention.rope.sin[:, :, key_positions, :].to(device=query.device, dtype=query.dtype)
        decoded_key = apply_rotary(decoded_key, key_cosine, key_sine)
        allowed = key_positions.view(1, 1, 1, -1) <= padded_positions.view(
            active_batches.numel(),
            1,
            max_selected,
            1,
        )
        scores = torch.einsum("bhqd,bhkd->bhqk", padded_query, decoded_key) * (
            attention.head_dim**-0.5
        )
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        weights = F.dropout(weights, p=attention.dropout, training=block.training)
        latent_value = torch.einsum(
            "bhqk,bkr->bhqr",
            weights,
            value_codes.to(dtype=weights.dtype),
        )
        latent_value = latent_value.transpose(1, 2).reshape(
            active_batches.numel() * max_selected,
            attention.n_heads,
            self.bdre_config.value_dim,
        ).index_select(0, flat_slots)
        attended = torch.einsum(
            "nhr,hrd->nhd",
            latent_value.to(dtype=projection.value_read.dtype),
            projection.value_read,
        ).to(dtype=routed_input.dtype)
        attended = attended.reshape(routed_input.size(0), 1, -1)
        return attended

    def _forward_one_token(
        self,
        input_ids: torch.Tensor,
        state: BDREIncrementalState,
        *,
        token_position: int,
        route_mode: str,
        route_targets: list[torch.Tensor],
        routing_constraints: Mapping[str, Any],
        routing_options: Mapping[str, Any],
        hard_exit: bool,
        router_probability: float | None,
        global_step: int,
        diagnostics: _BDREDiagnostics,
        record: bool,
        collect_router_space: bool,
        collect_bdre_visualization: bool,
    ) -> tuple[torch.Tensor, BDREIncrementalState, dict[str, Any], list[dict[str, Any]] | None]:
        batch = input_ids.size(0)
        hidden = self.token_embedding(input_ids)
        pre_states: list[IncrementalAttentionState] = []
        for block, block_state in zip(self.pre_blocks, state.pre):
            hidden, next_block_state = _incremental_transformer_block(block, hidden, block_state, token_position)
            pre_states.append(next_block_state)

        position = self.position_table.initial(batch, input_ids.device)
        route_info = self._empty_route_info(hidden, hard_exit, global_step, routing_options)
        router_records: list[dict[str, Any]] | None = [] if collect_router_space else None
        max_steps = len(route_targets) if route_mode in {"fixed", "pseudo"} else self.config.max_route_steps
        exited = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        last_internal = torch.full((batch,), -1, dtype=torch.long, device=input_ids.device)
        recur_length = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        writer_keys: list[torch.Tensor] = []
        writer_values: list[torch.Tensor] = []
        writer_blocks: list[torch.Tensor] = []
        writer_valid: list[torch.Tensor] = []
        has_writer = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        exited_host = torch.zeros(batch, dtype=torch.bool) if self.bdre_config.dispatch_mode == "grouped_host" else None

        for step in range(max_steps):
            exited_before = exited
            router_position = self._router_position(position)
            router_embedding = self.router.embedding(hidden, router_position) if router_records is not None else None
            raw_logits = (
                self.router.logits_from_embedding(router_embedding)
                if router_embedding is not None
                else self.router(hidden, router_position)
            )
            logits = self._apply_location_bias(raw_logits, position)
            logits = self._apply_route_logit_noise(logits, global_step, routing_options)
            logits = self._apply_route_constraints(logits, step, max_steps, routing_constraints)
            logits, recur_cap_mask = self._apply_self_recur_cap(
                logits,
                last_internal,
                recur_length,
                routing_constraints,
            )
            probs = F.softmax(logits, dim=-1)
            top_actions, top_weights = self._topk_actions(probs, 1)
            target = (
                route_targets[step]
                if route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets)
                else torch.full((batch,), self.out_action, dtype=torch.long, device=input_ids.device)
            )
            if route_mode in {"fixed", "pseudo"}:
                selected = target
            elif route_mode == "scheduled":
                selected, _ = self._scheduled_select(
                    logits,
                    target,
                    global_step,
                    router_probability,
                    routing_options,
                )
            elif route_mode == "free":
                selected = self._router_action(logits, routing_options)
            else:
                raise ValueError(f"Unknown route_mode: {route_mode}")

            if route_mode in {"free", "scheduled"}:
                selected, random_mask = self._apply_random_route_override(
                    selected,
                    global_step,
                    routing_options,
                    last_internal,
                    recur_length,
                    routing_constraints,
                )
            else:
                random_mask = torch.zeros_like(selected, dtype=torch.bool)
            selected, selected_cap_mask = self._enforce_self_recur_cap_on_selected(
                selected,
                logits,
                last_internal,
                recur_length,
                routing_constraints,
            )
            if self._force_final_exit(step, max_steps, routing_constraints):
                selected = torch.full_like(selected, self.out_action)
            selected = torch.where(exited, torch.full_like(selected, self.out_action), selected)

            missing_writer = ~has_writer & (selected == self.out_action) & ~exited
            internal_choice = logits[..., : self.config.route_pool_blocks].argmax(dim=-1)
            selected = torch.where(missing_writer, internal_choice, selected)

            exit_now = selected == self.out_action
            next_hidden = hidden
            step_key = writer_keys[0].new_zeros((batch, self.bdre_config.key_dim)) if writer_keys else None
            step_value = writer_values[0].new_zeros((batch, self.bdre_config.value_dim)) if writer_values else None
            valid = (selected != self.out_action) & ~exited
            has_writer = has_writer | valid
            previous_key_all = torch.stack(writer_keys, dim=1) if writer_keys else None
            previous_value_all = torch.stack(writer_values, dim=1) if writer_values else None
            previous_blocks_all = torch.stack(writer_blocks, dim=1) if writer_blocks else None
            previous_valid_all = torch.stack(writer_valid, dim=1) if writer_valid else None
            action_groups, selected_host = self._active_action_groups(selected, valid)
            for action, indexes in action_groups:
                block = self.route_blocks[action]
                projection = self.bdre_projections[action]
                historical_key, historical_value = self._historical_reader_codes(
                    state.cache,
                    action=action,
                    reader_step=step,
                    indexes=indexes,
                    diagnostics=diagnostics,
                )
                previous_key = previous_key_all[indexes] if previous_key_all is not None else None
                previous_value = previous_value_all[indexes] if previous_value_all is not None else None
                previous_blocks = previous_blocks_all[indexes] if previous_blocks_all is not None else None
                previous_valid = previous_valid_all[indexes] if previous_valid_all is not None else None
                block_output, canonical_key, canonical_value = self._run_bdre_block(
                    block,
                    projection,
                    hidden[indexes],
                    position[indexes],
                    historical_key,
                    historical_value,
                    previous_key,
                    previous_value,
                    previous_blocks,
                    previous_valid,
                    reader_action=action,
                    reader_step=step,
                    token_position=token_position,
                    diagnostics=diagnostics,
                )
                if step_key is None or step_value is None:
                    step_key = canonical_key.new_zeros((batch, self.bdre_config.key_dim))
                    step_value = canonical_value.new_zeros((batch, self.bdre_config.value_dim))
                next_hidden = next_hidden.index_copy(0, indexes, block_output)
                step_key = step_key.index_copy(0, indexes, canonical_key)
                step_value = step_value.index_copy(0, indexes, canonical_value)

            if step_key is None or step_value is None:
                step_key = hidden.new_zeros((batch, self.bdre_config.key_dim))
                step_value = hidden.new_zeros((batch, self.bdre_config.value_dim))
            writer_keys.append(step_key)
            writer_values.append(step_value)
            writer_blocks.append(selected.clamp(min=0, max=self.config.route_pool_blocks - 1))
            writer_valid.append(valid)
            if hard_exit:
                exited = exited | exit_now
            hidden = next_hidden
            position = self.position_table.by_action(selected)

            if record:
                route_info["route_logits"].append(logits)
                route_info["route_probs"].append(probs)
                route_info["selected_actions"].append(selected)
                route_info["topk_actions"].append(top_actions)
                route_info["topk_weights"].append(top_weights)
                route_info["used_weighted_fusion"].append(torch.zeros_like(selected, dtype=torch.bool))
                route_info["exit_flags"].append(exit_now)
                if route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets):
                    route_info["route_targets"].append(target)
                route_info["location_distance"].append(self.position_table.location_distance(position, probs))
                route_info["position_norms"].append(position.norm(dim=-1).mean())
                route_info["random_route_override_count"].append(random_mask.to(hidden.dtype).sum())
                route_info["self_recur_cap_count"].append(
                    (recur_cap_mask | selected_cap_mask).to(hidden.dtype).sum()
                )
            if router_records is not None and router_embedding is not None:
                router_records.append(
                    {
                        "step": int(step),
                        "embedding": router_embedding.detach(),
                        "raw_logits": raw_logits.detach(),
                        "effective_logits": logits.detach(),
                        "probs": probs.detach(),
                        "selected_actions": selected.detach(),
                        "top_actions": top_actions.detach(),
                        "top_weights": top_weights.detach(),
                        "random_route_override": random_mask.detach(),
                        "self_recur_cap_active": recur_cap_mask.detach(),
                        "exited_before": exited_before.detach(),
                        "exit_now": exit_now.detach(),
                    }
                )
            last_internal, recur_length = self._update_self_recur_state(
                selected,
                last_internal,
                recur_length,
            )
            if hard_exit:
                if selected_host is not None and exited_host is not None:
                    exited_host |= selected_host.eq(self.out_action)
                    if bool(exited_host.all()):
                        break
                elif torch.all(exited):
                    break

        next_cache = self._compile_completed_token(
            state.cache,
            writer_keys,
            writer_values,
            writer_blocks,
            writer_valid,
            diagnostics,
            capture_visualization=collect_bdre_visualization,
        )
        out_actions = torch.full((batch,), self.out_action, dtype=torch.long, device=input_ids.device)
        out_position = self._block_position(self.position_table.by_action(out_actions))
        hidden = self.exit_block(hidden, out_position)
        post_states: list[IncrementalAttentionState] = []
        for block, block_state in zip(self.post_blocks, state.post):
            hidden, next_block_state = _incremental_transformer_block(block, hidden, block_state, token_position)
            post_states.append(next_block_state)
        logits_out = self.lm_head(self.norm(hidden))
        next_state = BDREIncrementalState(tuple(pre_states), tuple(post_states), next_cache)
        return logits_out, next_state, route_info, router_records

    def _empty_route_info(
        self,
        hidden: torch.Tensor,
        hard_exit: bool,
        global_step: int,
        routing_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "route_logits": [],
            "route_probs": [],
            "selected_actions": [],
            "topk_actions": [],
            "topk_weights": [],
            "used_weighted_fusion": [],
            "exit_flags": [],
            "route_targets": [],
            "position_norms": [],
            "location_distance": [],
            "random_route_override_count": [],
            "self_recur_cap_count": [],
            "random_route_probability": torch.tensor(
                self._random_route_probability(global_step, routing_options),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            "route_logit_noise_std": torch.tensor(
                self._route_logit_noise_std(global_step, routing_options),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
            "hard_exit_enabled": bool(hard_exit),
            "max_route_steps": self.config.max_route_steps,
        }

    def _active_synchronous_action_groups(
        self,
        selected: torch.Tensor,
        valid: torch.Tensor,
        *,
        chunk: int,
    ) -> list[_SynchronousActionGroup]:
        if self.bdre_config.dispatch_mode == "grouped_host":
            selected_host = selected.detach().to(device="cpu")
            active_host = selected_host != self.out_action
            actions = torch.unique(selected_host[active_host], sorted=True).tolist()
            groups: list[_SynchronousActionGroup] = []
            for action_value in actions:
                action = int(action_value)
                indexes_host = torch.nonzero(selected_host == action, as_tuple=False).flatten()
                batch_indexes = torch.div(indexes_host, chunk, rounding_mode="floor")
                active_batches, counts = torch.unique_consecutive(batch_indexes, return_counts=True)
                active_rows = torch.repeat_interleave(torch.arange(active_batches.numel()), counts)
                row_starts = torch.cumsum(counts, dim=0) - counts
                local_slots = torch.arange(indexes_host.numel()) - row_starts[active_rows]
                max_selected = int(counts.max().item())
                groups.append(
                    _SynchronousActionGroup(
                        action=action,
                        indexes=indexes_host.to(device=selected.device),
                        active_batches=active_batches.to(device=selected.device),
                        flat_slots=(active_rows * max_selected + local_slots).to(device=selected.device),
                        max_selected=max_selected,
                    )
                )
            return groups

        action_groups, _ = self._active_action_groups(selected, valid)
        groups: list[_SynchronousActionGroup] = []
        for action, indexes in action_groups:
            batch_indexes = torch.div(indexes, chunk, rounding_mode="floor")
            active_batches, counts = torch.unique_consecutive(batch_indexes, return_counts=True)
            active_rows = torch.repeat_interleave(
                torch.arange(active_batches.numel(), device=indexes.device),
                counts,
            )
            row_starts = torch.cumsum(counts, dim=0) - counts
            local_slots = torch.arange(indexes.numel(), device=indexes.device) - row_starts[active_rows]
            max_selected = int(counts.max().item())
            groups.append(
                _SynchronousActionGroup(
                    action=action,
                    indexes=indexes,
                    active_batches=active_batches,
                    flat_slots=active_rows * max_selected + local_slots,
                    max_selected=max_selected,
                )
            )
        return groups

    def _active_action_groups(
        self,
        selected: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[list[tuple[int, torch.Tensor]], torch.Tensor | None]:
        if self.bdre_config.dispatch_mode == "legacy_cuda_scan":
            groups: list[tuple[int, torch.Tensor]] = []
            for action in range(self.config.route_pool_blocks):
                action_mask = valid & (selected == action)
                if not torch.any(action_mask):
                    continue
                groups.append((action, torch.nonzero(action_mask, as_tuple=False).flatten()))
            return groups, None

        selected_host = selected.detach().to(device="cpu")
        active_host = selected_host != self.out_action
        actions = torch.unique(selected_host[active_host], sorted=True).tolist()
        groups = [
            (
                int(action),
                torch.nonzero(selected_host == action, as_tuple=False).flatten().to(device=selected.device),
            )
            for action in actions
        ]
        return groups, selected_host

    def _internal_block_positions(self) -> torch.Tensor:
        return F.normalize(self.position_table.embeddings[: self.config.route_pool_blocks], dim=-1)

    def _historical_reader_codes(
        self,
        state: BDRECacheState,
        *,
        action: int,
        reader_step: int,
        indexes: torch.Tensor,
        diagnostics: _BDREDiagnostics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.tokens == 0:
            batch = indexes.numel()
            reference = self.position_table.embeddings
            return (
                reference.new_empty((batch, 0, self.bdre_config.key_dim)),
                reference.new_empty((batch, 0, self.bdre_config.value_dim)),
            )
        if self.bdre_config.depth_mode == "none":
            keys, values = state.stacked_blocks()
            return keys[indexes, :, action, :], values[indexes, :, action, :]
        if self.bdre_config.reader_step_cache == "eager":
            keys, values = state.stacked_steps()
            return keys[indexes, :, reader_step, action, :], values[indexes, :, reader_step, action, :]

        cache_key = (action, reader_step)
        diagnostics.lazy_requests += int(self.bdre_config.reader_step_cache == "lazy")
        if self.bdre_config.reader_step_cache == "lazy" and cache_key in state.lazy_cache:
            diagnostics.lazy_hits += 1
            keys, values = state.lazy_cache[cache_key]
        else:
            writer_key, writer_value, writer_block, writer_valid = state.stacked_writers()
            started = time.perf_counter()
            keys, values = self.bdre_compiler.compile_history_for_reader(
                writer_key,
                writer_value,
                writer_block,
                writer_valid,
                self._internal_block_positions(),
                reader_action=action,
                reader_step=reader_step,
            )
            diagnostics.compile_seconds += time.perf_counter() - started
            if self.bdre_config.reader_step_cache == "lazy":
                state.lazy_cache[cache_key] = (keys, values)
        return keys[indexes], values[indexes]

    def _run_bdre_block(
        self,
        block: Any,
        projection: BDREBlockProjection,
        hidden: torch.Tensor,
        position: torch.Tensor,
        historical_key: torch.Tensor,
        historical_value: torch.Tensor,
        previous_key: torch.Tensor | None,
        previous_value: torch.Tensor | None,
        previous_blocks: torch.Tensor | None,
        previous_valid: torch.Tensor | None,
        *,
        reader_action: int,
        reader_step: int,
        token_position: int,
        diagnostics: _BDREDiagnostics,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_position = self._block_position(position)
        routed_input = hidden + block._position_bias(block_position)
        attn_input = block.block.attn_norm(routed_input)
        attention = block.block.attn
        batch, one, dim = attn_input.shape
        if one != 1:
            raise ValueError("BDRE route blocks accept exactly one token.")
        query, key, value = attention.qkv(attn_input).chunk(3, dim=-1)
        query = query.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
        key = key.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
        value = value.view(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
        canonical_key = projection.key_write(key.transpose(1, 2).reshape(batch, 1, dim)).squeeze(1)
        canonical_value = projection.value_write(value.transpose(1, 2).reshape(batch, 1, dim)).squeeze(1)

        self_key: torch.Tensor | None
        self_value: torch.Tensor | None
        if self.bdre_config.self_kv_mode == "bdre_prefix":
            prefix_key = (
                torch.cat([previous_key, canonical_key.unsqueeze(1)], dim=1)
                if previous_key is not None
                else canonical_key.unsqueeze(1)
            )
            prefix_value = (
                torch.cat([previous_value, canonical_value.unsqueeze(1)], dim=1)
                if previous_value is not None
                else canonical_value.unsqueeze(1)
            )
            current_block = torch.full(
                (batch, 1),
                reader_action,
                dtype=torch.long,
                device=hidden.device,
            )
            prefix_blocks = (
                torch.cat([previous_blocks, current_block], dim=1)
                if previous_blocks is not None
                else current_block
            )
            current_valid = torch.ones(batch, 1, dtype=torch.bool, device=hidden.device)
            prefix_valid = (
                torch.cat([previous_valid, current_valid], dim=1)
                if previous_valid is not None
                else current_valid
            )
            started = time.perf_counter()
            compiled = self.bdre_compiler.compile(
                prefix_key,
                prefix_value,
                prefix_blocks,
                prefix_valid,
                self._internal_block_positions(),
                reader_step=reader_step if self.bdre_config.depth_mode == "reader_step" else None,
                reader_actions=torch.full(
                    (batch,),
                    reader_action,
                    dtype=torch.long,
                    device=hidden.device,
                ),
            )
            elapsed = time.perf_counter() - started
            diagnostics.self_prefix_seconds += elapsed
            diagnostics.record_compile(compiled, elapsed)
            self_key = compiled.keys[:, 0]
            self_value = compiled.values[:, 0]
        elif self.bdre_config.self_kv_mode == "current_step":
            self_key = canonical_key
            self_value = canonical_value
        else:
            self_key = None
            self_value = None

        key_codes = historical_key
        value_codes = historical_value
        if self_key is not None and self_value is not None:
            key_codes = torch.cat([key_codes, self_key.unsqueeze(1)], dim=1)
            value_codes = torch.cat([value_codes, self_value.unsqueeze(1)], dim=1)

        if key_codes.size(1) == 0:
            attention_output = torch.zeros_like(routed_input)
        else:
            decoded_key = torch.einsum(
                "bnr,hrd->bhnd",
                key_codes.to(dtype=projection.key_read.dtype),
                projection.key_read,
            ).to(dtype=query.dtype)
            key_positions = torch.arange(key_codes.size(1), device=hidden.device)
            key_cosine = attention.rope.cos[:, :, key_positions, :].to(device=hidden.device, dtype=query.dtype)
            key_sine = attention.rope.sin[:, :, key_positions, :].to(device=hidden.device, dtype=query.dtype)
            decoded_key = apply_rotary(decoded_key, key_cosine, key_sine)
            query_cosine = attention.rope.cos[:, :, token_position : token_position + 1, :].to(
                device=hidden.device,
                dtype=query.dtype,
            )
            query_sine = attention.rope.sin[:, :, token_position : token_position + 1, :].to(
                device=hidden.device,
                dtype=query.dtype,
            )
            query = apply_rotary(query, query_cosine, query_sine).squeeze(2)
            scores = torch.einsum("bhd,bhnd->bhn", query, decoded_key) * (attention.head_dim**-0.5)
            weights = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
            weights = F.dropout(weights, p=attention.dropout, training=block.training)
            latent_value = torch.einsum(
                "bhn,bnr->bhr",
                weights,
                value_codes.to(dtype=weights.dtype),
            )
            attended = torch.einsum(
                "bhr,hrd->bhd",
                latent_value.to(dtype=projection.value_read.dtype),
                projection.value_read,
            ).to(dtype=hidden.dtype)
            attended = attended.reshape(batch, 1, dim)
            attention_output = attention.out(attended)

        routed = routed_input + attention_output
        routed = routed + block.block.ffn(block.block.ffn_norm(routed))
        return routed, canonical_key, canonical_value

    def _compile_completed_token(
        self,
        state: BDRECacheState,
        writer_keys: list[torch.Tensor],
        writer_values: list[torch.Tensor],
        writer_blocks: list[torch.Tensor],
        writer_valid: list[torch.Tensor],
        diagnostics: _BDREDiagnostics,
        *,
        capture_visualization: bool,
    ) -> BDRECacheState:
        key = torch.stack(writer_keys, dim=1)
        value = torch.stack(writer_values, dim=1)
        block = torch.stack(writer_blocks, dim=1)
        valid = torch.stack(writer_valid, dim=1)
        padding = self.config.max_route_steps - key.size(1)
        if padding < 0:
            raise ValueError("BDRE token route exceeded max_route_steps.")
        if padding:
            key = F.pad(key, (0, 0, 0, padding))
            value = F.pad(value, (0, 0, 0, padding))
            block = F.pad(block, (0, padding))
            valid = F.pad(valid, (0, padding), value=False)
        diagnostics.writer_counts.append(valid.sum(dim=-1))
        started = time.perf_counter()
        compiled = self.bdre_compiler.compile(
            key,
            value,
            block,
            valid,
            self._internal_block_positions(),
        )
        diagnostics.record_compile(compiled, time.perf_counter() - started)

        step_key = step_value = None
        if self.bdre_config.depth_mode == "reader_step" and self.bdre_config.reader_step_cache == "eager":
            started = time.perf_counter()
            step_key, step_value = self.bdre_compiler.compile_all_reader_steps(
                key,
                value,
                block,
                valid,
                self._internal_block_positions(),
            )
            diagnostics.compile_seconds += time.perf_counter() - started
        retain_writers = (
            self.bdre_config.depth_mode == "reader_step"
            and self.bdre_config.reader_step_cache in {"lazy", "dynamic"}
        )
        next_lazy_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.bdre_config.depth_mode == "reader_step" and self.bdre_config.reader_step_cache == "lazy":
            next_lazy_cache = {}
            for (reader_action, reader_step), (history_key, history_value) in state.lazy_cache.items():
                started = time.perf_counter()
                current = self.bdre_compiler.compile(
                    key,
                    value,
                    block,
                    valid,
                    self._internal_block_positions(),
                    reader_step=reader_step,
                    reader_actions=torch.full(
                        (key.size(0),),
                        reader_action,
                        dtype=torch.long,
                        device=key.device,
                    ),
                )
                diagnostics.compile_seconds += time.perf_counter() - started
                next_lazy_cache[(reader_action, reader_step)] = (
                    torch.cat([history_key, current.keys], dim=1),
                    torch.cat([history_value, current.values], dim=1),
                )
        if capture_visualization:
            visualization = {
                "writer_blocks": block.detach().cpu(),
                "writer_valid": valid.detach().cpu(),
                "key_weights": compiled.key_weights.detach().cpu(),
                "value_weights": compiled.value_weights.detach().cpu(),
                "block_positions": self._internal_block_positions().detach().cpu(),
            }
            if self.bdre_config.depth_mode == "reader_step":
                started = time.perf_counter()
                reader_step_outputs = [
                    self.bdre_compiler.compile(
                        key,
                        value,
                        block,
                        valid,
                        self._internal_block_positions(),
                        reader_step=reader_step,
                    )
                    for reader_step in range(self.config.max_route_steps)
                ]
                diagnostics.compile_seconds += time.perf_counter() - started
                visualization["reader_step_key_weights"] = torch.stack(
                    [output.key_weights for output in reader_step_outputs],
                    dim=1,
                ).detach().cpu()
                visualization["reader_step_value_weights"] = torch.stack(
                    [output.value_weights for output in reader_step_outputs],
                    dim=1,
                ).detach().cpu()
            diagnostics.last_visualization = visualization
        return state.append(
            compiled.keys,
            compiled.values,
            step_key=step_key,
            step_value=step_value,
            writer_key=key if retain_writers else None,
            writer_value=value if retain_writers else None,
            writer_block=block if retain_writers else None,
            writer_valid=valid if retain_writers else None,
            lazy_cache=next_lazy_cache,
        )

    def _build_output(
        self,
        logits: torch.Tensor,
        route_info: dict[str, Any],
        *,
        targets: torch.Tensor | None,
        lm_loss: torch.Tensor | None = None,
        include_auxiliary_losses: bool = True,
        loss_weights: Mapping[str, Any],
        routing_constraints: Mapping[str, Any],
        max_steps: int,
        summarize_routing: bool,
        log_path_counts: bool,
        bdre_metrics: Mapping[str, float],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"logits": logits, "route_info": route_info}
        if summarize_routing:
            summary = summarize_routes(
                route_info,
                self.config.route_pool_blocks,
                include_path_counts=log_path_counts,
            )
            summary.update(bdre_metrics)
            output["routing_summary"] = summary
        if targets is not None and lm_loss is not None:
            raise ValueError("Provide targets or lm_loss, not both.")
        if targets is None and lm_loss is None:
            return output

        lm = build_causal_lm_loss(logits, targets) if lm_loss is None else lm_loss
        if not include_auxiliary_losses:
            output["loss"] = lm
            output["loss_components"] = {"lm_loss": lm.detach()}
            return output
        route_weight = _loss_weight(loss_weights, "route")
        balance_weight = _loss_weight(loss_weights, "balance")
        cost_weight = _loss_weight(loss_weights, "cost")
        location_weight = _loss_weight(loss_weights, "location")
        selected_balance_weight = _loss_weight(loss_weights, "selected_balance")
        coverage_floor_weight = _loss_weight(loss_weights, "coverage_floor")
        transition_diversity_weight = _loss_weight(loss_weights, "transition_diversity")
        exit_boundary_weight = _loss_weight(loss_weights, "exit_boundary")
        input_anchor_weight = _loss_weight(loss_weights, "input_anchor")
        zero = _zero_loss_like(lm)
        route = (
            route_imitation_loss(route_info["route_logits"], route_info["route_targets"]).to(lm.device)
            if route_weight > 0.0
            else zero
        )
        balance = (
            block_balance_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            if balance_weight > 0.0
            else zero
        )
        cost = (
            route_cost_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            if cost_weight > 0.0
            else zero
        )
        location = (
            location_loss(route_info["location_distance"]).to(lm.device) if location_weight > 0.0 else zero
        )
        selected_balance = (
            selected_block_balance_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
            ).to(lm.device)
            if selected_balance_weight > 0.0
            else zero
        )
        coverage_floor = (
            block_coverage_floor_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
                floor=_coverage_floor_min(routing_constraints),
            ).to(lm.device)
            if coverage_floor_weight > 0.0
            else zero
        )
        transition_diversity = (
            transition_diversity_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
            ).to(lm.device)
            if transition_diversity_weight > 0.0
            else zero
        )
        exit_boundary = (
            exit_boundary_loss(
                route_info["route_probs"],
                self.config.route_pool_blocks,
                {**routing_constraints, "max_route_steps": max_steps},
            ).to(lm.device)
            if exit_boundary_weight > 0.0
            else zero
        )
        input_anchor = (
            self.position_table.input_anchor_loss().to(lm.device) if input_anchor_weight > 0.0 else zero
        )
        geometry = (
            self.position_table.geometry_loss().to(lm.device)
            if self.bdre_config.position_geometry_enabled and self.bdre_config.position_geometry_weight > 0.0
            else zero
        )
        total = (
            lm
            + route_weight * route
            + balance_weight * balance
            + cost_weight * cost
            + location_weight * location
            + selected_balance_weight * selected_balance
            + coverage_floor_weight * coverage_floor
            + transition_diversity_weight * transition_diversity
            + exit_boundary_weight * exit_boundary
            + input_anchor_weight * input_anchor
            + self.bdre_config.position_geometry_weight * geometry
        )
        output["loss"] = total
        output["loss_components"] = {
            "lm_loss": lm.detach(),
            "route_loss": route.detach(),
            "balance_loss": balance.detach(),
            "cost_loss": cost.detach(),
            "location_loss": location.detach(),
            "selected_balance_loss": selected_balance.detach(),
            "coverage_floor_loss": coverage_floor.detach(),
            "transition_diversity_loss": transition_diversity.detach(),
            "exit_boundary_loss": exit_boundary.detach(),
            "input_anchor_loss": input_anchor.detach(),
            "position_geometry_loss": geometry.detach(),
        }
        return output

    def model_stats(self) -> dict[str, int | str]:
        stats = super().model_stats()
        stats.update(
            {
                "model_name": self.config.model_name,
                "parameter_count": count_parameters(self),
                "bdre_shared_kv": "True",
                "bdre_key_dim": self.bdre_config.key_dim,
                "bdre_value_dim": self.bdre_config.value_dim,
                "bdre_depth_mode": self.bdre_config.depth_mode,
                "bdre_reader_step_cache": self.bdre_config.reader_step_cache,
                "bdre_self_kv_mode": self.bdre_config.self_kv_mode,
                "bdre_compile_top_k": str(self.bdre_config.compile_top_k),
                "bdre_key_temperature": str(self.bdre_config.key_temperature),
                "bdre_value_temperature": str(self.bdre_config.value_temperature),
                "bdre_execution_mode": self.bdre_config.execution_mode,
                "bdre_dispatch_mode": self.bdre_config.dispatch_mode,
                "bdre_chunk_size": self.bdre_config.chunk_size,
                "bdre_normalize_step_distance": str(self.bdre_config.normalize_step_distance),
                "bdre_synchronous_attention_backend": self.bdre_config.synchronous_attention_backend,
            }
        )
        return stats
