from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from brian_sphere_llm.memory import (
    AttentionGlobalKVState,
    CanonicalAttentionGlobalKVCache,
    CanonicalGlobalCache,
    GlobalReadAdapter,
    GlobalWriteAdapter,
)
from brian_sphere_llm.model.baseline import BaselineConfig, _float_value, _int_value
from brian_sphere_llm.model.exit_block import ExitBlock
from brian_sphere_llm.model.llama_backbone import (
    RMSNorm,
    TransformerBlock,
    apply_rotary,
    build_causal_lm_loss,
    checkpoint_if_enabled,
    count_parameters,
    require_torch,
)
from brian_sphere_llm.model.route_block import RouteBlock
from brian_sphere_llm.routing.block_position import BlockPositionTable
from brian_sphere_llm.routing.metrics import summarize_routes
from brian_sphere_llm.routing.parallel_passing import prune_branches
from brian_sphere_llm.routing.pseudo_policy import actions_for_policy
from brian_sphere_llm.routing.router import LatentRouter
from brian_sphere_llm.utils.config import load_config

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
class BrianRouteConfig:
    base: BaselineConfig
    pre_blocks: int
    route_pool_blocks: int
    post_blocks: int
    block_position_dim: int
    max_route_steps: int
    model_name: str = "brian_route_core"
    route_pool_finegrained: bool = False
    route_block_ffn_multiplier: float | None = None
    top_k: int = 1
    later_top_k: int = 2
    hard_exit: bool = False
    global_kv: bool = False
    global_code_dim: int = 64
    global_sink_slots: int = 4
    global_window_slots: int = 32
    global_adapter_scope: str = "shared"
    global_head_delta_rank: int = 0
    attention_global_kv: bool = False
    attention_global_kv_scope: str = "route"
    attention_global_kv_mode: str = "summary"
    attention_global_code_dim: int | None = None
    attention_global_sink_slots: int = 4
    attention_global_window_slots: int = 32
    attention_global_tokens_per_write: int = 1
    attention_global_logit_bias_init: float = -4.0
    attention_global_route_execution: str = "selected"
    parallel_passing: bool = False
    beam_size: int = 2
    branch_cost: float = 0.01
    branch_score_decay: float = 0.99
    parallel_exit_policy: str = "branch"
    block_position_mode: str = "open_arc"
    block_position_injection: str = "adapter"
    independent_input_position: bool = False
    position_to_router: bool = True
    position_to_blocks: bool = True
    location_bias_weight: float = 0.0
    route_block_execution: str = "full_sequence"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, config_dir: str | Path | None = None) -> "BrianRouteConfig":
        config_dir = Path(config_dir or ".")
        base_data: dict[str, Any]
        if "base" in data:
            base_data = dict(data["base"])
        else:
            base_config = data.get("base_config")
            if not base_config:
                raise ValueError("BRIAN config requires `base` or `base_config`")
            base_data = load_config(config_dir / str(base_config))
        return cls(
            base=BaselineConfig.from_dict(base_data),
            pre_blocks=_int_value(data["pre_blocks"], "pre_blocks", minimum=0),
            route_pool_blocks=_int_value(data["route_pool_blocks"], "route_pool_blocks", minimum=1),
            post_blocks=_int_value(data["post_blocks"], "post_blocks", minimum=0),
            block_position_dim=_int_value(data["block_position_dim"], "block_position_dim", minimum=1),
            max_route_steps=_int_value(data["max_route_steps"], "max_route_steps", minimum=1),
            model_name=str(data.get("model_name", "brian_route_core")),
            route_pool_finegrained=_bool_value(data.get("route_pool_finegrained", False), "route_pool_finegrained"),
            route_block_ffn_multiplier=_optional_float_value(
                data.get("route_block_ffn_multiplier"),
                "route_block_ffn_multiplier",
                minimum=0.01,
            ),
            top_k=_int_value(data.get("top_k", 1), "top_k", minimum=1),
            later_top_k=_int_value(data.get("later_top_k", 2), "later_top_k", minimum=1),
            hard_exit=_bool_value(data.get("hard_exit", False), "hard_exit"),
            global_kv=_bool_value(data.get("global_kv", False), "global_kv"),
            global_code_dim=_int_value(
                data.get("global_code_dim", data.get("block_position_dim", 64)),
                "global_code_dim",
                minimum=1,
            ),
            global_sink_slots=_int_value(data.get("global_sink_slots", 4), "global_sink_slots", minimum=0),
            global_window_slots=_int_value(data.get("global_window_slots", 32), "global_window_slots", minimum=0),
            global_adapter_scope=str(data.get("global_adapter_scope", "shared")),
            global_head_delta_rank=_int_value(
                data.get("global_head_delta_rank", 0),
                "global_head_delta_rank",
                minimum=0,
            ),
            attention_global_kv=_bool_value(data.get("attention_global_kv", False), "attention_global_kv"),
            attention_global_kv_scope=str(data.get("attention_global_kv_scope", "route")),
            attention_global_kv_mode=str(data.get("attention_global_kv_mode", "summary")),
            attention_global_code_dim=_optional_int_value(
                data.get("attention_global_code_dim"),
                "attention_global_code_dim",
                minimum=1,
            ),
            attention_global_sink_slots=_int_value(
                data.get("attention_global_sink_slots", 4),
                "attention_global_sink_slots",
                minimum=0,
            ),
            attention_global_window_slots=_int_value(
                data.get("attention_global_window_slots", 32),
                "attention_global_window_slots",
                minimum=0,
            ),
            attention_global_tokens_per_write=_int_value(
                data.get("attention_global_tokens_per_write", 1),
                "attention_global_tokens_per_write",
                minimum=1,
            ),
            attention_global_logit_bias_init=_float_value(
                data.get("attention_global_logit_bias_init", -4.0),
                "attention_global_logit_bias_init",
            ),
            attention_global_route_execution=str(data.get("attention_global_route_execution", "selected")),
            parallel_passing=_bool_value(data.get("parallel_passing", False), "parallel_passing"),
            beam_size=_int_value(data.get("beam_size", 2), "beam_size", minimum=1),
            branch_cost=_float_value(data.get("branch_cost", 0.01), "branch_cost", minimum=0.0),
            branch_score_decay=_float_value(
                data.get("branch_score_decay", 0.99),
                "branch_score_decay",
                minimum=0.0,
                maximum=1.0,
            ),
            parallel_exit_policy=str(data.get("parallel_exit_policy", "branch")),
            block_position_mode=str(data.get("block_position_mode", "open_arc")),
            block_position_injection=str(data.get("block_position_injection", "adapter")),
            independent_input_position=_bool_value(
                data.get("independent_input_position", False),
                "independent_input_position",
            ),
            position_to_router=_bool_value(data.get("position_to_router", True), "position_to_router"),
            position_to_blocks=_bool_value(data.get("position_to_blocks", True), "position_to_blocks"),
            location_bias_weight=_float_value(data.get("location_bias_weight", 0.0), "location_bias_weight", minimum=0.0),
            route_block_execution=_route_block_execution_value(data.get("route_block_execution", "full_sequence")),
        )


class BrianRouteCore(ModuleBase):
    def __init__(self, config: BrianRouteConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self._compiled_grouped_route_blocks_forward = None
        self._compiled_grouped_routed_blocks_forward = None
        if config.route_pool_finegrained:
            if config.pre_blocks + config.post_blocks > config.base.layers:
                raise ValueError("pre + post must be <= base layer count for fine-grained route pools")
        elif config.pre_blocks + config.route_pool_blocks + config.post_blocks != config.base.layers:
            raise ValueError("pre + route_pool + post must equal base layer count")
        if config.attention_global_kv:
            if config.attention_global_kv_scope != "route":
                raise ValueError("attention_global_kv_scope currently supports only 'route'")
            if config.attention_global_kv_mode not in {"summary", "token_compressed", "pure_factorized"}:
                raise ValueError(f"Unknown attention_global_kv_mode: {config.attention_global_kv_mode}")
            if config.attention_global_tokens_per_write != 1:
                raise ValueError("attention_global_tokens_per_write currently supports only 1")
            if config.attention_global_route_execution not in {
                "selected",
                "top1_fast",
                "grouped_selected",
                "cache_only",
            }:
                raise ValueError(
                    "attention_global_route_execution must be 'selected', 'top1_fast', "
                    "'grouped_selected', or 'cache_only'."
                )
            if (
                config.attention_global_route_execution != "selected"
                and config.attention_global_kv_mode != "pure_factorized"
            ):
                raise ValueError("attention_global_route_execution fast paths currently require pure_factorized mode.")
            if config.attention_global_route_execution == "cache_only" and (
                config.top_k != 1 or config.later_top_k != 1
            ):
                raise ValueError("cache_only attention global execution currently requires top_k=later_top_k=1.")
            if (
                config.attention_global_kv_mode != "pure_factorized"
                and config.attention_global_sink_slots + config.attention_global_window_slots <= 0
            ):
                raise ValueError("Attention Global KV requires at least one retained slot")
            if config.parallel_passing:
                raise ValueError("Attention Global KV is route-only and does not support parallel_passing yet")
        if config.route_block_execution not in {"full_sequence", "sparse", "sparse_varlen", "grouped_dense"}:
            raise ValueError(
                "route_block_execution must be 'full_sequence', 'sparse', 'sparse_varlen', or 'grouped_dense'."
            )
        backbone = config.base.backbone()
        route_backbone = backbone
        if config.route_block_ffn_multiplier is not None:
            route_backbone = replace(backbone, ffn_multiplier=config.route_block_ffn_multiplier)
        if config.attention_global_kv:
            route_backbone = replace(
                route_backbone,
                attention_global_logit_bias_init=config.attention_global_logit_bias_init,
                attention_global_sink_slots=config.attention_global_sink_slots,
                attention_global_kv_mode=config.attention_global_kv_mode,
                attention_global_code_dim=config.attention_global_code_dim,
            )
        self.token_embedding = nn.Embedding(config.base.vocab_size, config.base.d_model)
        self.pre_blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.pre_blocks)])
        self.route_blocks = nn.ModuleList(
            [
                RouteBlock(route_backbone, config.block_position_dim, config.block_position_injection)
                for _ in range(config.route_pool_blocks)
            ]
        )
        if config.attention_global_kv and config.attention_global_kv_mode == "pure_factorized":
            self._tie_pure_factorized_global_writers()
        self.exit_block = ExitBlock(config.base.d_model, config.block_position_dim, config.block_position_injection)
        self.post_blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.post_blocks)])
        self.position_table = BlockPositionTable(
            config.route_pool_blocks,
            config.block_position_dim,
            mode=config.block_position_mode,
            independent_input_position=config.independent_input_position,
        )
        self.router = LatentRouter(
            config.base.d_model,
            config.block_position_dim,
            num_actions=config.route_pool_blocks + 1,
        )
        if config.global_kv:
            if config.global_adapter_scope not in {"shared", "per_block"}:
                raise ValueError(f"Unknown global_adapter_scope: {config.global_adapter_scope}")
            self.global_cache = CanonicalGlobalCache(config.global_sink_slots, config.global_window_slots)
            adapter_kwargs = {
                "n_heads": config.base.n_heads,
                "head_delta_rank": config.global_head_delta_rank,
            }
            if config.global_adapter_scope == "per_block":
                adapter_count = config.route_pool_blocks + 1
                self.global_write = nn.ModuleList(
                    [
                        GlobalWriteAdapter(config.base.d_model, config.global_code_dim, **adapter_kwargs)
                        for _ in range(adapter_count)
                    ]
                )
                self.global_read = nn.ModuleList(
                    [
                        GlobalReadAdapter(config.base.d_model, config.global_code_dim, **adapter_kwargs)
                        for _ in range(adapter_count)
                    ]
                )
            else:
                self.global_write = GlobalWriteAdapter(config.base.d_model, config.global_code_dim, **adapter_kwargs)
                self.global_read = GlobalReadAdapter(config.base.d_model, config.global_code_dim, **adapter_kwargs)
        else:
            self.global_cache = None
            self.global_write = None
            self.global_read = None
        if config.attention_global_kv:
            self.attention_global_cache = CanonicalAttentionGlobalKVCache(
                config.attention_global_sink_slots,
                config.attention_global_window_slots,
                latest_token_only=config.attention_global_kv_mode == "pure_factorized",
            )
        else:
            self.attention_global_cache = None
        self.norm = RMSNorm(config.base.d_model)
        self.lm_head = nn.Linear(config.base.d_model, config.base.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    @property
    def out_action(self) -> int:
        return self.config.route_pool_blocks

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
        summarize_routing: bool = True,
    ) -> dict[str, Any]:
        loss_weights = _loss_weights_mapping(loss_weights)
        routing_constraints = _routing_constraints_mapping(routing_constraints)
        routing_options = _routing_options_mapping(routing_options)
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        batch_size = input_ids.size(0)
        hidden = self.token_embedding(input_ids)
        for block in self.pre_blocks:
            hidden = checkpoint_if_enabled(self, block, hidden)

        parallel_active = route_mode == "parallel" or self.config.parallel_passing
        base_position = self.position_table.initial(batch_size, input_ids.device)
        position = (
            base_position
            if parallel_active
            else base_position.unsqueeze(1).expand(-1, input_ids.size(1), -1).contiguous()
        )
        route_info: dict[str, Any] = {
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
            "global_attention_mass": [],
            "global_sink_attention_mass": [],
            "global_window_attention_mass": [],
            "global_read_gate": [],
            "global_cache_slots": [],
            "attention_global_kv_slots": [],
            "attention_global_kv_write_count": [],
            "attention_global_kv_logit_bias": [],
            "attention_global_kv_last_token_mass": [],
            "attention_global_kv_sink_last_token_mass": [],
            "attention_global_kv_window_last_token_mass": [],
            "parallel_delta_cache_slots": [],
            "random_route_probability": torch.tensor(
                self._random_route_probability(global_step, routing_options),
                dtype=hidden.dtype,
                device=input_ids.device,
            ),
            "random_route_override_count": [],
            "self_recur_cap_count": [],
            "hard_exit_enabled": bool(hard_exit),
            "max_route_steps": self.config.max_route_steps,
            "route_logit_noise_std": torch.tensor(
                self._route_logit_noise_std(global_step, routing_options),
                dtype=hidden.dtype,
                device=input_ids.device,
            ),
        }
        route_targets = self._targets_for_mode(route_mode, pseudo_policy, input_ids)
        max_steps = len(route_targets) if route_mode in {"fixed", "pseudo"} else self.config.max_route_steps
        route_shape = (batch_size,) if parallel_active else (batch_size, input_ids.size(1))
        exited = torch.zeros(route_shape, dtype=torch.bool, device=input_ids.device)
        global_state = None
        if self.config.global_kv:
            assert self.global_cache is not None and self.global_write is not None
            global_state = self.global_cache.empty(
                batch_size=batch_size,
                code_dim=self.config.global_code_dim,
                device=input_ids.device,
                dtype=hidden.dtype,
            )
            global_state = self.global_cache.write(global_state, self._global_write(hidden, tokenwise=not parallel_active))
        last_global_actions = None
        attention_global_state = None
        if self.config.attention_global_kv:
            assert self.attention_global_cache is not None
            attention_global_state = self.attention_global_cache.empty(
                batch_size=batch_size,
                n_heads=self._attention_global_cache_heads(),
                head_dim=self._attention_global_cache_dim(),
                device=input_ids.device,
                dtype=hidden.dtype,
                sequence_length=None if parallel_active else hidden.size(1),
            )
            if self._cache_only_attention_global_route_execution_enabled() and not parallel_active:
                init_key, init_value, init_valid = self._pure_factorized_shared_global_write(hidden)
                attention_global_state = self.attention_global_cache.write(
                    attention_global_state,
                    init_key,
                    init_value,
                    init_valid,
                )
        router_space_records: list[dict[str, Any]] | None = [] if collect_router_space else None
        record_route_diagnostics = summarize_routing or collect_router_space
        last_internal_action = torch.full(route_shape, -1, dtype=torch.long, device=input_ids.device)
        same_internal_run_length = torch.zeros(route_shape, dtype=torch.long, device=input_ids.device)
        grouped_route_weights = (
            self._grouped_route_block_weights() if self._grouped_dense_route_block_execution_enabled() else None
        )
        grouped_attention_global_weights = (
            self._grouped_pure_factorized_attention_global_weights()
            if self._grouped_attention_global_route_execution_enabled()
            else None
        )

        if parallel_active:
            hidden, position, route_info = self._run_parallel_route(hidden, position, route_info, hard_exit, global_state)
            global_state = None

        for step in range(max_steps if not parallel_active else 0):
            if self.config.global_kv and global_state is not None:
                assert self.global_read is not None
                hidden, global_metrics = self._global_read(
                    hidden,
                    global_state.codes,
                    sink_slots=self.config.global_sink_slots,
                    actions=last_global_actions,
                )
                route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
                route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
                route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
                route_info["global_read_gate"].append(global_metrics["global_read_gate"])
                route_info["global_cache_slots"].append(
                    torch.tensor(float(global_state.slots), device=input_ids.device, dtype=hidden.dtype)
                )
            router_position = self._router_position(position)
            token_route = position.dim() == 3
            router_embedding = (
                self.router.token_embedding(hidden, router_position)
                if router_space_records is not None and token_route
                else self.router.embedding(hidden, router_position)
                if router_space_records is not None
                else None
            )
            raw_logits = (
                self.router.logits_from_embedding(router_embedding)
                if router_embedding is not None
                else self.router.token_logits(hidden, router_position)
                if token_route
                else self.router(hidden, router_position)
            )
            logits = self._apply_location_bias(raw_logits, position)
            logits = self._apply_route_logit_noise(logits, global_step, routing_options)
            logits = self._apply_route_constraints(logits, step, max_steps, routing_constraints)
            logits, self_recur_cap_mask = self._apply_self_recur_cap(
                logits,
                last_internal_action,
                same_internal_run_length,
                routing_constraints,
            )
            probs = F.softmax(logits, dim=-1)
            effective_top_k = self._top_k_for_step(step)
            top_actions, top_weights = self._topk_actions(probs, effective_top_k)
            has_route_target = route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets)
            if has_route_target:
                target_action = route_targets[step]
                if token_route:
                    target_action = target_action.unsqueeze(1).expand(-1, hidden.size(1))
            else:
                target_action = torch.full(route_shape, self.out_action, dtype=torch.long, device=input_ids.device)

            use_weighted_fusion = torch.zeros(route_shape, dtype=torch.bool, device=input_ids.device)
            weighted_fusion_possible = False
            if route_mode in {"fixed", "pseudo"}:
                selected = target_action
            elif route_mode == "scheduled":
                selected, use_router = self._scheduled_select(
                    logits,
                    target_action,
                    global_step,
                    router_probability,
                    routing_options,
                )
                use_weighted_fusion = use_router & (effective_top_k > 1)
                weighted_fusion_possible = bool(
                    effective_top_k > 1 and (router_probability is None or router_probability > 0.0)
                )
            elif route_mode == "free":
                selected = self._router_action(logits, routing_options)
                use_weighted_fusion = torch.full(
                    route_shape,
                    effective_top_k > 1,
                    dtype=torch.bool,
                    device=input_ids.device,
                )
                weighted_fusion_possible = effective_top_k > 1
            else:
                raise ValueError(f"Unknown route_mode: {route_mode}")

            if route_mode in {"free", "scheduled"}:
                selected, random_route_mask = self._apply_random_route_override(
                    selected,
                    global_step,
                    routing_options,
                    last_internal_action,
                    same_internal_run_length,
                    routing_constraints,
                )
            else:
                random_route_mask = torch.zeros_like(selected, dtype=torch.bool)
            selected, selected_cap_mask = self._enforce_self_recur_cap_on_selected(
                selected,
                logits,
                last_internal_action,
                same_internal_run_length,
                routing_constraints,
            )
            if self._force_final_exit(step, max_steps, routing_constraints):
                selected = torch.full_like(selected, self.out_action)
            selected = torch.where(exited, torch.full_like(selected, self.out_action), selected)
            use_weighted_fusion = (
                use_weighted_fusion
                & ~random_route_mask
                & ~selected_cap_mask
                & ~exited
                & (selected != self.out_action)
            )
            exit_now = selected == self.out_action
            record_embedding = self._last_token_view(router_embedding) if router_embedding is not None else None
            record_logits = self._last_token_view(logits)
            record_probs = self._last_token_view(probs)
            record_selected = self._last_token_view(selected)
            record_target_action = self._last_token_view(target_action)
            if router_space_records is not None and router_embedding is not None:
                record_raw_logits = self._last_token_view(raw_logits)
                record_top_actions = self._last_token_view(top_actions)
                record_top_weights = self._last_token_view(top_weights)
                record_random_route_mask = self._last_token_view(random_route_mask)
                record_self_recur_cap_mask = self._last_token_view(self_recur_cap_mask)
                record_exited = self._last_token_view(exited)
                record_exit_now = self._last_token_view(exit_now)
                router_space_records.append(
                    {
                        "step": int(step),
                        "embedding": record_embedding.detach(),
                        "raw_logits": record_raw_logits.detach(),
                        "effective_logits": record_logits.detach(),
                        "probs": record_probs.detach(),
                        "selected_actions": record_selected.detach(),
                        "top_actions": record_top_actions.detach(),
                        "top_weights": record_top_weights.detach(),
                        "random_route_override": record_random_route_mask.detach(),
                        "self_recur_cap_active": record_self_recur_cap_mask.detach(),
                        "exited_before": record_exited.detach(),
                        "exit_now": record_exit_now.detach(),
                    }
                )
            if self.config.attention_global_kv and attention_global_state is not None:
                hidden, write_key, write_value, write_valid = self._apply_routed_blocks_with_attention_global(
                    hidden,
                    position,
                    selected,
                    top_actions,
                    top_weights,
                    use_weighted_fusion,
                    attention_global_state,
                    route_info,
                    grouped_attention_global_weights=grouped_attention_global_weights,
                    weighted_fusion_possible=weighted_fusion_possible,
                )
            else:
                hidden = self._apply_routed_blocks(
                    hidden,
                    position,
                    selected,
                    top_actions,
                    top_weights,
                    use_weighted_fusion,
                    grouped_route_weights=grouped_route_weights,
                    weighted_fusion_possible=weighted_fusion_possible,
                )
                write_key = write_value = write_valid = None
            if hard_exit:
                exited = exited | exit_now
            position = self._next_position(selected, top_actions, top_weights, use_weighted_fusion)
            if self.config.global_kv and global_state is not None:
                assert self.global_write is not None and self.global_cache is not None
                global_state = self.global_cache.write(
                    global_state,
                    self._global_write(hidden, selected, tokenwise=position.dim() == 3),
                )
                last_global_actions = selected
            if self.config.attention_global_kv and attention_global_state is not None:
                assert self.attention_global_cache is not None
                assert write_key is not None and write_value is not None and write_valid is not None
                write_allowed = (~exited & (selected != self.out_action)).unsqueeze(1)
                write_valid = write_valid & write_allowed
                if torch.any(write_valid):
                    attention_global_state = self.attention_global_cache.write(
                        attention_global_state,
                        write_key,
                        write_value,
                        write_valid,
                    )
                route_info["attention_global_kv_slots"].append(
                    torch.tensor(float(attention_global_state.slots), device=input_ids.device, dtype=hidden.dtype)
                )
                route_info["attention_global_kv_write_count"].append(write_valid.to(hidden.dtype).sum())

            route_info["route_logits"].append(record_logits)
            route_info["route_probs"].append(record_probs)
            route_info["selected_actions"].append(record_selected)
            if has_route_target:
                route_info["route_targets"].append(record_target_action)
            record_position = self._last_token_view(position)
            route_info["location_distance"].append(self.position_table.location_distance(record_position, record_probs))
            if record_route_diagnostics:
                record_top_actions = self._last_token_view(top_actions)
                record_top_weights = self._last_token_view(top_weights)
                record_weighted_fusion = self._last_token_view(use_weighted_fusion)
                record_random_route_mask = self._last_token_view(random_route_mask)
                record_self_recur_cap_mask = self._last_token_view(self_recur_cap_mask)
                record_selected_cap_mask = self._last_token_view(selected_cap_mask)
                record_exit_now = self._last_token_view(exit_now)
                route_info["topk_actions"].append(record_top_actions)
                route_info["topk_weights"].append(record_top_weights)
                route_info["used_weighted_fusion"].append(record_weighted_fusion)
                route_info["exit_flags"].append(record_exit_now)
                route_info["random_route_override_count"].append(record_random_route_mask.to(hidden.dtype).sum())
                route_info["self_recur_cap_count"].append(
                    (record_self_recur_cap_mask | record_selected_cap_mask).to(hidden.dtype).sum()
                )
                route_info["position_norms"].append(record_position.norm(dim=-1).mean())
            last_internal_action, same_internal_run_length = self._update_self_recur_state(
                selected,
                last_internal_action,
                same_internal_run_length,
            )
            if hard_exit and torch.all(exited):
                break

        hidden = checkpoint_if_enabled(self, self.exit_block, hidden, self._block_position(position))
        if self.config.global_kv and global_state is not None:
            assert self.global_read is not None
            hidden, global_metrics = self._global_read(
                hidden,
                global_state.codes,
                sink_slots=self.config.global_sink_slots,
                actions=last_global_actions,
            )
            route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
            route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
            route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
            route_info["global_read_gate"].append(global_metrics["global_read_gate"])
            route_info["global_cache_slots"].append(
                torch.tensor(float(global_state.slots), device=input_ids.device, dtype=hidden.dtype)
            )
        for block in self.post_blocks:
            hidden = checkpoint_if_enabled(self, block, hidden)
        logits = self.lm_head(self.norm(hidden))

        output: dict[str, Any] = {
            "logits": logits,
            "route_info": route_info,
        }
        if summarize_routing:
            output["routing_summary"] = summarize_routes(
                route_info,
                self.config.route_pool_blocks,
                include_path_counts=log_path_counts,
            )
        if router_space_records is not None:
            output["router_space"] = {
                "records": router_space_records,
                "num_actions": self.config.route_pool_blocks + 1,
                "out_action": self.out_action,
            }
        if targets is not None:
            lm = build_causal_lm_loss(logits, targets)
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
            loc = location_loss(route_info["location_distance"]).to(lm.device) if location_weight > 0.0 else zero
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
            input_anchor = self.position_table.input_anchor_loss().to(lm.device) if input_anchor_weight > 0.0 else zero
            total = (
                lm
                + route_weight * route
                + balance_weight * balance
                + cost_weight * cost
                + location_weight * loc
                + selected_balance_weight * selected_balance
                + coverage_floor_weight * coverage_floor
                + transition_diversity_weight * transition_diversity
                + exit_boundary_weight * exit_boundary
                + input_anchor_weight * input_anchor
            )
            output["loss"] = total
            output["loss_components"] = {
                "lm_loss": lm.detach(),
                "route_loss": route.detach(),
                "balance_loss": balance.detach(),
                "cost_loss": cost.detach(),
                "location_loss": loc.detach(),
                "selected_balance_loss": selected_balance.detach(),
                "coverage_floor_loss": coverage_floor.detach(),
                "transition_diversity_loss": transition_diversity.detach(),
                "exit_boundary_loss": exit_boundary.detach(),
                "input_anchor_loss": input_anchor.detach(),
            }
        return output

    def _targets_for_mode(self, route_mode: str, pseudo_policy: str, input_ids: torch.Tensor) -> list[torch.Tensor]:
        if route_mode in {"fixed", "pseudo", "scheduled"}:
            difficulty = _content_difficulty_ids(input_ids) if pseudo_policy == "mixed_skip_recur" else None
            return actions_for_policy(
                pseudo_policy,
                num_internal_blocks=self.config.route_pool_blocks,
                max_route_steps=self.config.max_route_steps,
                batch_size=input_ids.size(0),
                device=input_ids.device,
                difficulty=difficulty,
            )
        return []

    def _run_parallel_route(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        route_info: dict[str, Any],
        hard_exit: bool,
        global_state: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        batch_size = hidden.size(0)
        branch_hidden = hidden.unsqueeze(1)
        branch_position = position.unsqueeze(1)
        branch_scores = torch.zeros(batch_size, 1, device=hidden.device, dtype=hidden.dtype)
        branch_exited = torch.zeros(batch_size, 1, device=hidden.device, dtype=torch.bool)
        branch_delta_codes = None
        branch_last_actions = None
        if self.config.global_kv:
            branch_delta_codes = torch.empty(
                batch_size,
                1,
                0,
                self.config.global_code_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            branch_last_actions = torch.full(
                (batch_size, 1),
                self.out_action,
                device=hidden.device,
                dtype=torch.long,
            )
        beam_size = max(1, self.config.beam_size)

        for step in range(self.config.max_route_steps):
            top_k = max(1, min(self._top_k_for_step(step), self.config.route_pool_blocks + 1))
            current_beam = branch_hidden.size(1)
            flat_hidden = branch_hidden.reshape(batch_size * current_beam, *branch_hidden.shape[2:])
            flat_position = branch_position.reshape(batch_size * current_beam, branch_position.size(-1))
            if self.config.global_kv and global_state is not None:
                assert self.global_read is not None
                assert branch_delta_codes is not None
                assert branch_last_actions is not None
                base_codes = global_state.codes.unsqueeze(1).expand(-1, current_beam, -1, -1)
                combined_codes = torch.cat([base_codes, branch_delta_codes], dim=2)
                flat_codes = combined_codes.reshape(batch_size * current_beam, combined_codes.size(2), combined_codes.size(3))
                flat_hidden, global_metrics = self._global_read(
                    flat_hidden,
                    flat_codes,
                    sink_slots=self.config.global_sink_slots,
                    actions=branch_last_actions.reshape(-1),
                )
                route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
                route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
                route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
                route_info["global_read_gate"].append(global_metrics["global_read_gate"])
                route_info["global_cache_slots"].append(
                    torch.tensor(float(flat_codes.size(1)), device=hidden.device, dtype=hidden.dtype)
                )
            logits = self._apply_location_bias(self.router(flat_hidden, self._router_position(flat_position)), flat_position)
            probs = F.softmax(logits, dim=-1).view(batch_size, current_beam, -1)
            log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_actions = log_probs.topk(top_k, dim=-1)
            top_actions = top_actions.view(batch_size, current_beam, top_k)
            top_log_probs = top_log_probs.view(batch_size, current_beam, top_k)

            expanded_actions = top_actions
            expanded_log_probs = top_log_probs
            parent_exit = self._parallel_parent_exit(top_actions)
            if torch.any(parent_exit):
                out = torch.full_like(expanded_actions, self.out_action)
                out_log_probs = log_probs.view(batch_size, current_beam, -1)[..., self.out_action]
                expanded_actions = torch.where(parent_exit.unsqueeze(-1), out, expanded_actions)
                expanded_log_probs = torch.where(
                    parent_exit.unsqueeze(-1),
                    out_log_probs.unsqueeze(-1).expand_as(expanded_log_probs),
                    expanded_log_probs,
                )
            if torch.any(branch_exited):
                out = torch.full_like(expanded_actions, self.out_action)
                zeros = torch.zeros_like(expanded_log_probs)
                expanded_actions = torch.where(branch_exited.unsqueeze(-1), out, expanded_actions)
                expanded_log_probs = torch.where(branch_exited.unsqueeze(-1), zeros, expanded_log_probs)

            internal = expanded_actions != self.out_action
            candidate_scores = (
                branch_scores.unsqueeze(-1) * self.config.branch_score_decay
                + expanded_log_probs
                - self.config.branch_cost * internal.float()
            )
            candidate_actions = expanded_actions.reshape(batch_size, current_beam * top_k)
            candidate_scores_flat = candidate_scores.reshape(batch_size, current_beam * top_k)
            pruned = prune_branches(candidate_actions, candidate_scores_flat, beam_size)
            selected_indices = candidate_scores_flat.topk(min(beam_size, candidate_scores_flat.size(-1)), dim=-1).indices
            parent_indices = torch.div(selected_indices, top_k, rounding_mode="floor")
            selected_actions = pruned.actions
            selected_scores = pruned.scores

            gather_hidden_index = parent_indices.view(batch_size, -1, 1, 1).expand(-1, -1, branch_hidden.size(2), branch_hidden.size(3))
            gather_position_index = parent_indices.view(batch_size, -1, 1).expand(-1, -1, branch_position.size(2))
            next_hidden = branch_hidden.gather(1, gather_hidden_index)
            next_position = branch_position.gather(1, gather_position_index)
            next_delta_codes = None
            if branch_delta_codes is not None:
                if branch_delta_codes.size(2) == 0:
                    next_delta_codes = branch_delta_codes.expand(batch_size, selected_actions.size(1), 0, self.config.global_code_dim)
                else:
                    gather_delta_index = parent_indices.view(batch_size, -1, 1, 1).expand(
                        -1,
                        -1,
                        branch_delta_codes.size(2),
                        branch_delta_codes.size(3),
                    )
                    next_delta_codes = branch_delta_codes.gather(1, gather_delta_index)
            parent_exited = branch_exited.gather(1, parent_indices)
            flat_next_hidden = next_hidden.reshape(batch_size * selected_actions.size(1), *next_hidden.shape[2:])
            flat_next_position = next_position.reshape(batch_size * selected_actions.size(1), next_position.size(-1))
            flat_actions = selected_actions.reshape(-1)
            flat_parent_exited = parent_exited.reshape(-1)
            apply_actions = torch.where(flat_parent_exited, torch.full_like(flat_actions, self.out_action), flat_actions)
            routed = self._apply_selected_blocks(flat_next_hidden, flat_next_position, apply_actions)
            routed_position = self.position_table.by_action(apply_actions)
            flat_exit = flat_parent_exited | (apply_actions == self.out_action)

            branch_count = selected_actions.size(1)
            branch_hidden = routed.reshape(batch_size, branch_count, *hidden.shape[1:])
            branch_position = routed_position.reshape(batch_size, branch_count, position.size(-1))
            branch_scores = selected_scores
            branch_exited = flat_exit.reshape(batch_size, branch_count)
            if self.config.global_kv and next_delta_codes is not None:
                assert self.global_write is not None
                delta_write = self._global_write(routed, apply_actions, tokenwise=False).reshape(
                    batch_size,
                    branch_count,
                    self.config.global_code_dim,
                )
                branch_delta_codes = torch.cat([next_delta_codes, delta_write.unsqueeze(2)], dim=2)
                if self.config.global_window_slots:
                    branch_delta_codes = branch_delta_codes[:, :, -self.config.global_window_slots :, :]
                route_info.setdefault("parallel_delta_cache_slots", []).append(
                    torch.tensor(float(branch_delta_codes.size(2)), device=hidden.device, dtype=hidden.dtype)
                )
            if branch_last_actions is not None:
                branch_last_actions = apply_actions.reshape(batch_size, branch_count)

            branch_weights = F.softmax(branch_scores, dim=-1)
            route_info["route_logits"].append((probs * branch_weights.unsqueeze(-1)).sum(dim=1).log().clamp_min(-50.0))
            route_info["route_probs"].append((probs * branch_weights.unsqueeze(-1)).sum(dim=1))
            route_info["selected_actions"].append(selected_actions[:, 0])
            route_info["topk_actions"].append(selected_actions)
            route_info["topk_weights"].append(branch_weights)
            route_info["used_weighted_fusion"].append(torch.zeros(batch_size, dtype=torch.bool, device=hidden.device))
            route_info["exit_flags"].append(branch_exited[:, 0])
            route_info["position_norms"].append(branch_position[:, 0, :].norm(dim=-1).mean())
            route_info["location_distance"].append(self.position_table.location_distance(branch_position[:, 0, :], route_info["route_probs"][-1]))
            route_info.setdefault("parallel_branch_count", []).append(
                torch.tensor(float(branch_count), device=hidden.device, dtype=hidden.dtype)
            )
            if branch_count > 1:
                margin = branch_scores[:, 0] - branch_scores[:, 1]
            else:
                margin = torch.zeros(batch_size, device=hidden.device, dtype=hidden.dtype)
            route_info.setdefault("parallel_score_margin", []).append(margin.mean())
            if hard_exit and torch.all(branch_exited):
                break

        weights = F.softmax(branch_scores, dim=-1)
        merged_hidden = (branch_hidden * weights.view(batch_size, -1, 1, 1)).sum(dim=1)
        merged_position = F.normalize((branch_position * weights.view(batch_size, -1, 1)).sum(dim=1), dim=-1)
        return merged_hidden, merged_position, route_info

    def _parallel_parent_exit(self, top_actions: torch.Tensor) -> torch.Tensor:
        policy = self.config.parallel_exit_policy
        if policy == "branch":
            return torch.zeros(top_actions.shape[:-1], dtype=torch.bool, device=top_actions.device)
        if policy == "top1":
            return top_actions[..., 0] == self.out_action
        if policy == "any_topk":
            return top_actions[..., 0] == self.out_action
        raise ValueError(f"Unknown parallel_exit_policy: {policy}")

    def _scheduled_select(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        global_step: int,
        router_probability: float | None,
        routing_options: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_choice = self._router_action(logits, routing_options)
        probability = router_probability
        if probability is None:
            probability = min(1.0, max(0.0, global_step / max(1, self.config.max_route_steps * 100)))
        use_router = torch.rand_like(target.float()) < probability
        return torch.where(use_router, router_choice, target), use_router

    def _router_action(self, logits: torch.Tensor, routing_options: Mapping[str, Any]) -> torch.Tensor:
        selection = str(routing_options.get("router_selection", "argmax"))
        if selection == "argmax":
            return torch.argmax(logits, dim=-1)
        if selection != "sample":
            raise ValueError("routing.router_selection must be 'argmax' or 'sample'.")
        sample_eval = _routing_bool(routing_options, "router_sampling_eval", default=False)
        if not self.training and not sample_eval:
            return torch.argmax(logits, dim=-1)
        temperature = _routing_float(routing_options, "router_sampling_temperature", default=1.0, minimum=1e-6)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        flat = probs.reshape(-1, probs.size(-1))
        return torch.multinomial(flat, num_samples=1).view(*probs.shape[:-1])

    def _random_route_probability(self, global_step: int, options: Mapping[str, Any]) -> float:
        if not self.training:
            return 0.0
        start = _routing_float(options, "random_route_probability", default=0.0, minimum=0.0)
        if start <= 0.0:
            return 0.0
        if start > 1.0:
            raise ValueError("routing.random_route_probability must be <= 1.0.")
        minimum = _routing_float(options, "random_route_min_probability", default=0.0, minimum=0.0)
        if minimum > 1.0:
            raise ValueError("routing.random_route_min_probability must be <= 1.0.")
        if minimum > start:
            raise ValueError("routing.random_route_min_probability must be <= routing.random_route_probability.")
        decay_steps = _routing_int(options, "random_route_decay_steps", default=0, minimum=0)
        if decay_steps <= 0:
            return start
        progress = min(1.0, max(0.0, float(global_step) / float(decay_steps)))
        return start + (minimum - start) * progress

    def _apply_random_route_override(
        self,
        selected: torch.Tensor,
        global_step: int,
        routing_options: Mapping[str, Any],
        last_internal_action: torch.Tensor,
        same_internal_run_length: torch.Tensor,
        routing_constraints: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probability = self._random_route_probability(global_step, routing_options)
        if probability <= 0.0:
            return selected, torch.zeros_like(selected, dtype=torch.bool)
        eligible = selected != self.out_action
        random_mask = (torch.rand_like(selected.float()) < probability) & eligible
        if not torch.any(random_mask):
            return selected, random_mask
        random_actions = torch.randint(
            0,
            self.config.route_pool_blocks,
            selected.shape,
            dtype=torch.long,
            device=selected.device,
        )
        random_actions = self._avoid_capped_self_recur_actions(
            random_actions,
            last_internal_action,
            same_internal_run_length,
            routing_constraints,
        )
        return torch.where(random_mask, random_actions, selected), random_mask

    def _apply_selected_blocks(self, hidden: torch.Tensor, position: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if selected.dim() == 2 and self._grouped_dense_route_block_execution_enabled():
            return self._apply_selected_blocks_grouped_dense(hidden, position, selected)
        if selected.dim() == 2 and self._sparse_route_block_execution_enabled():
            return self._apply_selected_blocks_sparse(hidden, position, selected)
        if selected.dim() == 2:
            next_hidden = hidden.clone()
            block_position = self._block_position(position)
            for action, block in enumerate(self.route_blocks):
                mask = selected == action
                if torch.any(mask):
                    action_output = checkpoint_if_enabled(self, block, hidden, block_position)
                    next_hidden = torch.where(mask.unsqueeze(-1), action_output, next_hidden)
            return next_hidden
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            mask = selected == action
            if torch.any(mask):
                next_hidden[mask] = checkpoint_if_enabled(self, block, hidden[mask], block_position[mask])
        return next_hidden

    def _sparse_route_block_execution_enabled(self) -> bool:
        return (
            self.config.route_block_execution in {"sparse", "sparse_varlen"}
            and not self.config.global_kv
            and not self.config.attention_global_kv
            and not self.config.parallel_passing
        )

    def _grouped_dense_route_block_execution_enabled(self) -> bool:
        return (
            self.config.route_block_execution == "grouped_dense"
            and not self.config.global_kv
            and not self.config.attention_global_kv
            and not self.config.parallel_passing
            and not self.activation_checkpointing
        )

    def _grouped_attention_global_route_execution_enabled(self) -> bool:
        return (
            self.config.attention_global_route_execution in {"grouped_selected", "cache_only"}
            and self.config.attention_global_kv
            and self.config.attention_global_kv_mode == "pure_factorized"
            and not self.config.parallel_passing
            and not self.activation_checkpointing
        )

    def _cache_only_attention_global_route_execution_enabled(self) -> bool:
        return (
            self.config.attention_global_route_execution == "cache_only"
            and self.config.attention_global_kv
            and self.config.attention_global_kv_mode == "pure_factorized"
            and not self.config.parallel_passing
        )

    def _forward_sparse_route_block(
        self,
        block: RouteBlock,
        hidden: torch.Tensor,
        block_position: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.route_block_execution == "sparse_varlen":
            backend = _sparse_varlen_backend(hidden)
            if backend == "flex":
                return block.forward_selected_varlen(hidden, block_position, query_mask)
            if backend == "dense_compiled" and not self.activation_checkpointing:
                return block.forward_selected_dense(hidden, block_position, query_mask, compile_cuda=True)
            return checkpoint_if_enabled(self, block, hidden, block_position)[query_mask]
        return block.forward_selected(hidden, block_position, query_mask)

    def _apply_selected_blocks_sparse(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            mask = selected == action
            if torch.any(mask):
                next_hidden[mask] = self._forward_sparse_route_block(block, hidden, block_position, mask)
        return next_hidden

    def _apply_selected_blocks_grouped_dense(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        *,
        grouped_route_weights: tuple[Any, ...] | None = None,
    ) -> torch.Tensor:
        block_position = self._block_position(position)
        action_outputs = self._grouped_route_blocks_forward(hidden, block_position, grouped_route_weights)
        selected_output = self._gather_grouped_action_outputs(action_outputs, selected)
        internal = selected != self.out_action
        return torch.where(internal.unsqueeze(-1), selected_output, hidden)

    def _top_k_for_step(self, step: int) -> int:
        value = self.config.top_k if step <= 0 else self.config.later_top_k
        return max(1, min(int(value), self.config.route_pool_blocks + 1))

    def _topk_actions(self, probs: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = max(1, min(int(top_k), self.config.route_pool_blocks + 1))
        top_probs, top_actions = probs.topk(top_k, dim=-1)
        internal_mask = top_actions != self.out_action
        internal_weights = top_probs * internal_mask.float()
        denom = internal_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        top_weights = internal_weights / denom
        return top_actions, top_weights

    def _last_token_view(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 3:
            return tensor[:, -1, ...]
        if tensor.dim() == 2 and not torch.is_floating_point(tensor):
            return tensor[:, -1]
        return tensor

    def _apply_routed_blocks(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        *,
        grouped_route_weights: tuple[Any, ...] | None = None,
        weighted_fusion_possible: bool = True,
    ) -> torch.Tensor:
        if selected.dim() == 2:
            if self._grouped_dense_route_block_execution_enabled():
                return self._apply_routed_blocks_grouped_dense(
                    hidden,
                    position,
                    selected,
                    top_actions,
                    top_weights,
                    use_weighted_fusion,
                    grouped_route_weights=grouped_route_weights,
                    weighted_fusion_possible=weighted_fusion_possible,
                )
            if self._sparse_route_block_execution_enabled():
                return self._apply_routed_blocks_sparse(
                    hidden,
                    position,
                    selected,
                    top_actions,
                    top_weights,
                    use_weighted_fusion,
                )
            next_hidden = hidden.clone()
            block_position = self._block_position(position)
            for action, block in enumerate(self.route_blocks):
                top1_mask = (selected == action) & ~use_weighted_fusion
                if torch.any(top1_mask):
                    action_output = checkpoint_if_enabled(self, block, hidden, block_position)
                    next_hidden = torch.where(top1_mask.unsqueeze(-1), action_output, next_hidden)

            if torch.any(use_weighted_fusion):
                accum = torch.zeros_like(hidden)
                weight_sum = torch.zeros_like(selected, dtype=hidden.dtype)
                for action, block in enumerate(self.route_blocks):
                    action_output = None
                    for rank in range(top_actions.size(-1)):
                        mask = use_weighted_fusion & (top_actions[..., rank] == action)
                        if not torch.any(mask):
                            continue
                        if action_output is None:
                            action_output = checkpoint_if_enabled(self, block, hidden, block_position)
                        weight = top_weights[..., rank].to(hidden.dtype) * mask.to(hidden.dtype)
                        accum = accum + action_output * weight.unsqueeze(-1)
                        weight_sum = weight_sum + weight
                weighted_mask = use_weighted_fusion & (weight_sum > 0)
                if torch.any(weighted_mask):
                    weighted_hidden = accum / weight_sum.clamp_min(1e-9).unsqueeze(-1)
                    next_hidden = torch.where(weighted_mask.unsqueeze(-1), weighted_hidden, next_hidden)
                fallback_mask = use_weighted_fusion & (weight_sum <= 0)
                if torch.any(fallback_mask):
                    fallback_hidden = self._apply_selected_blocks(hidden, position, selected)
                    next_hidden = torch.where(fallback_mask.unsqueeze(-1), fallback_hidden, next_hidden)
            return next_hidden
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            top1_mask = (selected == action) & ~use_weighted_fusion
            if torch.any(top1_mask):
                next_hidden[top1_mask] = checkpoint_if_enabled(self, block, hidden[top1_mask], block_position[top1_mask])

        if torch.any(use_weighted_fusion):
            accum = torch.zeros_like(hidden)
            weight_sum = torch.zeros(hidden.size(0), dtype=hidden.dtype, device=hidden.device)
            for action, block in enumerate(self.route_blocks):
                for rank in range(top_actions.size(1)):
                    mask = use_weighted_fusion & (top_actions[:, rank] == action)
                    if not torch.any(mask):
                        continue
                    action_output = checkpoint_if_enabled(self, block, hidden[mask], block_position[mask])
                    weight = top_weights[mask, rank].to(hidden.dtype)
                    accum[mask] = accum[mask] + action_output * weight.view(-1, 1, 1)
                    weight_sum[mask] = weight_sum[mask] + weight
            weighted_mask = use_weighted_fusion & (weight_sum > 0)
            if torch.any(weighted_mask):
                next_hidden[weighted_mask] = accum[weighted_mask] / weight_sum[weighted_mask].view(-1, 1, 1)
            fallback_mask = use_weighted_fusion & (weight_sum <= 0)
            if torch.any(fallback_mask):
                next_hidden[fallback_mask] = self._apply_selected_blocks(hidden, position, selected)[fallback_mask]
        return next_hidden

    def _apply_routed_blocks_grouped_dense(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        *,
        grouped_route_weights: tuple[Any, ...] | None = None,
        weighted_fusion_possible: bool = True,
    ) -> torch.Tensor:
        block_position = self._block_position(position)
        return self._grouped_routed_blocks_forward(
            hidden,
            block_position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            grouped_route_weights,
            weighted_fusion_possible,
        )

    def _grouped_routed_blocks_forward(
        self,
        hidden: torch.Tensor,
        block_position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        grouped_route_weights: tuple[Any, ...] | None,
        weighted_fusion_possible: bool,
    ) -> torch.Tensor:
        if hidden.is_cuda:
            if self._compiled_grouped_routed_blocks_forward is None:
                self._compiled_grouped_routed_blocks_forward = torch.compile(
                    self._grouped_routed_blocks_forward_eager,
                    dynamic=False,
                )
            return self._compiled_grouped_routed_blocks_forward(
                hidden,
                block_position,
                selected,
                top_actions,
                top_weights,
                use_weighted_fusion,
                grouped_route_weights,
                weighted_fusion_possible,
            )
        return self._grouped_routed_blocks_forward_eager(
            hidden,
            block_position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            grouped_route_weights,
            weighted_fusion_possible,
        )

    def _grouped_routed_blocks_forward_eager(
        self,
        hidden: torch.Tensor,
        block_position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        grouped_route_weights: tuple[Any, ...] | None,
        weighted_fusion_possible: bool,
    ) -> torch.Tensor:
        action_outputs = self._grouped_route_blocks_forward_eager(hidden, block_position, grouped_route_weights)
        selected_output = self._gather_grouped_action_outputs(action_outputs, selected)
        base = torch.where((selected != self.out_action).unsqueeze(-1), selected_output, hidden)
        if not weighted_fusion_possible:
            return base

        top_outputs = self._gather_grouped_topk_outputs(action_outputs, top_actions)
        internal = top_actions != self.out_action
        weights = top_weights.to(hidden.dtype) * internal.to(hidden.dtype)
        weight_sum = weights.sum(dim=-1)
        weighted_hidden = (top_outputs * weights.unsqueeze(-1)).sum(dim=-2) / weight_sum.clamp_min(1e-9).unsqueeze(-1)
        weighted_mask = use_weighted_fusion & (weight_sum > 0)
        return torch.where(weighted_mask.unsqueeze(-1), weighted_hidden, base)

    def _gather_grouped_action_outputs(self, action_outputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        outputs_by_token = action_outputs.permute(1, 2, 0, 3)
        safe_actions = actions.clamp(min=0, max=self.config.route_pool_blocks - 1)
        gather_index = safe_actions.unsqueeze(-1).unsqueeze(-1).expand(*safe_actions.shape, 1, action_outputs.size(-1))
        return outputs_by_token.gather(dim=2, index=gather_index).squeeze(2)

    def _gather_grouped_topk_outputs(self, action_outputs: torch.Tensor, top_actions: torch.Tensor) -> torch.Tensor:
        outputs_by_token = action_outputs.permute(1, 2, 0, 3)
        safe_actions = top_actions.clamp(min=0, max=self.config.route_pool_blocks - 1)
        gather_index = safe_actions.unsqueeze(-1).expand(*safe_actions.shape, action_outputs.size(-1))
        return outputs_by_token.gather(dim=2, index=gather_index)

    def _grouped_route_blocks_forward(
        self,
        hidden: torch.Tensor,
        block_position: torch.Tensor,
        grouped_route_weights: tuple[Any, ...] | None = None,
    ) -> torch.Tensor:
        if hidden.is_cuda:
            if self._compiled_grouped_route_blocks_forward is None:
                self._compiled_grouped_route_blocks_forward = torch.compile(
                    self._grouped_route_blocks_forward_eager,
                    dynamic=False,
                )
            return self._compiled_grouped_route_blocks_forward(hidden, block_position, grouped_route_weights)
        return self._grouped_route_blocks_forward_eager(hidden, block_position, grouped_route_weights)

    def _grouped_route_block_weights(self) -> tuple[Any, ...]:
        if not self.route_blocks:
            return tuple()
        first = self.route_blocks[0]
        if any(block.position_injection != first.position_injection for block in self.route_blocks):
            raise ValueError("grouped_dense requires route blocks to share position injection mode.")
        if any(float(block.block.attn.dropout) != 0.0 for block in self.route_blocks):
            raise ValueError("grouped_dense route block execution currently requires attention dropout == 0.")
        adapter_weight = (
            None
            if first.position_injection == "direct_add"
            else torch.stack([block.position_adapter.weight for block in self.route_blocks])
        )
        return (
            adapter_weight,
            torch.stack([block.block.attn_norm.weight for block in self.route_blocks]),
            torch.stack([block.block.attn.qkv.weight for block in self.route_blocks]),
            torch.stack([block.block.attn.out.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn_norm.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w1.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w2.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w3.weight for block in self.route_blocks]),
        )

    def _grouped_pure_factorized_attention_global_weights(self) -> tuple[Any, ...]:
        if not self.route_blocks:
            return tuple()
        first = self.route_blocks[0]
        first_attn = first.block.attn
        if any(block.position_injection != first.position_injection for block in self.route_blocks):
            raise ValueError("grouped_selected attention global requires route blocks to share position injection mode.")
        if any(float(block.block.attn.dropout) != 0.0 for block in self.route_blocks):
            raise ValueError("grouped_selected attention global currently requires attention dropout == 0.")
        if first_attn.attention_global_kv_mode != "pure_factorized":
            raise ValueError("grouped_selected attention global requires pure_factorized Attention Global KV.")
        if first_attn.global_key_write is None or first_attn.global_value_write is None:
            raise ValueError("pure_factorized Attention Global KV writers are missing.")
        adapter_weight = (
            None
            if first.position_injection == "direct_add"
            else torch.stack([block.position_adapter.weight for block in self.route_blocks])
        )
        dim = self.config.base.d_model
        logit_biases = [
            block.block.attn._global_logit_bias(
                dtype=first.block.attn.global_key_write.weight.dtype,
                device=first.block.attn.global_key_write.weight.device,
            )
            for block in self.route_blocks
        ]
        return (
            adapter_weight,
            torch.stack([block.block.attn_norm.weight for block in self.route_blocks]),
            torch.stack([block.block.attn.qkv.weight[:dim, :] for block in self.route_blocks]),
            torch.stack([block.block.attn.out.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn_norm.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w1.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w2.weight for block in self.route_blocks]),
            torch.stack([block.block.ffn.w3.weight for block in self.route_blocks]),
            first_attn.global_key_write.weight,
            first_attn.global_value_write.weight,
            torch.stack([block.block.attn.global_key_head_read for block in self.route_blocks]),
            torch.stack([block.block.attn.global_value_head_read for block in self.route_blocks]),
            torch.stack(logit_biases),
        )

    def _grouped_route_blocks_forward_eager(
        self,
        hidden: torch.Tensor,
        block_position: torch.Tensor,
        grouped_route_weights: tuple[Any, ...] | None = None,
    ) -> torch.Tensor:
        if not self.route_blocks:
            return hidden.new_empty((0, *hidden.shape))
        first = self.route_blocks[0]
        weights = grouped_route_weights if grouped_route_weights is not None else self._grouped_route_block_weights()
        (
            adapter_weight,
            attn_norm_weight,
            qkv_weight,
            out_weight,
            ffn_norm_weight,
            w1,
            w2,
            w3,
        ) = weights
        num_blocks = len(self.route_blocks)
        batch, seq_len, dim = hidden.shape
        if first.position_injection == "direct_add":
            bias = block_position.unsqueeze(0).expand(num_blocks, -1, -1, -1)
        else:
            if adapter_weight is None:
                raise ValueError("grouped_dense adapter weights are missing.")
            bias = torch.einsum("bsp,edp->ebsd", block_position, adapter_weight)
        x = hidden.unsqueeze(0) + bias

        attn_norm_weight = attn_norm_weight.view(num_blocks, 1, 1, dim)
        attn_norm_eps = first.block.attn_norm.eps
        attn_input = attn_norm_weight * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + attn_norm_eps) * x
        qkv = torch.einsum("ebsd,eod->ebso", attn_input, qkv_weight)
        q, k, v = qkv.chunk(3, dim=-1)
        n_heads = first.block.attn.n_heads
        head_dim = first.block.attn.head_dim
        q = q.view(num_blocks, batch, seq_len, n_heads, head_dim).permute(0, 1, 3, 2, 4)
        k = k.view(num_blocks, batch, seq_len, n_heads, head_dim).permute(0, 1, 3, 2, 4)
        v = v.view(num_blocks, batch, seq_len, n_heads, head_dim).permute(0, 1, 3, 2, 4)
        q = q.reshape(num_blocks * batch, n_heads, seq_len, head_dim)
        k = k.reshape(num_blocks * batch, n_heads, seq_len, head_dim)
        v = v.reshape(num_blocks * batch, n_heads, seq_len, head_dim)
        cos = first.block.attn.rope.cos[:, :, :seq_len, :].to(device=q.device)
        sin = first.block.attn.rope.sin[:, :, :seq_len, :].to(device=q.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.reshape(num_blocks, batch, n_heads, seq_len, head_dim).permute(0, 1, 3, 2, 4)
        y = y.reshape(num_blocks, batch, seq_len, dim)
        x = x + torch.einsum("ebsd,eod->ebso", y, out_weight)

        ffn_norm_weight = ffn_norm_weight.view(num_blocks, 1, 1, dim)
        ffn_norm_eps = first.block.ffn_norm.eps
        ffn_input = ffn_norm_weight * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + ffn_norm_eps) * x
        gate = torch.einsum("ebsd,ehd->ebsh", ffn_input, w1)
        up = torch.einsum("ebsd,ehd->ebsh", ffn_input, w2)
        ffn_hidden = F.silu(gate) * up
        return x + torch.einsum("ebsh,edh->ebsd", ffn_hidden, w3)

    def _apply_routed_blocks_sparse(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
    ) -> torch.Tensor:
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            top1_mask = (selected == action) & ~use_weighted_fusion
            if torch.any(top1_mask):
                next_hidden[top1_mask] = self._forward_sparse_route_block(block, hidden, block_position, top1_mask)

        if torch.any(use_weighted_fusion):
            accum = torch.zeros_like(hidden)
            weight_sum = torch.zeros_like(selected, dtype=hidden.dtype)
            for action, block in enumerate(self.route_blocks):
                action_mask = torch.zeros_like(selected, dtype=torch.bool)
                for rank in range(top_actions.size(-1)):
                    action_mask = action_mask | (use_weighted_fusion & (top_actions[..., rank] == action))
                if not torch.any(action_mask):
                    continue
                action_output = self._forward_sparse_route_block(block, hidden, block_position, action_mask)
                flat_action_mask = action_mask.reshape(-1)
                local_index = torch.empty_like(selected.reshape(-1), dtype=torch.long)
                local_index[flat_action_mask] = torch.arange(action_output.size(0), device=hidden.device)
                flat_accum = accum.reshape(-1, hidden.size(-1))
                flat_weight_sum = weight_sum.reshape(-1)
                for rank in range(top_actions.size(-1)):
                    mask = use_weighted_fusion & (top_actions[..., rank] == action)
                    if not torch.any(mask):
                        continue
                    flat_mask = mask.reshape(-1)
                    weight = top_weights[..., rank].to(hidden.dtype).reshape(-1)[flat_mask]
                    flat_accum[flat_mask] = (
                        flat_accum[flat_mask] + action_output[local_index[flat_mask]] * weight.unsqueeze(-1)
                    )
                    flat_weight_sum[flat_mask] = flat_weight_sum[flat_mask] + weight
            weighted_mask = use_weighted_fusion & (weight_sum > 0)
            if torch.any(weighted_mask):
                weighted_hidden = accum / weight_sum.clamp_min(1e-9).unsqueeze(-1)
                next_hidden = torch.where(weighted_mask.unsqueeze(-1), weighted_hidden, next_hidden)
            fallback_mask = use_weighted_fusion & (weight_sum <= 0)
            if torch.any(fallback_mask):
                fallback_hidden = self._apply_selected_blocks_sparse(hidden, position, selected)
                next_hidden = torch.where(fallback_mask.unsqueeze(-1), fallback_hidden, next_hidden)
        return next_hidden

    def _apply_routed_blocks_with_attention_global(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
        *,
        grouped_attention_global_weights: tuple[Any, ...] | None = None,
        weighted_fusion_possible: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if selected.dim() == 2 and self.config.attention_global_kv_mode == "pure_factorized":
            if self.config.attention_global_route_execution == "cache_only":
                if weighted_fusion_possible:
                    raise ValueError("cache_only attention global execution currently supports top-1 routing only.")
                return self._apply_routed_blocks_with_pure_factorized_attention_global_cache_only(
                    hidden,
                    position,
                    selected,
                    attention_global_state,
                    route_info,
                    grouped_attention_global_weights,
                )
            if (
                self.config.attention_global_route_execution == "grouped_selected"
                and not weighted_fusion_possible
            ):
                return self._apply_routed_blocks_with_pure_factorized_attention_global_grouped_selected(
                    hidden,
                    position,
                    selected,
                    attention_global_state,
                    route_info,
                    grouped_attention_global_weights,
                )
            if (
                self.config.attention_global_route_execution == "top1_fast"
                and not weighted_fusion_possible
            ):
                return self._apply_routed_blocks_with_pure_factorized_attention_global_top1_fast(
                    hidden,
                    position,
                    selected,
                    attention_global_state,
                    route_info,
                )
            return self._apply_routed_blocks_with_pure_factorized_attention_global_selected(
                hidden,
                position,
                selected,
                top_actions,
                top_weights,
                use_weighted_fusion,
                attention_global_state,
                route_info,
            )
        if selected.dim() == 2:
            next_hidden = hidden.clone()
            write_key, write_value, write_valid = self._empty_attention_global_write(hidden)
            block_position = self._block_position(position)
            for action, block in enumerate(self.route_blocks):
                top1_mask = (selected == action) & ~use_weighted_fusion
                if torch.any(top1_mask):
                    output, key_summary, value_summary, metrics = block(
                        hidden,
                        block_position,
                        attention_global_state,
                        return_attention_kv=True,
                    )
                    key_write = self._attention_global_write_tensor(key_summary).to(write_key.dtype)
                    value_write = self._attention_global_write_tensor(value_summary).to(write_value.dtype)
                    token_mask = top1_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                    next_hidden = torch.where(top1_mask.unsqueeze(-1), output, next_hidden)
                    write_key = torch.where(token_mask, key_write, write_key)
                    write_value = torch.where(token_mask, value_write, write_value)
                    write_valid = write_valid | top1_mask.unsqueeze(1)
                    self._append_attention_global_metrics(route_info, metrics)

            if torch.any(use_weighted_fusion):
                accum = torch.zeros_like(hidden)
                key_accum = torch.zeros_like(write_key)
                value_accum = torch.zeros_like(write_value)
                weight_sum = torch.zeros_like(selected, dtype=hidden.dtype)
                for action, block in enumerate(self.route_blocks):
                    action_output = None
                    key_write = None
                    value_write = None
                    for rank in range(top_actions.size(-1)):
                        mask = use_weighted_fusion & (top_actions[..., rank] == action)
                        if not torch.any(mask):
                            continue
                        if action_output is None:
                            action_output, key_summary, value_summary, metrics = block(
                                hidden,
                                block_position,
                                attention_global_state,
                                return_attention_kv=True,
                            )
                            key_write = self._attention_global_write_tensor(key_summary).to(key_accum.dtype)
                            value_write = self._attention_global_write_tensor(value_summary).to(value_accum.dtype)
                            self._append_attention_global_metrics(route_info, metrics)
                        weight = top_weights[..., rank].to(hidden.dtype) * mask.to(hidden.dtype)
                        accum = accum + action_output * weight.unsqueeze(-1)
                        key_weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                        assert key_write is not None and value_write is not None
                        key_accum = key_accum + key_write * key_weight
                        value_accum = value_accum + value_write * key_weight
                        weight_sum = weight_sum + weight
                weighted_mask = use_weighted_fusion & (weight_sum > 0)
                if torch.any(weighted_mask):
                    denom = weight_sum.clamp_min(1e-9)
                    weighted_hidden = accum / denom.unsqueeze(-1)
                    write_denom = denom.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                    weighted_key = key_accum / write_denom
                    weighted_value = value_accum / write_denom
                    token_mask = weighted_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                    next_hidden = torch.where(weighted_mask.unsqueeze(-1), weighted_hidden, next_hidden)
                    write_key = torch.where(token_mask, weighted_key, write_key)
                    write_value = torch.where(token_mask, weighted_value, write_value)
                    write_valid = write_valid | weighted_mask.unsqueeze(1)
                fallback_mask = use_weighted_fusion & (weight_sum <= 0)
                if torch.any(fallback_mask):
                    selected_hidden, selected_key, selected_value, selected_valid = self._apply_selected_blocks_with_attention_global(
                        hidden,
                        position,
                        selected,
                        attention_global_state,
                        route_info,
                    )
                    token_mask = fallback_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                    next_hidden = torch.where(fallback_mask.unsqueeze(-1), selected_hidden, next_hidden)
                    write_key = torch.where(token_mask, selected_key, write_key)
                    write_value = torch.where(token_mask, selected_value, write_value)
                    write_valid = write_valid | (selected_valid & fallback_mask.unsqueeze(1))
            return next_hidden, write_key, write_value, write_valid
        next_hidden = hidden.clone()
        write_key, write_value, write_valid = self._empty_attention_global_write(hidden)
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            top1_mask = (selected == action) & ~use_weighted_fusion
            if torch.any(top1_mask):
                indices = torch.nonzero(top1_mask, as_tuple=False).flatten()
                output, key_summary, value_summary, metrics = block(
                    hidden[indices],
                    block_position[indices],
                    attention_global_state.index_select(indices),
                    return_attention_kv=True,
                )
                next_hidden[indices] = output
                write_key[indices] = self._attention_global_write_tensor(key_summary).to(write_key.dtype)
                write_value[indices] = self._attention_global_write_tensor(value_summary).to(write_value.dtype)
                write_valid[indices] = True
                self._append_attention_global_metrics(route_info, metrics)

        if torch.any(use_weighted_fusion):
            accum = torch.zeros_like(hidden)
            key_accum = torch.zeros_like(write_key)
            value_accum = torch.zeros_like(write_value)
            weight_sum = torch.zeros(hidden.size(0), dtype=hidden.dtype, device=hidden.device)
            for action, block in enumerate(self.route_blocks):
                for rank in range(top_actions.size(1)):
                    mask = use_weighted_fusion & (top_actions[:, rank] == action)
                    if not torch.any(mask):
                        continue
                    indices = torch.nonzero(mask, as_tuple=False).flatten()
                    action_output, key_summary, value_summary, metrics = block(
                        hidden[indices],
                        block_position[indices],
                        attention_global_state.index_select(indices),
                        return_attention_kv=True,
                    )
                    weight = top_weights[indices, rank].to(hidden.dtype)
                    accum[indices] = accum[indices] + action_output * weight.view(-1, 1, 1)
                    write_key_summary = self._attention_global_write_tensor(key_summary).to(key_accum.dtype)
                    write_value_summary = self._attention_global_write_tensor(value_summary).to(value_accum.dtype)
                    key_accum[indices] = key_accum[indices] + write_key_summary * weight.view(-1, 1, 1, 1, 1)
                    value_accum[indices] = (
                        value_accum[indices] + write_value_summary * weight.view(-1, 1, 1, 1, 1)
                    )
                    weight_sum[indices] = weight_sum[indices] + weight
                    self._append_attention_global_metrics(route_info, metrics)
            weighted_mask = use_weighted_fusion & (weight_sum > 0)
            if torch.any(weighted_mask):
                denom = weight_sum[weighted_mask]
                next_hidden[weighted_mask] = accum[weighted_mask] / denom.view(-1, 1, 1)
                write_key[weighted_mask] = key_accum[weighted_mask] / denom.view(-1, 1, 1, 1, 1)
                write_value[weighted_mask] = value_accum[weighted_mask] / denom.view(-1, 1, 1, 1, 1)
                write_valid[weighted_mask] = True
            fallback_mask = use_weighted_fusion & (weight_sum <= 0)
            if torch.any(fallback_mask):
                selected_hidden, selected_key, selected_value, selected_valid = self._apply_selected_blocks_with_attention_global(
                    hidden,
                    position,
                    selected,
                    attention_global_state,
                    route_info,
                )
                next_hidden[fallback_mask] = selected_hidden[fallback_mask]
                write_key[fallback_mask] = selected_key[fallback_mask]
                write_value[fallback_mask] = selected_value[fallback_mask]
                write_valid[fallback_mask] = selected_valid[fallback_mask]
        return next_hidden, write_key, write_value, write_valid

    def _apply_routed_blocks_with_pure_factorized_attention_global_cache_only(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
        grouped_attention_global_weights: tuple[Any, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del route_info
        weights = (
            grouped_attention_global_weights
            if grouped_attention_global_weights is not None
            else self._grouped_pure_factorized_attention_global_weights()
        )
        if not weights:
            return hidden, *self._empty_attention_global_write(hidden)
        first = self.route_blocks[0]
        (
            adapter_weight,
            attn_norm_weight,
            q_weight,
            out_weight,
            ffn_norm_weight,
            w1,
            w2,
            w3,
            global_key_write_weight,
            global_value_write_weight,
            global_key_head_read,
            global_value_head_read,
            global_logit_bias,
        ) = weights
        num_blocks = len(self.route_blocks)
        batch, seq_len, dim = hidden.shape
        code_dim = self._attention_global_cache_dim()
        action_ids = torch.arange(num_blocks, device=selected.device).view(num_blocks, 1, 1)
        query_mask = selected.unsqueeze(0) == action_ids
        expert_indices, batch_indices, query_positions = torch.where(query_mask)
        next_hidden = hidden.clone()
        write_key = hidden.new_empty((batch, self._attention_global_cache_heads(), 1, seq_len, code_dim))
        write_value = hidden.new_empty((batch, self._attention_global_cache_heads(), 1, seq_len, code_dim))
        write_valid = torch.zeros(batch, 1, seq_len, dtype=torch.bool, device=hidden.device)
        if expert_indices.numel() == 0:
            return next_hidden, write_key, write_value, write_valid

        counts = query_mask.sum(dim=-1)
        group_experts, group_batches = torch.where(counts > 0)
        active_counts = counts[group_experts, group_batches]
        group_count = int(active_counts.numel())
        max_selected = int(active_counts.max().item())
        offsets = torch.repeat_interleave(torch.cumsum(active_counts, dim=0) - active_counts, active_counts)
        local_indices = torch.arange(expert_indices.numel(), device=hidden.device) - offsets
        group_lookup = torch.full((num_blocks * batch,), -1, dtype=torch.long, device=hidden.device)
        group_lookup[group_experts * batch + group_batches] = torch.arange(group_count, device=hidden.device)
        token_group_indices = group_lookup[expert_indices * batch + batch_indices]

        selected_hidden_padded = hidden.new_zeros((group_count, max_selected, dim))
        selected_position_padded = position.new_zeros((group_count, max_selected, position.size(-1)))
        query_positions_padded = torch.zeros(group_count, max_selected, dtype=torch.long, device=hidden.device)
        query_valid = torch.zeros(group_count, max_selected, dtype=torch.bool, device=hidden.device)
        block_position = self._block_position(position)
        selected_hidden_padded[token_group_indices, local_indices] = hidden[batch_indices, query_positions]
        selected_position_padded[token_group_indices, local_indices] = block_position[batch_indices, query_positions]
        query_positions_padded[token_group_indices, local_indices] = query_positions
        query_valid[token_group_indices, local_indices] = True

        if first.position_injection == "direct_add":
            bias = selected_position_padded
        else:
            if adapter_weight is None:
                raise ValueError("cache_only adapter weights are missing.")
            bias = torch.einsum("gqp,gdp->gqd", selected_position_padded, adapter_weight[group_experts])
        x = selected_hidden_padded + bias

        attn_norm_eps = first.block.attn_norm.eps
        attn_input = (
            attn_norm_weight[group_experts].view(group_count, 1, dim).to(dtype=x.dtype)
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + attn_norm_eps)
            * x
        )
        n_heads = first.block.attn.n_heads
        head_dim = first.block.attn.head_dim
        q = torch.einsum("gqd,god->gqo", attn_input, q_weight[group_experts].to(dtype=attn_input.dtype))
        q = q.view(group_count, max_selected, n_heads, head_dim)
        q_code = torch.einsum(
            "gqhd,ghdc->gqhc",
            q,
            global_key_head_read[group_experts].to(dtype=q.dtype),
        )

        key_code, value_code, allowed = self._grouped_pure_factorized_previous_pool_for_positions(
            attention_global_state,
            group_batches,
            query_positions_padded,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        if key_code.size(1) == 0:
            attn_out = x.new_zeros((group_count, max_selected, dim))
        else:
            attn_mask = torch.zeros(
                group_count,
                1,
                max_selected,
                key_code.size(1),
                dtype=q_code.dtype,
                device=hidden.device,
            )
            attn_mask = attn_mask + global_logit_bias.to(dtype=q_code.dtype, device=hidden.device)[group_experts].view(
                group_count,
                1,
                1,
                1,
            )
            attn_mask = attn_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(q_code.dtype).min)
            value_code_out = F.scaled_dot_product_attention(
                q_code.permute(0, 2, 1, 3),
                key_code[:, None, :, :].expand(-1, n_heads, -1, -1),
                value_code[:, None, :, :].expand(-1, n_heads, -1, -1),
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=first.block.attn.dropout if self.training else 0.0,
                scale=code_dim**-0.5,
            )
            selected_code = value_code_out.permute(0, 2, 1, 3).to(dtype=q.dtype)
            y = torch.einsum(
                "gqhc,ghcd->gqhd",
                selected_code,
                global_value_head_read[group_experts].to(dtype=selected_code.dtype),
            ).reshape(group_count, max_selected, dim)
            attn_out = torch.einsum("gqd,god->gqo", y, out_weight[group_experts].to(dtype=y.dtype))

        selected_hidden = x + attn_out
        ffn_norm_eps = first.block.ffn_norm.eps
        ffn_input = (
            ffn_norm_weight[group_experts].view(group_count, 1, dim).to(dtype=selected_hidden.dtype)
            * torch.rsqrt(selected_hidden.pow(2).mean(dim=-1, keepdim=True) + ffn_norm_eps)
            * selected_hidden
        )
        gate = torch.einsum("gqd,ghd->gqh", ffn_input, w1[group_experts].to(dtype=ffn_input.dtype))
        up = torch.einsum("gqd,ghd->gqh", ffn_input, w2[group_experts].to(dtype=ffn_input.dtype))
        ffn_hidden = F.silu(gate) * up
        output_padded = selected_hidden + torch.einsum(
            "gqh,gdh->gqd",
            ffn_hidden,
            w3[group_experts].to(dtype=ffn_hidden.dtype),
        )
        output = output_padded[query_valid]

        write_payload_key = F.linear(output, global_key_write_weight.to(dtype=output.dtype))
        write_payload_value = F.linear(output, global_value_write_weight.to(dtype=output.dtype))
        flat_indices = batch_indices * seq_len + query_positions
        flat_next = next_hidden.reshape(-1, dim)
        flat_write_key = write_key[:, 0, 0].reshape(-1, code_dim)
        flat_write_value = write_value[:, 0, 0].reshape(-1, code_dim)
        flat_write_valid = write_valid[:, 0].reshape(-1)
        flat_next.index_copy_(0, flat_indices, output)
        flat_write_key.index_copy_(0, flat_indices, write_payload_key.to(flat_write_key.dtype))
        flat_write_value.index_copy_(0, flat_indices, write_payload_value.to(flat_write_value.dtype))
        flat_write_valid.index_fill_(0, flat_indices, True)
        return next_hidden, write_key, write_value, write_valid

    def _apply_routed_blocks_with_pure_factorized_attention_global_grouped_selected(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
        grouped_attention_global_weights: tuple[Any, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (
            grouped_attention_global_weights
            if grouped_attention_global_weights is not None
            else self._grouped_pure_factorized_attention_global_weights()
        )
        if not weights:
            return hidden, *self._empty_attention_global_write(hidden)
        first = self.route_blocks[0]
        (
            adapter_weight,
            attn_norm_weight,
            q_weight,
            out_weight,
            ffn_norm_weight,
            w1,
            w2,
            w3,
            global_key_write_weight,
            global_value_write_weight,
            global_key_head_read,
            global_value_head_read,
            global_logit_bias,
        ) = weights
        num_blocks = len(self.route_blocks)
        batch, seq_len, dim = hidden.shape
        code_dim = self._attention_global_cache_dim()
        action_ids = torch.arange(num_blocks, device=selected.device).view(num_blocks, 1, 1)
        query_mask = selected.unsqueeze(0) == action_ids
        expert_indices, batch_indices, query_positions = torch.where(query_mask)
        next_hidden = hidden.clone()
        write_key = hidden.new_empty((batch, self._attention_global_cache_heads(), 1, seq_len, code_dim))
        write_value = hidden.new_empty((batch, self._attention_global_cache_heads(), 1, seq_len, code_dim))
        write_valid = torch.zeros(batch, 1, seq_len, dtype=torch.bool, device=hidden.device)
        if expert_indices.numel() == 0:
            return next_hidden, write_key, write_value, write_valid

        counts = query_mask.sum(dim=-1)
        group_experts, group_batches = torch.where(counts > 0)
        active_counts = counts[group_experts, group_batches]
        group_count = int(active_counts.numel())
        max_selected = int(active_counts.max().item())
        offsets = torch.repeat_interleave(torch.cumsum(active_counts, dim=0) - active_counts, active_counts)
        local_indices = torch.arange(expert_indices.numel(), device=hidden.device) - offsets
        group_lookup = torch.full((num_blocks * batch,), -1, dtype=torch.long, device=hidden.device)
        group_lookup[group_experts * batch + group_batches] = torch.arange(group_count, device=hidden.device)
        token_group_indices = group_lookup[expert_indices * batch + batch_indices]

        block_position = self._block_position(position).index_select(0, group_batches)
        grouped_hidden = hidden.index_select(0, group_batches)
        if first.position_injection == "direct_add":
            bias = block_position
        else:
            if adapter_weight is None:
                raise ValueError("grouped_selected adapter weights are missing.")
            bias = torch.einsum("gsp,gdp->gsd", block_position, adapter_weight[group_experts])
        x = grouped_hidden + bias

        grouped_attn_norm_weight = attn_norm_weight[group_experts].view(group_count, 1, dim)
        attn_norm_eps = first.block.attn_norm.eps
        attn_input = grouped_attn_norm_weight * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + attn_norm_eps
        ) * x
        current_key_code = torch.einsum(
            "gsd,cd->gsc",
            attn_input,
            global_key_write_weight.to(dtype=attn_input.dtype),
        )
        current_value_code = torch.einsum(
            "gsd,cd->gsc",
            attn_input,
            global_value_write_weight.to(dtype=attn_input.dtype),
        )

        n_heads = first.block.attn.n_heads
        head_dim = first.block.attn.head_dim
        selected_attn_input_padded = attn_input.new_zeros((group_count, max_selected, dim))
        selected_x_padded = x.new_zeros((group_count, max_selected, dim))
        query_positions_padded = torch.zeros(
            group_count,
            max_selected,
            dtype=torch.long,
            device=hidden.device,
        )
        query_valid = torch.zeros(group_count, max_selected, dtype=torch.bool, device=hidden.device)
        selected_attn_input_padded[token_group_indices, local_indices] = attn_input[
            token_group_indices,
            query_positions,
        ]
        selected_x_padded[token_group_indices, local_indices] = x[token_group_indices, query_positions]
        query_positions_padded[token_group_indices, local_indices] = query_positions
        query_valid[token_group_indices, local_indices] = True
        q = torch.einsum(
            "gqd,god->gqo",
            selected_attn_input_padded,
            q_weight[group_experts].to(dtype=selected_attn_input_padded.dtype),
        )
        q = q.view(group_count, max_selected, n_heads, head_dim)
        q_code_padded = torch.einsum(
            "gqhd,ghdc->gqhc",
            q,
            global_key_head_read[group_experts].to(dtype=q.dtype),
        )

        previous_key_code, previous_value_code, previous_allowed = self._grouped_pure_factorized_previous_pool_for_positions(
            attention_global_state,
            group_batches,
            query_positions_padded,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        key_positions = torch.arange(seq_len, device=hidden.device)
        current_allowed = key_positions.view(1, 1, seq_len) <= query_positions_padded.unsqueeze(-1)
        key_code = torch.cat([previous_key_code, current_key_code], dim=1)
        value_code = torch.cat([previous_value_code, current_value_code], dim=1)
        allowed = torch.cat([previous_allowed, current_allowed], dim=-1)

        previous_key_count = previous_key_code.size(1)
        attn_mask = torch.zeros(
            group_count,
            1,
            max_selected,
            key_code.size(1),
            dtype=q_code_padded.dtype,
            device=hidden.device,
        )
        if previous_key_count:
            attn_mask[..., :previous_key_count] = global_logit_bias.to(
                dtype=q_code_padded.dtype,
                device=hidden.device,
            )[group_experts].view(group_count, 1, 1, 1)
        attn_mask = attn_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(q_code_padded.dtype).min)
        q_for_attention = q_code_padded.permute(0, 2, 1, 3)
        key_for_attention = key_code[:, None, :, :].expand(-1, n_heads, -1, -1)
        value_for_attention = value_code[:, None, :, :].expand(-1, n_heads, -1, -1)
        value_code_out = F.scaled_dot_product_attention(
            q_for_attention,
            key_for_attention,
            value_for_attention,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=first.block.attn.dropout if self.training else 0.0,
            scale=code_dim**-0.5,
        )
        selected_code_padded = value_code_out.permute(0, 2, 1, 3).to(dtype=q.dtype)
        y = torch.einsum(
            "gqhc,ghcd->gqhd",
            selected_code_padded,
            global_value_head_read[group_experts].to(dtype=selected_code_padded.dtype),
        ).reshape(group_count, max_selected, dim)
        attn_out = torch.einsum("gqd,god->gqo", y, out_weight[group_experts].to(dtype=y.dtype))

        selected_hidden = selected_x_padded + attn_out
        ffn_norm_eps = first.block.ffn_norm.eps
        ffn_input = (
            ffn_norm_weight[group_experts].to(dtype=selected_hidden.dtype).view(group_count, 1, dim)
            * torch.rsqrt(selected_hidden.pow(2).mean(dim=-1, keepdim=True) + ffn_norm_eps)
            * selected_hidden
        )
        gate = torch.einsum("gqd,ghd->gqh", ffn_input, w1[group_experts].to(dtype=ffn_input.dtype))
        up = torch.einsum("gqd,ghd->gqh", ffn_input, w2[group_experts].to(dtype=ffn_input.dtype))
        ffn_hidden = F.silu(gate) * up
        output_padded = selected_hidden + torch.einsum(
            "gqh,gdh->gqd",
            ffn_hidden,
            w3[group_experts].to(dtype=ffn_hidden.dtype),
        )
        output = output_padded[query_valid]

        flat_indices = batch_indices * seq_len + query_positions
        flat_next = next_hidden.reshape(-1, dim)
        flat_write_key = write_key[:, 0, 0].reshape(-1, code_dim)
        flat_write_value = write_value[:, 0, 0].reshape(-1, code_dim)
        flat_write_valid = write_valid[:, 0].reshape(-1)
        flat_next.index_copy_(0, flat_indices, output)
        flat_write_key.index_copy_(
            0,
            flat_indices,
            current_key_code[token_group_indices, query_positions].to(flat_write_key.dtype),
        )
        flat_write_value.index_copy_(
            0,
            flat_indices,
            current_value_code[token_group_indices, query_positions].to(flat_write_value.dtype),
        )
        flat_write_valid.index_fill_(0, flat_indices, True)

        global_keys = getattr(attention_global_state, "keys", None)
        global_values = getattr(attention_global_state, "values", None)
        global_valid = getattr(attention_global_state, "valid", None)
        for group_index in range(group_count):
            action = int(group_experts[group_index].item())
            batch_index = group_batches[group_index : group_index + 1]
            metrics = self.route_blocks[action].block.attn._pure_factorized_metrics_for_last_token(
                attn_input[group_index : group_index + 1],
                q_weight[action],
                current_key_code[group_index : group_index + 1],
                global_keys.index_select(0, batch_index) if global_keys is not None else None,
                global_values.index_select(0, batch_index) if global_values is not None else None,
                global_valid.index_select(0, batch_index) if global_valid is not None else None,
            )
            self._append_attention_global_metrics(route_info, metrics)

        return next_hidden, write_key, write_value, write_valid

    def _grouped_pure_factorized_previous_pool_for_positions(
        self,
        attention_global_state: AttentionGlobalKVState,
        group_batches: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group_count, query_len = query_positions.shape
        code_dim = self._attention_global_cache_dim()
        global_keys = getattr(attention_global_state, "keys", None)
        global_values = getattr(attention_global_state, "values", None)
        global_valid = getattr(attention_global_state, "valid", None)
        if global_keys is None or global_values is None or global_valid is None or int(global_keys.size(2)) == 0:
            return (
                torch.empty(group_count, 0, code_dim, device=device, dtype=dtype),
                torch.empty(group_count, 0, code_dim, device=device, dtype=dtype),
                torch.empty(group_count, query_len, 0, device=device, dtype=torch.bool),
            )
        if global_keys.dim() != 5 or global_values.dim() != 5:
            raise ValueError("pure_factorized Attention Global KV requires token-shaped rank-5 cache state.")
        _, heads, slots, global_seq_len, cached_code_dim = global_keys.shape
        if heads != 1:
            raise ValueError("pure_factorized global pool stores shared headless key/value codes.")
        if cached_code_dim != code_dim:
            raise ValueError("pure_factorized global pool code dim does not match model config.")
        key_code = global_keys.index_select(0, group_batches).to(device=device, dtype=dtype).reshape(
            group_count,
            slots * global_seq_len,
            code_dim,
        )
        value_code = global_values.index_select(0, group_batches).to(device=device, dtype=dtype).reshape(
            group_count,
            slots * global_seq_len,
            code_dim,
        )
        valid = global_valid.index_select(0, group_batches).to(device=device, dtype=torch.bool)
        global_pos = torch.arange(global_seq_len, device=device)
        causal = global_pos.view(1, 1, 1, global_seq_len) <= query_positions.view(
            group_count,
            query_len,
            1,
            1,
        )
        allowed = (valid.view(group_count, 1, slots, global_seq_len) & causal).reshape(
            group_count,
            query_len,
            slots * global_seq_len,
        )
        return key_code, value_code, allowed

    def _apply_routed_blocks_with_pure_factorized_attention_global_top1_fast(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_hidden = hidden.clone()
        dim = self._attention_global_cache_dim()
        shape = (hidden.size(0), self._attention_global_cache_heads(), 1, hidden.size(1), dim)
        write_key = hidden.new_empty(shape)
        write_value = hidden.new_empty(shape)
        write_valid = torch.zeros(hidden.size(0), 1, hidden.size(1), dtype=torch.bool, device=hidden.device)
        block_position = self._block_position(position)
        flat_selected = selected.reshape(-1)
        flat_next = next_hidden.reshape(-1, hidden.size(-1))
        flat_write_key = write_key[:, 0, 0].reshape(-1, write_key.size(-1))
        flat_write_value = write_value[:, 0, 0].reshape(-1, write_value.size(-1))
        flat_write_valid = write_valid[:, 0].reshape(-1)

        for action, block in enumerate(self.route_blocks):
            flat_mask = flat_selected == action
            if not torch.any(flat_mask):
                continue
            mask = flat_mask.view_as(selected)
            indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
            output, key_summary, value_summary, metrics = block.forward_selected_attention_global(
                hidden,
                block_position,
                mask,
                attention_global_state,
                return_attention_kv=True,
            )
            flat_next.index_copy_(0, indices, output)
            flat_write_key.index_copy_(0, indices, key_summary.to(flat_write_key.dtype))
            flat_write_value.index_copy_(0, indices, value_summary.to(flat_write_value.dtype))
            flat_write_valid.index_fill_(0, indices, True)
            self._append_attention_global_metrics(route_info, metrics)

        return next_hidden, write_key, write_value, write_valid

    def _apply_routed_blocks_with_pure_factorized_attention_global_selected(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_hidden = hidden.clone()
        write_key, write_value, write_valid = self._empty_attention_global_write(hidden)
        block_position = self._block_position(position)
        flat_next = next_hidden.reshape(-1, hidden.size(-1))
        flat_write_key = write_key[:, 0, 0].reshape(-1, write_key.size(-1))
        flat_write_value = write_value[:, 0, 0].reshape(-1, write_value.size(-1))
        flat_write_valid = write_valid[:, 0].reshape(-1)

        def apply_action_mask(action: int, mask: torch.Tensor) -> None:
            flat_mask = mask.reshape(-1)
            if not torch.any(flat_mask):
                return
            indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
            output, key_summary, value_summary, metrics = self.route_blocks[action].forward_selected_attention_global(
                hidden,
                block_position,
                mask,
                attention_global_state,
                return_attention_kv=True,
            )
            flat_next[indices] = output
            flat_write_key[indices] = key_summary.to(flat_write_key.dtype)
            flat_write_value[indices] = value_summary.to(flat_write_value.dtype)
            flat_write_valid[indices] = True
            self._append_attention_global_metrics(route_info, metrics)

        for action in range(len(self.route_blocks)):
            apply_action_mask(action, (selected == action) & ~use_weighted_fusion)

        if torch.any(use_weighted_fusion):
            flat_size = hidden.size(0) * hidden.size(1)
            accum = hidden.new_zeros((flat_size, hidden.size(-1)))
            key_accum = write_key.new_zeros((flat_size, write_key.size(-1)))
            value_accum = write_value.new_zeros((flat_size, write_value.size(-1)))
            weight_sum = hidden.new_zeros((flat_size,))
            flat_top_weights = top_weights.reshape(flat_size, top_weights.size(-1))
            flat_use_weighted = use_weighted_fusion.reshape(-1)

            for action, block in enumerate(self.route_blocks):
                action_rank_mask = (top_actions == action) & use_weighted_fusion.unsqueeze(-1)
                action_query_mask = action_rank_mask.any(dim=-1)
                if not torch.any(action_query_mask):
                    continue
                action_indices = torch.nonzero(action_query_mask.reshape(-1), as_tuple=False).flatten()
                output, key_summary, value_summary, metrics = block.forward_selected_attention_global(
                    hidden,
                    block_position,
                    action_query_mask,
                    attention_global_state,
                    return_attention_kv=True,
                )
                self._append_attention_global_metrics(route_info, metrics)
                flat_action_rank_mask = action_rank_mask.reshape(flat_size, action_rank_mask.size(-1))
                for rank in range(top_actions.size(-1)):
                    selected_for_rank = flat_action_rank_mask[action_indices, rank]
                    if not torch.any(selected_for_rank):
                        continue
                    rank_indices = action_indices[selected_for_rank]
                    weight = flat_top_weights[rank_indices, rank].to(hidden.dtype)
                    accum.index_add_(0, rank_indices, output[selected_for_rank] * weight.unsqueeze(-1))
                    key_accum.index_add_(
                        0,
                        rank_indices,
                        key_summary[selected_for_rank].to(key_accum.dtype) * weight.to(key_accum.dtype).unsqueeze(-1),
                    )
                    value_accum.index_add_(
                        0,
                        rank_indices,
                        value_summary[selected_for_rank].to(value_accum.dtype)
                        * weight.to(value_accum.dtype).unsqueeze(-1),
                    )
                    weight_sum.index_add_(0, rank_indices, weight)

            weighted_mask = flat_use_weighted & (weight_sum > 0)
            if torch.any(weighted_mask):
                denom = weight_sum[weighted_mask].clamp_min(1e-9)
                flat_next[weighted_mask] = accum[weighted_mask] / denom.unsqueeze(-1)
                flat_write_key[weighted_mask] = key_accum[weighted_mask] / denom.to(key_accum.dtype).unsqueeze(-1)
                flat_write_value[weighted_mask] = value_accum[weighted_mask] / denom.to(value_accum.dtype).unsqueeze(-1)
                flat_write_valid[weighted_mask] = True

            fallback_mask = use_weighted_fusion & (weight_sum.reshape_as(use_weighted_fusion) <= 0)
            for action in range(len(self.route_blocks)):
                apply_action_mask(action, fallback_mask & (selected == action))

        return next_hidden, write_key, write_value, write_valid

    def _apply_selected_blocks_with_attention_global(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        selected: torch.Tensor,
        attention_global_state: AttentionGlobalKVState,
        route_info: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if selected.dim() == 2:
            next_hidden = hidden.clone()
            write_key, write_value, write_valid = self._empty_attention_global_write(hidden)
            block_position = self._block_position(position)
            for action, block in enumerate(self.route_blocks):
                mask = selected == action
                if torch.any(mask):
                    output, key_summary, value_summary, metrics = block(
                        hidden,
                        block_position,
                        attention_global_state,
                        return_attention_kv=True,
                    )
                    key_write = self._attention_global_write_tensor(key_summary).to(write_key.dtype)
                    value_write = self._attention_global_write_tensor(value_summary).to(write_value.dtype)
                    token_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                    next_hidden = torch.where(mask.unsqueeze(-1), output, next_hidden)
                    write_key = torch.where(token_mask, key_write, write_key)
                    write_value = torch.where(token_mask, value_write, write_value)
                    write_valid = write_valid | mask.unsqueeze(1)
                    self._append_attention_global_metrics(route_info, metrics)
            return next_hidden, write_key, write_value, write_valid
        next_hidden = hidden.clone()
        write_key, write_value, write_valid = self._empty_attention_global_write(hidden)
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            mask = selected == action
            if torch.any(mask):
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                output, key_summary, value_summary, metrics = block(
                    hidden[indices],
                    block_position[indices],
                    attention_global_state.index_select(indices),
                    return_attention_kv=True,
                )
                next_hidden[indices] = output
                write_key[indices] = self._attention_global_write_tensor(key_summary).to(write_key.dtype)
                write_value[indices] = self._attention_global_write_tensor(value_summary).to(write_value.dtype)
                write_valid[indices] = True
                self._append_attention_global_metrics(route_info, metrics)
        return next_hidden, write_key, write_value, write_valid

    def _empty_attention_global_write(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = self._attention_global_cache_dim()
        shape = (hidden.size(0), self._attention_global_cache_heads(), 1, hidden.size(1), dim)
        return (
            hidden.new_zeros(shape),
            hidden.new_zeros(shape),
            torch.zeros(hidden.size(0), 1, hidden.size(1), dtype=torch.bool, device=hidden.device),
        )

    def _pure_factorized_shared_global_write(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.route_blocks:
            raise ValueError("pure_factorized shared global write requires at least one route block.")
        attn = self.route_blocks[0].block.attn
        if attn.global_key_write is None or attn.global_value_write is None:
            raise ValueError("pure_factorized shared global writers are missing.")
        key = attn.global_key_write(hidden).unsqueeze(1).unsqueeze(2)
        value = attn.global_value_write(hidden).unsqueeze(1).unsqueeze(2)
        if valid is None:
            valid = torch.ones(hidden.size(0), 1, hidden.size(1), dtype=torch.bool, device=hidden.device)
        elif valid.dim() == 2:
            valid = valid.unsqueeze(1)
        return key, value, valid.to(device=hidden.device, dtype=torch.bool)

    def _attention_global_cache_heads(self) -> int:
        if self.config.attention_global_kv_mode == "pure_factorized":
            return 1
        return self.config.base.n_heads

    def _attention_global_cache_dim(self) -> int:
        if self.config.attention_global_kv_mode in {"token_compressed", "pure_factorized"}:
            return int(self.config.attention_global_code_dim or (self.config.base.d_model // self.config.base.n_heads))
        return self.config.base.d_model // self.config.base.n_heads

    def _tie_pure_factorized_global_writers(self) -> None:
        if len(self.route_blocks) <= 1:
            return
        source = self.route_blocks[0].block.attn
        for block in self.route_blocks[1:]:
            block.block.attn.tie_pure_factorized_writers_from(source)

    def _attention_global_write_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(2).unsqueeze(3)
        if tensor.dim() == 4:
            return tensor.unsqueeze(2)
        if tensor.dim() == 5:
            return tensor
        raise ValueError("Attention Global KV writes must be rank 3, rank 4, or rank 5 tensors.")

    def _append_attention_global_metrics(self, route_info: dict[str, Any], metrics: dict[str, torch.Tensor]) -> None:
        for key in [
            "attention_global_kv_logit_bias",
            "attention_global_kv_last_token_mass",
            "attention_global_kv_sink_last_token_mass",
            "attention_global_kv_window_last_token_mass",
        ]:
            value = metrics.get(key)
            if value is not None:
                route_info[key].append(value)

    def _next_position(
        self,
        selected: torch.Tensor,
        top_actions: torch.Tensor,
        top_weights: torch.Tensor,
        use_weighted_fusion: torch.Tensor,
    ) -> torch.Tensor:
        position = self.position_table.by_action(selected)
        if torch.any(use_weighted_fusion):
            action_probs = torch.zeros(
                *selected.shape,
                self.config.route_pool_blocks + 1,
                dtype=top_weights.dtype,
                device=selected.device,
            )
            action_probs.scatter_add_(-1, top_actions, top_weights)
            weighted_position = self.position_table.weighted(action_probs)
            position = torch.where(use_weighted_fusion.unsqueeze(-1), weighted_position, position)
        return position

    def _router_position(self, position: torch.Tensor) -> torch.Tensor:
        if self.config.position_to_router:
            return position
        return torch.zeros_like(position)

    def _block_position(self, position: torch.Tensor) -> torch.Tensor:
        if self.config.position_to_blocks:
            return position
        return torch.zeros_like(position)

    def _apply_location_bias(self, logits: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if self.config.location_bias_weight <= 0.0:
            return logits
        return logits - float(self.config.location_bias_weight) * self.position_table.action_distances(position).to(
            logits.dtype
        )

    def _route_logit_noise_std(self, global_step: int, options: Mapping[str, Any]) -> float:
        if not self.training:
            return 0.0
        start = _routing_float(options, "logit_noise_std", default=0.0, minimum=0.0)
        if start <= 0.0:
            return 0.0
        minimum = _routing_float(options, "logit_noise_min_std", default=0.0, minimum=0.0)
        if minimum > start:
            raise ValueError("routing.logit_noise_min_std must be <= routing.logit_noise_std.")
        decay_steps = _routing_int(options, "logit_noise_decay_steps", default=0, minimum=0)
        if decay_steps <= 0:
            return start
        progress = min(1.0, max(0.0, float(global_step) / float(decay_steps)))
        return start + (minimum - start) * progress

    def _apply_route_logit_noise(
        self,
        logits: torch.Tensor,
        global_step: int,
        options: Mapping[str, Any],
    ) -> torch.Tensor:
        std = self._route_logit_noise_std(global_step, options)
        if std <= 0.0:
            return logits
        return logits + torch.randn_like(logits) * std

    def _self_recur_max_consecutive(self, constraints: Mapping[str, Any]) -> int:
        if "self_recur_max_consecutive" in constraints:
            key = "self_recur_max_consecutive"
        elif "self_recur_cap" in constraints:
            key = "self_recur_cap"
        elif "max_consecutive_self_recur" in constraints:
            key = "max_consecutive_self_recur"
        else:
            return 0
        return _constraint_int(constraints, key, default=0, minimum=0)

    def _apply_self_recur_cap(
        self,
        logits: torch.Tensor,
        last_internal_action: torch.Tensor,
        same_internal_run_length: torch.Tensor,
        constraints: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_consecutive = self._self_recur_max_consecutive(constraints)
        if max_consecutive <= 0:
            return logits, torch.zeros_like(last_internal_action, dtype=torch.bool)
        capped = (last_internal_action >= 0) & (same_internal_run_length >= max_consecutive)
        if not torch.any(capped):
            return logits, torch.zeros_like(last_internal_action, dtype=torch.bool)
        adjusted = logits.clone()
        action_mask = F.one_hot(
            last_internal_action.clamp(min=0),
            num_classes=self.config.route_pool_blocks + 1,
        ).to(dtype=torch.bool)
        adjusted = adjusted.masked_fill(capped.unsqueeze(-1) & action_mask, logits.new_tensor(-1.0e4))
        return adjusted, capped

    def _enforce_self_recur_cap_on_selected(
        self,
        selected: torch.Tensor,
        logits: torch.Tensor,
        last_internal_action: torch.Tensor,
        same_internal_run_length: torch.Tensor,
        constraints: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_consecutive = self._self_recur_max_consecutive(constraints)
        if max_consecutive <= 0:
            return selected, torch.zeros_like(selected, dtype=torch.bool)
        capped = (last_internal_action >= 0) & (same_internal_run_length >= max_consecutive)
        invalid = capped & (selected == last_internal_action) & (selected != self.out_action)
        if not torch.any(invalid):
            return selected, torch.zeros_like(selected, dtype=torch.bool)
        replacement = torch.argmax(logits, dim=-1)
        return torch.where(invalid, replacement, selected), invalid

    def _avoid_capped_self_recur_actions(
        self,
        actions: torch.Tensor,
        last_internal_action: torch.Tensor,
        same_internal_run_length: torch.Tensor,
        constraints: Mapping[str, Any],
    ) -> torch.Tensor:
        max_consecutive = self._self_recur_max_consecutive(constraints)
        if max_consecutive <= 0 or self.config.route_pool_blocks <= 1:
            return actions
        capped = (last_internal_action >= 0) & (same_internal_run_length >= max_consecutive)
        invalid = capped & (actions == last_internal_action)
        if not torch.any(invalid):
            return actions
        replacement = (actions + 1) % self.config.route_pool_blocks
        return torch.where(invalid, replacement, actions)

    def _update_self_recur_state(
        self,
        selected: torch.Tensor,
        last_internal_action: torch.Tensor,
        same_internal_run_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        internal = selected != self.out_action
        same_as_last = internal & (selected == last_internal_action)
        next_run_length = torch.where(
            internal,
            torch.where(same_as_last, same_internal_run_length + 1, torch.ones_like(same_internal_run_length)),
            torch.zeros_like(same_internal_run_length),
        )
        next_last = torch.where(internal, selected, torch.full_like(last_internal_action, -1))
        return next_last, next_run_length

    def _apply_route_constraints(
        self,
        logits: torch.Tensor,
        step: int,
        max_steps: int,
        constraints: Mapping[str, Any],
    ) -> torch.Tensor:
        if not constraints:
            return logits
        step_number = step + 1
        adjusted = logits
        min_exit_step = _constraint_int(constraints, "min_exit_step", default=1, minimum=1)
        early_penalty = _constraint_float(constraints, "early_exit_logit_penalty", default=0.0, minimum=0.0)
        if early_penalty > 0.0 and step_number < min_exit_step:
            adjusted = adjusted.clone()
            adjusted[..., self.out_action] = adjusted[..., self.out_action] - early_penalty

        ramp_start = _constraint_int(constraints, "exit_ramp_start", default=max_steps, minimum=1)
        ramp_bias = _constraint_float(constraints, "exit_ramp_logit_bias", default=0.0, minimum=0.0)
        if ramp_bias > 0.0 and step_number >= ramp_start:
            progress = (step_number - ramp_start + 1) / max(1, max_steps - ramp_start + 1)
            adjusted = adjusted.clone()
            adjusted[..., self.out_action] = adjusted[..., self.out_action] + ramp_bias * progress

        final_bias = _constraint_float(constraints, "final_exit_logit_bias", default=0.0, minimum=0.0)
        if final_bias > 0.0 and step_number >= max_steps:
            adjusted = adjusted.clone()
            adjusted[..., self.out_action] = adjusted[..., self.out_action] + final_bias
        return adjusted

    def _force_final_exit(self, step: int, max_steps: int, constraints: Mapping[str, Any]) -> bool:
        return _constraint_bool(constraints, "force_final_exit", default=False) and step + 1 >= max_steps

    def _global_write(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor | None = None,
        *,
        tokenwise: bool = False,
    ) -> torch.Tensor:
        assert self.global_write is not None
        if not isinstance(self.global_write, nn.ModuleList):
            return self.global_write(hidden, tokenwise=tokenwise)
        if tokenwise:
            indices = self._global_token_adapter_indices(actions, hidden.shape[:2], hidden.device)
            codes = hidden.new_empty((hidden.size(0), hidden.size(1), self.config.global_code_dim))
            for index, adapter in enumerate(self.global_write):
                mask = indices == index
                if torch.any(mask):
                    adapter_codes = adapter(hidden, tokenwise=True)
                    codes[mask] = adapter_codes[mask]
            return codes
        indices = self._global_adapter_indices(actions, hidden.size(0), hidden.device)
        codes = hidden.new_empty((hidden.size(0), self.config.global_code_dim))
        for index, adapter in enumerate(self.global_write):
            mask = indices == index
            if torch.any(mask):
                codes[mask] = adapter(hidden[mask], tokenwise=False)
        return codes

    def _global_read(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
        *,
        sink_slots: int,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert self.global_read is not None
        if not isinstance(self.global_read, nn.ModuleList):
            return self.global_read(hidden, codes, sink_slots=sink_slots)
        if codes.dim() == 4:
            indices = self._global_token_adapter_indices(actions, hidden.shape[:2], hidden.device)
            updated = hidden.clone()
            metric_sums: dict[str, torch.Tensor] = {}
            total = max(1, int(indices.numel()))
            for index, adapter in enumerate(self.global_read):
                mask = indices == index
                if not torch.any(mask):
                    continue
                group_hidden, metrics = adapter(hidden, codes, sink_slots=sink_slots)
                updated[mask] = group_hidden[mask]
                weight = torch.tensor(float(mask.sum().item()) / float(total), device=hidden.device, dtype=hidden.dtype)
                for key, value in metrics.items():
                    value = value.to(device=hidden.device, dtype=hidden.dtype)
                    if key not in metric_sums:
                        metric_sums[key] = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
                    metric_sums[key] = metric_sums[key] + value * weight
            return updated, metric_sums
        indices = self._global_adapter_indices(actions, hidden.size(0), hidden.device)
        updated = hidden.clone()
        metric_sums: dict[str, torch.Tensor] = {}
        total = max(1, hidden.size(0))
        for index, adapter in enumerate(self.global_read):
            mask = indices == index
            if not torch.any(mask):
                continue
            group_hidden, metrics = adapter(hidden[mask], codes[mask], sink_slots=sink_slots)
            updated[mask] = group_hidden
            weight = torch.tensor(float(mask.sum().item()) / float(total), device=hidden.device, dtype=hidden.dtype)
            for key, value in metrics.items():
                value = value.to(device=hidden.device, dtype=hidden.dtype)
                if key not in metric_sums:
                    metric_sums[key] = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
                metric_sums[key] = metric_sums[key] + value * weight
        return updated, metric_sums

    def _global_token_adapter_indices(
        self,
        actions: torch.Tensor | None,
        shape: torch.Size | tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size, seq_len = int(shape[0]), int(shape[1])
        if actions is None:
            return torch.full((batch_size, seq_len), self.out_action, dtype=torch.long, device=device)
        indices = actions.to(device=device, dtype=torch.long)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1).expand(-1, seq_len)
        if tuple(indices.shape) != (batch_size, seq_len):
            raise ValueError("Global adapter token action shape must match [batch, seq].")
        return indices.clamp(min=0, max=self.out_action)

    def _global_adapter_indices(
        self,
        actions: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if actions is None:
            return torch.full((batch_size,), self.out_action, dtype=torch.long, device=device)
        indices = actions.to(device=device, dtype=torch.long).reshape(-1)
        if indices.numel() == 1 and batch_size != 1:
            indices = indices.expand(batch_size)
        if indices.numel() != batch_size:
            raise ValueError("Global adapter action count must match batch size.")
        return indices.clamp(min=0, max=self.out_action)

    def model_stats(self) -> dict[str, int | str]:
        return {
            "model_name": self.config.model_name,
            "parameter_count": count_parameters(self),
            "pre_blocks": self.config.pre_blocks,
            "route_pool_blocks": self.config.route_pool_blocks,
            "post_blocks": self.config.post_blocks,
            "block_position_dim": self.config.block_position_dim,
            "route_pool_finegrained": str(self.config.route_pool_finegrained),
            "route_block_ffn_multiplier": str(self.config.route_block_ffn_multiplier),
            "top_k": self.config.top_k,
            "later_top_k": self.config.later_top_k,
            "block_position_mode": self.config.block_position_mode,
            "block_position_injection": self.config.block_position_injection,
            "independent_input_position": str(self.config.independent_input_position),
            "position_to_router": str(self.config.position_to_router),
            "position_to_blocks": str(self.config.position_to_blocks),
            "location_bias_weight": str(self.config.location_bias_weight),
            "route_block_execution": self.config.route_block_execution,
            "global_kv": str(self.config.global_kv),
            "global_code_dim": self.config.global_code_dim,
            "global_sink_slots": self.config.global_sink_slots,
            "global_window_slots": self.config.global_window_slots,
            "global_adapter_scope": self.config.global_adapter_scope,
            "global_head_delta_rank": self.config.global_head_delta_rank,
            "attention_global_kv": str(self.config.attention_global_kv),
            "attention_global_kv_scope": self.config.attention_global_kv_scope,
            "attention_global_kv_mode": self.config.attention_global_kv_mode,
            "attention_global_code_dim": str(self.config.attention_global_code_dim),
            "attention_global_route_execution": self.config.attention_global_route_execution,
            "attention_global_sink_slots": self.config.attention_global_sink_slots,
            "attention_global_window_slots": self.config.attention_global_window_slots,
            "attention_global_tokens_per_write": self.config.attention_global_tokens_per_write,
            "attention_global_logit_bias_init": str(self.config.attention_global_logit_bias_init),
            "parallel_passing": str(self.config.parallel_passing),
            "beam_size": self.config.beam_size,
            "branch_cost": str(self.config.branch_cost),
            "branch_score_decay": str(self.config.branch_score_decay),
            "parallel_exit_policy": self.config.parallel_exit_policy,
        }


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _route_block_execution_value(value: Any) -> str:
    execution = str(value)
    if execution not in {"full_sequence", "sparse", "sparse_varlen", "grouped_dense"}:
        raise ValueError(
            "route_block_execution must be 'full_sequence', 'sparse', 'sparse_varlen', or 'grouped_dense'."
        )
    return execution


def _sparse_varlen_backend(hidden: torch.Tensor) -> str:
    backend = os.environ.get("BRIAN_SPARSE_VARLEN_BACKEND", "auto").strip().lower()
    if backend in {"flex", "flex_attention"}:
        return "flex"
    if backend in {"dense", "dense_selected"}:
        return "dense"
    if backend in {"dense_compiled", "compiled_dense"}:
        return "dense_compiled"
    if backend != "auto":
        raise ValueError("BRIAN_SPARSE_VARLEN_BACKEND must be 'auto', 'dense', 'dense_compiled', or 'flex'.")
    return "dense" if hidden.is_cuda else "flex"


def _optional_float_value(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _float_value(value, name, minimum=minimum, maximum=maximum)


def _optional_int_value(value: Any, name: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _int_value(value, name, minimum=minimum)


def _loss_weights_mapping(loss_weights: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if loss_weights is None:
        return {}
    if not isinstance(loss_weights, Mapping):
        raise ValueError("loss_weights must be a mapping.")
    return loss_weights


def _routing_constraints_mapping(routing_constraints: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if routing_constraints is None:
        return {}
    if not isinstance(routing_constraints, Mapping):
        raise ValueError("routing_constraints must be a mapping.")
    return routing_constraints


def _routing_options_mapping(routing_options: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if routing_options is None:
        return {}
    if not isinstance(routing_options, Mapping):
        raise ValueError("routing_options must be a mapping.")
    return routing_options


def _loss_weight(loss_weights: Mapping[str, Any], key: str) -> float:
    return _float_value(loss_weights.get(key, 0.0), f"loss_weights.{key}", minimum=0.0)


def _zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _coverage_floor_min(routing_constraints: Mapping[str, Any]) -> float:
    return _float_value(
        routing_constraints.get("coverage_floor_min", 0.0),
        "routing.constraints.coverage_floor_min",
        minimum=0.0,
        maximum=1.0,
    )


def _routing_float(
    options: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
) -> float:
    return _float_value(options.get(key, default), f"routing.{key}", minimum=minimum)


def _routing_int(
    options: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"routing.{key} must be an integer, not a boolean.")
    if not isinstance(value, int):
        raise ValueError(f"routing.{key} must be an integer.")
    if value < minimum:
        raise ValueError(f"routing.{key} must be >= {minimum}.")
    return value


def _routing_bool(options: Mapping[str, Any], key: str, *, default: bool) -> bool:
    return _bool_value(options.get(key, default), f"routing.{key}")


def _constraint_int(
    constraints: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = constraints.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"routing.constraints.{key} must be an integer, not a boolean.")
    if not isinstance(value, int):
        raise ValueError(f"routing.constraints.{key} must be an integer.")
    if value < minimum:
        raise ValueError(f"routing.constraints.{key} must be >= {minimum}.")
    return value


def _constraint_float(
    constraints: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
) -> float:
    return _float_value(constraints.get(key, default), f"routing.constraints.{key}", minimum=minimum)


def _constraint_bool(constraints: Mapping[str, Any], key: str, *, default: bool) -> bool:
    return _bool_value(constraints.get(key, default), f"routing.constraints.{key}")


def _content_difficulty_ids(input_ids: torch.Tensor) -> torch.Tensor:
    batch_size = input_ids.size(0)
    if batch_size <= 1:
        return torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
    with torch.no_grad():
        values = input_ids.detach().float()
        mean_id = values.mean(dim=1)
        spread = values.std(dim=1, unbiased=False) if values.size(1) > 1 else torch.zeros_like(mean_id)
        score = mean_id + 0.01 * spread
        order = torch.argsort(score)
        ranks = torch.arange(batch_size, dtype=torch.long, device=input_ids.device)
        difficulty = torch.empty((batch_size,), dtype=torch.long, device=input_ids.device)
        difficulty[order] = torch.clamp((ranks * 3) // batch_size, max=2)
        return difficulty
