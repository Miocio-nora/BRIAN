from __future__ import annotations

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
from brian_sphere_llm.memory import CanonicalGlobalCache, GlobalReadAdapter, GlobalWriteAdapter
from brian_sphere_llm.model.baseline import BaselineConfig, _float_value, _int_value
from brian_sphere_llm.model.exit_block import ExitBlock
from brian_sphere_llm.model.llama_backbone import (
    RMSNorm,
    TransformerBlock,
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
        )


class BrianRouteCore(ModuleBase):
    def __init__(self, config: BrianRouteConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        if config.route_pool_finegrained:
            if config.pre_blocks + config.post_blocks > config.base.layers:
                raise ValueError("pre + post must be <= base layer count for fine-grained route pools")
        elif config.pre_blocks + config.route_pool_blocks + config.post_blocks != config.base.layers:
            raise ValueError("pre + route_pool + post must equal base layer count")
        backbone = config.base.backbone()
        route_backbone = backbone
        if config.route_block_ffn_multiplier is not None:
            route_backbone = replace(backbone, ffn_multiplier=config.route_block_ffn_multiplier)
        self.token_embedding = nn.Embedding(config.base.vocab_size, config.base.d_model)
        self.pre_blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.pre_blocks)])
        self.route_blocks = nn.ModuleList(
            [
                RouteBlock(route_backbone, config.block_position_dim, config.block_position_injection)
                for _ in range(config.route_pool_blocks)
            ]
        )
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
    ) -> dict[str, Any]:
        loss_weights = _loss_weights_mapping(loss_weights)
        routing_constraints = _routing_constraints_mapping(routing_constraints)
        routing_options = _routing_options_mapping(routing_options)
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        batch_size = input_ids.size(0)
        hidden = self.token_embedding(input_ids)
        for block in self.pre_blocks:
            hidden = checkpoint_if_enabled(self, block, hidden)

        position = self.position_table.initial(batch_size, input_ids.device)
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
            "parallel_delta_cache_slots": [],
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
        exited = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        global_state = None
        if self.config.global_kv:
            assert self.global_cache is not None and self.global_write is not None
            global_state = self.global_cache.empty(
                batch_size=batch_size,
                code_dim=self.config.global_code_dim,
                device=input_ids.device,
                dtype=hidden.dtype,
            )
            global_state = self.global_cache.write(global_state, self._global_write(hidden))
        last_global_actions = None

        if route_mode == "parallel" or self.config.parallel_passing:
            hidden, position, route_info = self._run_parallel_route(hidden, position, route_info, hard_exit, global_state)
            global_state = None

        for step in range(max_steps if route_mode != "parallel" and not self.config.parallel_passing else 0):
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
            logits = self._apply_location_bias(self.router(hidden, self._router_position(position)), position)
            logits = self._apply_route_logit_noise(logits, global_step, routing_options)
            logits = self._apply_route_constraints(logits, step, max_steps, routing_constraints)
            probs = F.softmax(logits, dim=-1)
            effective_top_k = self._top_k_for_step(step)
            top_actions, top_weights = self._topk_actions(probs, effective_top_k)
            has_route_target = route_mode in {"fixed", "pseudo", "scheduled"} and step < len(route_targets)
            if has_route_target:
                target_action = route_targets[step]
            else:
                target_action = torch.full((batch_size,), self.out_action, dtype=torch.long, device=input_ids.device)

            use_weighted_fusion = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
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
            elif route_mode == "free":
                selected = self._router_action(logits, routing_options)
                use_weighted_fusion = torch.full(
                    (batch_size,),
                    effective_top_k > 1,
                    dtype=torch.bool,
                    device=input_ids.device,
                )
            else:
                raise ValueError(f"Unknown route_mode: {route_mode}")

            if self._force_final_exit(step, max_steps, routing_constraints):
                selected = torch.full_like(selected, self.out_action)
            selected = torch.where(exited, torch.full_like(selected, self.out_action), selected)
            use_weighted_fusion = use_weighted_fusion & ~exited & (selected != self.out_action)
            exit_now = selected == self.out_action
            hidden = self._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
            if hard_exit:
                exited = exited | exit_now
            position = self._next_position(selected, top_actions, top_weights, use_weighted_fusion)
            if self.config.global_kv and global_state is not None:
                assert self.global_write is not None and self.global_cache is not None
                global_state = self.global_cache.write(global_state, self._global_write(hidden, selected))
                last_global_actions = selected

            route_info["route_logits"].append(logits)
            route_info["route_probs"].append(probs)
            route_info["selected_actions"].append(selected)
            route_info["topk_actions"].append(top_actions)
            route_info["topk_weights"].append(top_weights)
            route_info["used_weighted_fusion"].append(use_weighted_fusion)
            route_info["exit_flags"].append(exit_now)
            if has_route_target:
                route_info["route_targets"].append(target_action)
            route_info["position_norms"].append(position.norm(dim=-1).mean())
            route_info["location_distance"].append(self.position_table.location_distance(position, probs))
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
            "routing_summary": summarize_routes(
                route_info,
                self.config.route_pool_blocks,
                include_path_counts=log_path_counts,
            ),
        }
        if targets is not None:
            lm = build_causal_lm_loss(logits, targets)
            route = route_imitation_loss(route_info["route_logits"], route_info["route_targets"]).to(lm.device)
            balance = block_balance_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            cost = route_cost_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            loc = location_loss(route_info["location_distance"]).to(lm.device)
            selected_balance = selected_block_balance_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
            ).to(lm.device)
            coverage_floor = block_coverage_floor_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
                floor=_coverage_floor_min(routing_constraints),
            ).to(lm.device)
            transition_diversity = transition_diversity_loss(
                route_info["route_probs"],
                route_info["selected_actions"],
                self.config.route_pool_blocks,
            ).to(lm.device)
            exit_boundary = exit_boundary_loss(
                route_info["route_probs"],
                self.config.route_pool_blocks,
                {**routing_constraints, "max_route_steps": max_steps},
            ).to(lm.device)
            input_anchor = self.position_table.input_anchor_loss().to(lm.device)
            route_weight = _loss_weight(loss_weights, "route")
            balance_weight = _loss_weight(loss_weights, "balance")
            cost_weight = _loss_weight(loss_weights, "cost")
            location_weight = _loss_weight(loss_weights, "location")
            selected_balance_weight = _loss_weight(loss_weights, "selected_balance")
            coverage_floor_weight = _loss_weight(loss_weights, "coverage_floor")
            transition_diversity_weight = _loss_weight(loss_weights, "transition_diversity")
            exit_boundary_weight = _loss_weight(loss_weights, "exit_boundary")
            input_anchor_weight = _loss_weight(loss_weights, "input_anchor")
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
                delta_write = self._global_write(routed, apply_actions).reshape(
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
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _apply_selected_blocks(self, hidden: torch.Tensor, position: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            mask = selected == action
            if torch.any(mask):
                next_hidden[mask] = checkpoint_if_enabled(self, block, hidden[mask], block_position[mask])
        return next_hidden

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

    def _apply_routed_blocks(
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
                selected.size(0),
                self.config.route_pool_blocks + 1,
                dtype=top_weights.dtype,
                device=selected.device,
            )
            action_probs.scatter_add_(1, top_actions, top_weights)
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
            adjusted[:, self.out_action] = adjusted[:, self.out_action] - early_penalty

        ramp_start = _constraint_int(constraints, "exit_ramp_start", default=max_steps, minimum=1)
        ramp_bias = _constraint_float(constraints, "exit_ramp_logit_bias", default=0.0, minimum=0.0)
        if ramp_bias > 0.0 and step_number >= ramp_start:
            progress = (step_number - ramp_start + 1) / max(1, max_steps - ramp_start + 1)
            adjusted = adjusted.clone()
            adjusted[:, self.out_action] = adjusted[:, self.out_action] + ramp_bias * progress

        final_bias = _constraint_float(constraints, "final_exit_logit_bias", default=0.0, minimum=0.0)
        if final_bias > 0.0 and step_number >= max_steps:
            adjusted = adjusted.clone()
            adjusted[:, self.out_action] = adjusted[:, self.out_action] + final_bias
        return adjusted

    def _force_final_exit(self, step: int, max_steps: int, constraints: Mapping[str, Any]) -> bool:
        return _constraint_bool(constraints, "force_final_exit", default=False) and step + 1 >= max_steps

    def _global_write(self, hidden: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        assert self.global_write is not None
        if not isinstance(self.global_write, nn.ModuleList):
            return self.global_write(hidden)
        indices = self._global_adapter_indices(actions, hidden.size(0), hidden.device)
        codes = hidden.new_empty((hidden.size(0), self.config.global_code_dim))
        for index, adapter in enumerate(self.global_write):
            mask = indices == index
            if torch.any(mask):
                codes[mask] = adapter(hidden[mask])
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
            "global_kv": str(self.config.global_kv),
            "global_code_dim": self.config.global_code_dim,
            "global_sink_slots": self.config.global_sink_slots,
            "global_window_slots": self.config.global_window_slots,
            "global_adapter_scope": self.config.global_adapter_scope,
            "global_head_delta_rank": self.config.global_head_delta_rank,
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
