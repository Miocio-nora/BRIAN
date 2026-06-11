from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brian_sphere_llm.losses.balance_loss import block_balance_loss
from brian_sphere_llm.losses.cost_loss import route_cost_loss
from brian_sphere_llm.losses.location_loss import location_loss
from brian_sphere_llm.losses.route_loss import route_imitation_loss
from brian_sphere_llm.memory import CanonicalGlobalCache, GlobalReadAdapter, GlobalWriteAdapter
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.exit_block import ExitBlock
from brian_sphere_llm.model.llama_backbone import (
    RMSNorm,
    TransformerBlock,
    build_causal_lm_loss,
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
    top_k: int = 1
    later_top_k: int = 2
    hard_exit: bool = False
    global_kv: bool = False
    global_code_dim: int = 64
    global_sink_slots: int = 4
    global_window_slots: int = 32
    parallel_passing: bool = False
    beam_size: int = 2
    branch_cost: float = 0.01
    block_position_mode: str = "open_arc"
    position_to_router: bool = True
    position_to_blocks: bool = True

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
        global_kv_value = data.get("global_kv", False)
        global_kv = global_kv_value is True or str(global_kv_value).lower() in {"true", "on", "enabled"}
        parallel_value = data.get("parallel_passing", False)
        parallel_passing = parallel_value is True or str(parallel_value).lower() in {"true", "on", "enabled"}
        return cls(
            base=BaselineConfig.from_dict(base_data),
            pre_blocks=int(data["pre_blocks"]),
            route_pool_blocks=int(data["route_pool_blocks"]),
            post_blocks=int(data["post_blocks"]),
            block_position_dim=int(data["block_position_dim"]),
            max_route_steps=int(data["max_route_steps"]),
            top_k=int(data.get("top_k", 1)),
            later_top_k=int(data.get("later_top_k", 2)),
            hard_exit=bool(data.get("hard_exit", False)),
            global_kv=global_kv,
            global_code_dim=int(data.get("global_code_dim", data.get("block_position_dim", 64))),
            global_sink_slots=int(data.get("global_sink_slots", 4)),
            global_window_slots=int(data.get("global_window_slots", 32)),
            parallel_passing=parallel_passing,
            beam_size=int(data.get("beam_size", 2)),
            branch_cost=float(data.get("branch_cost", 0.01)),
            block_position_mode=str(data.get("block_position_mode", "open_arc")),
            position_to_router=_as_bool(data.get("position_to_router", True)),
            position_to_blocks=_as_bool(data.get("position_to_blocks", True)),
        )


class BrianRouteCore(ModuleBase):
    def __init__(self, config: BrianRouteConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        if config.pre_blocks + config.route_pool_blocks + config.post_blocks != config.base.layers:
            raise ValueError("pre + route_pool + post must equal base layer count")
        backbone = config.base.backbone()
        self.token_embedding = nn.Embedding(config.base.vocab_size, config.base.d_model)
        self.pre_blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.pre_blocks)])
        self.route_blocks = nn.ModuleList(
            [RouteBlock(backbone, config.block_position_dim) for _ in range(config.route_pool_blocks)]
        )
        self.exit_block = ExitBlock(config.base.d_model, config.block_position_dim)
        self.post_blocks = nn.ModuleList([TransformerBlock(backbone) for _ in range(config.post_blocks)])
        self.position_table = BlockPositionTable(
            config.route_pool_blocks,
            config.block_position_dim,
            mode=config.block_position_mode,
        )
        self.router = LatentRouter(
            config.base.d_model,
            config.block_position_dim,
            num_actions=config.route_pool_blocks + 1,
        )
        if config.global_kv:
            self.global_cache = CanonicalGlobalCache(config.global_sink_slots, config.global_window_slots)
            self.global_write = GlobalWriteAdapter(config.base.d_model, config.global_code_dim)
            self.global_read = GlobalReadAdapter(config.base.d_model, config.global_code_dim)
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
        loss_weights: dict[str, float] | None = None,
        hard_exit: bool | None = None,
        router_probability: float | None = None,
        global_step: int = 0,
    ) -> dict[str, Any]:
        loss_weights = loss_weights or {}
        hard_exit = self.config.hard_exit if hard_exit is None else hard_exit
        batch_size = input_ids.size(0)
        hidden = self.token_embedding(input_ids)
        for block in self.pre_blocks:
            hidden = block(hidden)

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
        }
        route_targets = self._targets_for_mode(route_mode, pseudo_policy, batch_size, input_ids.device)
        max_steps = len(route_targets) if route_mode == "fixed" else self.config.max_route_steps
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
            global_state = self.global_cache.write(global_state, self.global_write(hidden))

        if route_mode == "parallel" or self.config.parallel_passing:
            hidden, position, route_info = self._run_parallel_route(hidden, position, route_info, hard_exit, global_state)
            global_state = None

        for step in range(max_steps if route_mode != "parallel" and not self.config.parallel_passing else 0):
            if self.config.global_kv and global_state is not None:
                assert self.global_read is not None
                hidden, global_metrics = self.global_read(
                    hidden,
                    global_state.codes,
                    sink_slots=self.config.global_sink_slots,
                )
                route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
                route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
                route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
                route_info["global_read_gate"].append(global_metrics["global_read_gate"])
                route_info["global_cache_slots"].append(
                    torch.tensor(float(global_state.slots), device=input_ids.device, dtype=hidden.dtype)
                )
            logits = self.router(hidden, self._router_position(position))
            probs = F.softmax(logits, dim=-1)
            top_actions, top_weights = self._topk_actions(probs)
            if step < len(route_targets):
                target_action = route_targets[step]
            else:
                target_action = torch.full((batch_size,), self.out_action, dtype=torch.long, device=input_ids.device)

            use_weighted_fusion = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            if route_mode in {"fixed", "pseudo"}:
                selected = target_action
            elif route_mode == "scheduled":
                selected, use_router = self._scheduled_select(logits, target_action, global_step, router_probability)
                use_weighted_fusion = use_router & (self.config.top_k > 1)
            elif route_mode == "free":
                selected = torch.argmax(logits, dim=-1)
                use_weighted_fusion = torch.full((batch_size,), self.config.top_k > 1, dtype=torch.bool, device=input_ids.device)
            else:
                raise ValueError(f"Unknown route_mode: {route_mode}")

            selected = torch.where(exited, torch.full_like(selected, self.out_action), selected)
            use_weighted_fusion = use_weighted_fusion & ~exited & (selected != self.out_action)
            exit_now = selected == self.out_action
            hidden = self._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
            if hard_exit:
                exited = exited | exit_now
            position = self._next_position(selected, top_actions, top_weights, use_weighted_fusion)
            if self.config.global_kv and global_state is not None:
                assert self.global_write is not None and self.global_cache is not None
                global_state = self.global_cache.write(global_state, self.global_write(hidden))

            route_info["route_logits"].append(logits)
            route_info["route_probs"].append(probs)
            route_info["selected_actions"].append(selected)
            route_info["topk_actions"].append(top_actions)
            route_info["topk_weights"].append(top_weights)
            route_info["used_weighted_fusion"].append(use_weighted_fusion)
            route_info["exit_flags"].append(exit_now)
            route_info["route_targets"].append(target_action)
            route_info["position_norms"].append(position.norm(dim=-1).mean())
            route_info["location_distance"].append(self.position_table.location_distance(position, probs))
            if hard_exit and torch.all(exited):
                break

        hidden = self.exit_block(hidden, self._block_position(position))
        if self.config.global_kv and global_state is not None:
            assert self.global_read is not None
            hidden, global_metrics = self.global_read(
                hidden,
                global_state.codes,
                sink_slots=self.config.global_sink_slots,
            )
            route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
            route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
            route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
            route_info["global_read_gate"].append(global_metrics["global_read_gate"])
            route_info["global_cache_slots"].append(
                torch.tensor(float(global_state.slots), device=input_ids.device, dtype=hidden.dtype)
            )
        for block in self.post_blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))

        output: dict[str, Any] = {
            "logits": logits,
            "route_info": route_info,
            "routing_summary": summarize_routes(route_info, self.config.route_pool_blocks),
        }
        if targets is not None:
            lm = build_causal_lm_loss(logits, targets)
            route = route_imitation_loss(route_info["route_logits"], route_info["route_targets"]).to(lm.device)
            balance = block_balance_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            cost = route_cost_loss(route_info["route_probs"], self.config.route_pool_blocks).to(lm.device)
            loc = location_loss(route_info["location_distance"]).to(lm.device)
            total = (
                lm
                + float(loss_weights.get("route", 0.0)) * route
                + float(loss_weights.get("balance", 0.0)) * balance
                + float(loss_weights.get("cost", 0.0)) * cost
                + float(loss_weights.get("location", 0.0)) * loc
            )
            output["loss"] = total
            output["loss_components"] = {
                "lm_loss": lm.detach(),
                "route_loss": route.detach(),
                "balance_loss": balance.detach(),
                "cost_loss": cost.detach(),
                "location_loss": loc.detach(),
            }
        return output

    def _targets_for_mode(self, route_mode: str, pseudo_policy: str, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        if route_mode in {"fixed", "pseudo", "scheduled"}:
            return actions_for_policy(
                pseudo_policy,
                num_internal_blocks=self.config.route_pool_blocks,
                max_route_steps=self.config.max_route_steps,
                batch_size=batch_size,
                device=device,
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
        if self.config.global_kv:
            branch_delta_codes = torch.empty(
                batch_size,
                1,
                0,
                self.config.global_code_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        beam_size = max(1, self.config.beam_size)
        top_k = max(1, min(self.config.top_k, self.config.route_pool_blocks + 1))

        for _step in range(self.config.max_route_steps):
            current_beam = branch_hidden.size(1)
            flat_hidden = branch_hidden.reshape(batch_size * current_beam, *branch_hidden.shape[2:])
            flat_position = branch_position.reshape(batch_size * current_beam, branch_position.size(-1))
            if self.config.global_kv and global_state is not None:
                assert self.global_read is not None
                assert branch_delta_codes is not None
                base_codes = global_state.codes.unsqueeze(1).expand(-1, current_beam, -1, -1)
                combined_codes = torch.cat([base_codes, branch_delta_codes], dim=2)
                flat_codes = combined_codes.reshape(batch_size * current_beam, combined_codes.size(2), combined_codes.size(3))
                flat_hidden, global_metrics = self.global_read(
                    flat_hidden,
                    flat_codes,
                    sink_slots=self.config.global_sink_slots,
                )
                route_info["global_attention_mass"].append(global_metrics["global_attention_mass"])
                route_info["global_sink_attention_mass"].append(global_metrics["global_sink_attention_mass"])
                route_info["global_window_attention_mass"].append(global_metrics["global_window_attention_mass"])
                route_info["global_read_gate"].append(global_metrics["global_read_gate"])
                route_info["global_cache_slots"].append(
                    torch.tensor(float(flat_codes.size(1)), device=hidden.device, dtype=hidden.dtype)
                )
            logits = self.router(flat_hidden, self._router_position(flat_position))
            probs = F.softmax(logits, dim=-1).view(batch_size, current_beam, -1)
            log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_actions = log_probs.topk(top_k, dim=-1)
            top_actions = top_actions.view(batch_size, current_beam, top_k)
            top_log_probs = top_log_probs.view(batch_size, current_beam, top_k)

            expanded_actions = top_actions
            expanded_log_probs = top_log_probs
            if torch.any(branch_exited):
                out = torch.full_like(expanded_actions, self.out_action)
                zeros = torch.zeros_like(expanded_log_probs)
                expanded_actions = torch.where(branch_exited.unsqueeze(-1), out, expanded_actions)
                expanded_log_probs = torch.where(branch_exited.unsqueeze(-1), zeros, expanded_log_probs)

            internal = expanded_actions != self.out_action
            candidate_scores = branch_scores.unsqueeze(-1) + expanded_log_probs - self.config.branch_cost * internal.float()
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
                delta_write = self.global_write(routed).reshape(batch_size, branch_count, self.config.global_code_dim)
                branch_delta_codes = torch.cat([next_delta_codes, delta_write.unsqueeze(2)], dim=2)
                if self.config.global_window_slots:
                    branch_delta_codes = branch_delta_codes[:, :, -self.config.global_window_slots :, :]

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

    def _scheduled_select(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        global_step: int,
        router_probability: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_choice = torch.argmax(logits, dim=-1)
        probability = router_probability
        if probability is None:
            probability = min(1.0, max(0.0, global_step / max(1, self.config.max_route_steps * 100)))
        use_router = torch.rand_like(target.float()) < probability
        return torch.where(use_router, router_choice, target), use_router

    def _apply_selected_blocks(self, hidden: torch.Tensor, position: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        next_hidden = hidden.clone()
        block_position = self._block_position(position)
        for action, block in enumerate(self.route_blocks):
            mask = selected == action
            if torch.any(mask):
                next_hidden[mask] = block(hidden[mask], block_position[mask])
        return next_hidden

    def _topk_actions(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = max(1, min(self.config.top_k, self.config.route_pool_blocks + 1))
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
                next_hidden[top1_mask] = block(hidden[top1_mask], block_position[top1_mask])

        if torch.any(use_weighted_fusion):
            accum = torch.zeros_like(hidden)
            weight_sum = torch.zeros(hidden.size(0), dtype=hidden.dtype, device=hidden.device)
            for action, block in enumerate(self.route_blocks):
                for rank in range(top_actions.size(1)):
                    mask = use_weighted_fusion & (top_actions[:, rank] == action)
                    if not torch.any(mask):
                        continue
                    action_output = block(hidden[mask], block_position[mask])
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

    def model_stats(self) -> dict[str, int | str]:
        return {
            "model_name": "brian_route_core",
            "parameter_count": count_parameters(self),
            "pre_blocks": self.config.pre_blocks,
            "route_pool_blocks": self.config.route_pool_blocks,
            "post_blocks": self.config.post_blocks,
            "block_position_dim": self.config.block_position_dim,
            "top_k": self.config.top_k,
            "block_position_mode": self.config.block_position_mode,
            "position_to_router": str(self.config.position_to_router),
            "position_to_blocks": str(self.config.position_to_blocks),
            "global_kv": str(self.config.global_kv),
            "global_code_dim": self.config.global_code_dim,
            "global_sink_slots": self.config.global_sink_slots,
            "global_window_slots": self.config.global_window_slots,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)
