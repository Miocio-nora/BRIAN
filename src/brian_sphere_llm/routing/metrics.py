from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def entropy(probs: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    return -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=dim)


def block_load_entropy_from_counts(counts: dict[int, int], num_internal_blocks: int) -> tuple[float, float]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for routing metrics.")
    internal_counts = torch.tensor([counts.get(index, 0) for index in range(num_internal_blocks)], dtype=torch.float32)
    total = internal_counts.sum()
    if total <= 0:
        return 0.0, 0.0
    probs = internal_counts / total
    nonzero = probs[probs > 0]
    value = max(0.0, float((-(nonzero * nonzero.log()).sum()).cpu()))
    max_entropy = float(torch.log(torch.tensor(float(num_internal_blocks))).cpu()) if num_internal_blocks > 1 else 1.0
    normalized = max(0.0, value / max_entropy) if max_entropy > 0 else 0.0
    return value, normalized


def summarize_routes(
    route_info: dict[str, Any],
    num_internal_blocks: int,
    *,
    include_path_counts: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for routing metrics.")
    probs = route_info.get("route_probs")
    actions = route_info.get("selected_actions")
    topk_actions = route_info.get("topk_actions")
    used_weighted_fusion = route_info.get("used_weighted_fusion")
    exit_flags = route_info.get("exit_flags")
    hard_exit_enabled = route_info.get("hard_exit_enabled") is True
    max_route_steps = route_info.get("max_route_steps")
    summary: dict[str, Any] = {}
    if probs:
        all_probs = torch.stack(probs)
        summary["route_entropy"] = float(entropy(all_probs).mean().detach().cpu())
        summary["p_output_mean"] = float(all_probs[..., num_internal_blocks].mean().detach().cpu())
    if actions:
        stacked = torch.stack(actions).detach().cpu()
        flat = stacked.flatten().tolist()
        counts = Counter(int(value) for value in flat)
        summary["top1_block_histogram"] = {str(key): counts.get(key, 0) for key in range(num_internal_blocks + 1)}
        load_entropy, normalized_load_entropy = block_load_entropy_from_counts(counts, num_internal_blocks)
        summary["block_load_entropy"] = load_entropy
        summary["block_load_entropy_normalized"] = normalized_load_entropy
        internal = sum(counts.get(key, 0) for key in range(num_internal_blocks))
        total = max(1, len(flat))
        summary["active_block_evals_per_token"] = internal / total
        summary["average_route_steps"] = len(actions)
        paths = [tuple(int(value) for value in stacked[:, sample_index].tolist()) for sample_index in range(stacked.size(1))]
        summary["route_path_count"] = len(set(paths))
        summary["route_path_diversity"] = len(set(paths)) / max(1, len(paths))
        summary["route_path_examples"] = _path_examples(stacked, max_examples=8)
        if include_path_counts:
            path_counts = _path_counts(stacked, out_action=num_internal_blocks)
            summary["route_path_counts"] = path_counts
            summary["route_transition_counts"] = _transition_counts(path_counts)
        advances = 0
        skips = 0
        recurs = 0
        for first, second in zip(stacked[:-1], stacked[1:]):
            diff = second - first
            advances += int((diff == 1).sum())
            skips += int((diff > 1).sum())
            recurs += int((diff == 0).sum())
        denom = max(1, advances + skips + recurs)
        summary["advance_ratio"] = advances / denom
        summary["skip_ratio"] = skips / denom
        summary["recur_ratio"] = recurs / denom
    if topk_actions:
        flat_topk = torch.cat([actions.detach().cpu().reshape(-1) for actions in topk_actions]).tolist()
        counts = Counter(int(value) for value in flat_topk)
        summary["topk_block_histogram"] = {str(key): counts.get(key, 0) for key in range(num_internal_blocks + 1)}
    if used_weighted_fusion:
        stacked_weighted = torch.stack(used_weighted_fusion).float()
        summary["weighted_fusion_ratio"] = float(stacked_weighted.mean().detach().cpu())
    if exit_flags:
        exits = torch.stack(exit_flags).detach().cpu()
        summary["exit_step_distribution"] = [int(value) for value in exits.sum(dim=1).tolist()]
        first_exit_steps: list[int] = []
        for sample_index in range(exits.size(1)):
            sample_flags = exits[:, sample_index]
            hit = torch.nonzero(sample_flags, as_tuple=False)
            first_exit_steps.append(int(hit[0].item() + 1) if hit.numel() else 0)
        summary["first_exit_step_histogram"] = {
            str(step): first_exit_steps.count(step) for step in sorted(set(first_exit_steps))
        }
        if hard_exit_enabled:
            forced_count = first_exit_steps.count(0)
            summary["forced_max_step_exit_count"] = forced_count
            summary["forced_max_step_exit_fraction"] = forced_count / max(1, len(first_exit_steps))
            if isinstance(max_route_steps, int) and not isinstance(max_route_steps, bool):
                summary["max_route_steps"] = max_route_steps
    if "position_norms" in route_info and route_info["position_norms"]:
        norms = torch.stack(route_info["position_norms"])
        summary["position_norm_trajectory"] = [float(value) for value in norms.detach().cpu().flatten().tolist()]
        summary["position_norm_mean"] = float(norms.mean().detach().cpu())
    if "location_distance" in route_info and route_info["location_distance"]:
        distances = torch.stack(route_info["location_distance"])
        summary["location_distance_trajectory"] = [float(value) for value in distances.detach().cpu().flatten().tolist()]
        summary["location_distance_mean"] = float(distances.mean().detach().cpu())
    noise_std = route_info.get("route_logit_noise_std")
    if noise_std is not None:
        if isinstance(noise_std, (int, float)):
            summary["route_logit_noise_std"] = float(noise_std)
        elif isinstance(noise_std, torch.Tensor) and noise_std.numel() == 1:
            summary["route_logit_noise_std"] = float(noise_std.detach().cpu())
    if "global_attention_mass" in route_info and route_info["global_attention_mass"]:
        mass = torch.stack(route_info["global_attention_mass"])
        summary["global_attention_mass"] = float(mass.mean().detach().cpu())
    if "global_sink_attention_mass" in route_info and route_info["global_sink_attention_mass"]:
        mass = torch.stack(route_info["global_sink_attention_mass"])
        summary["global_sink_attention_mass"] = float(mass.mean().detach().cpu())
    if "global_window_attention_mass" in route_info and route_info["global_window_attention_mass"]:
        mass = torch.stack(route_info["global_window_attention_mass"])
        summary["global_window_attention_mass"] = float(mass.mean().detach().cpu())
    if "global_read_gate" in route_info and route_info["global_read_gate"]:
        gate = torch.stack(route_info["global_read_gate"])
        gate_mean = min(1.0, max(0.0, float(gate.mean().detach().cpu())))
        local_mean = max(0.0, 1.0 - gate_mean)
        summary["global_read_gate_mean"] = gate_mean
        summary["local_read_fraction_mean"] = local_mean
        summary["global_to_local_read_ratio"] = gate_mean / max(1e-9, local_mean)
        summary["local_to_global_read_ratio"] = local_mean / max(1e-9, gate_mean)
    if "global_cache_slots" in route_info and route_info["global_cache_slots"]:
        slots = torch.stack(route_info["global_cache_slots"])
        summary["global_cache_slots_mean"] = float(slots.mean().detach().cpu())
    if "attention_global_kv_slots" in route_info and route_info["attention_global_kv_slots"]:
        slots = torch.stack(route_info["attention_global_kv_slots"])
        summary["attention_global_kv_slots_mean"] = float(slots.mean().detach().cpu())
        summary["attention_global_kv_slots_max"] = float(slots.max().detach().cpu())
    if "attention_global_kv_write_count" in route_info and route_info["attention_global_kv_write_count"]:
        counts = torch.stack(route_info["attention_global_kv_write_count"])
        summary["attention_global_kv_write_count_mean"] = float(counts.mean().detach().cpu())
    for source_key, summary_key in [
        ("attention_global_kv_logit_bias", "attention_global_kv_logit_bias_mean"),
        ("attention_global_kv_last_token_mass", "attention_global_kv_last_token_mass"),
        ("attention_global_kv_sink_last_token_mass", "attention_global_kv_sink_last_token_mass"),
        ("attention_global_kv_window_last_token_mass", "attention_global_kv_window_last_token_mass"),
    ]:
        if source_key in route_info and route_info[source_key]:
            values = torch.stack(route_info[source_key])
            summary[summary_key] = float(values.mean().detach().cpu())
    if "parallel_branch_count" in route_info and route_info["parallel_branch_count"]:
        counts = torch.stack(route_info["parallel_branch_count"])
        summary["parallel_branch_count_mean"] = float(counts.mean().detach().cpu())
    if "parallel_score_margin" in route_info and route_info["parallel_score_margin"]:
        margins = torch.stack(route_info["parallel_score_margin"])
        summary["parallel_score_margin_mean"] = float(margins.mean().detach().cpu())
    if "parallel_delta_cache_slots" in route_info and route_info["parallel_delta_cache_slots"]:
        slots = torch.stack(route_info["parallel_delta_cache_slots"])
        summary["parallel_delta_cache_slots_mean"] = float(slots.mean().detach().cpu())
        summary["parallel_delta_cache_slots_max"] = float(slots.max().detach().cpu())
    if "route_targets" in route_info and route_info["route_targets"] and actions:
        correct = []
        for selected, target in zip(actions, route_info["route_targets"]):
            correct.append((selected == target).float().mean())
        summary["route_imitation_accuracy"] = float(torch.stack(correct).mean().detach().cpu())
    return summary


def _path_examples(stacked_actions: "torch.Tensor", *, max_examples: int) -> list[dict[str, Any]]:
    examples = []
    for sample_index in range(min(max_examples, stacked_actions.size(1))):
        examples.append(
            {
                "sample_index": sample_index,
                "actions": [int(value) for value in stacked_actions[:, sample_index].tolist()],
            }
        )
    return examples


def _path_counts(stacked_actions: "torch.Tensor", *, out_action: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for sample_index in range(stacked_actions.size(1)):
        path = _truncate_at_out(
            [int(value) for value in stacked_actions[:, sample_index].tolist()],
            out_action=out_action,
        )
        counts[tuple(path)] += 1
    return [
        {"actions": list(path), "count": int(count)}
        for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _transition_counts(path_counts: list[dict[str, Any]]) -> list[dict[str, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for item in path_counts:
        actions = [int(value) for value in item.get("actions", [])]
        count = int(item.get("count", 0))
        for source, target in zip(actions[:-1], actions[1:]):
            counts[(source, target)] += count
    return [
        {"source": int(source), "target": int(target), "count": int(count)}
        for (source, target), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _truncate_at_out(actions: list[int], *, out_action: int) -> list[int]:
    truncated: list[int] = []
    for action in actions:
        truncated.append(action)
        if action == out_action:
            break
    return truncated
