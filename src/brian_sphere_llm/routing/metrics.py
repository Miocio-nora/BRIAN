from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def entropy(probs: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    return -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=dim)


def summarize_routes(route_info: dict[str, Any], num_internal_blocks: int) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for routing metrics.")
    probs = route_info.get("route_probs")
    actions = route_info.get("selected_actions")
    exit_flags = route_info.get("exit_flags")
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
        internal = sum(counts.get(key, 0) for key in range(num_internal_blocks))
        total = max(1, len(flat))
        summary["active_block_evals_per_token"] = internal / total
        summary["average_route_steps"] = len(actions)
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
    if exit_flags:
        exits = torch.stack(exit_flags).detach().cpu()
        summary["exit_step_distribution"] = [int(value) for value in exits.sum(dim=1).tolist()]
    if "position_norms" in route_info and route_info["position_norms"]:
        norms = torch.stack(route_info["position_norms"])
        summary["position_norm_mean"] = float(norms.mean().detach().cpu())
    if "location_distance" in route_info and route_info["location_distance"]:
        distances = torch.stack(route_info["location_distance"])
        summary["location_distance_mean"] = float(distances.mean().detach().cpu())
    if "route_targets" in route_info and route_info["route_targets"] and actions:
        correct = []
        for selected, target in zip(actions, route_info["route_targets"]):
            correct.append((selected == target).float().mean())
        summary["route_imitation_accuracy"] = float(torch.stack(correct).mean().detach().cpu())
    return summary
