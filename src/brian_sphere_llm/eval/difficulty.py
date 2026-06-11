from __future__ import annotations

import math
from typing import Any

from brian_sphere_llm.eval.stage_gate_report import pearson_correlation


def difficulty_step_correlation(baseline_losses: list[float], route_steps: list[float]) -> float | None:
    """Critical diagnostic: corr(baseline_sample_loss, route_steps)."""
    return pearson_correlation(baseline_losses, route_steps)


def summarize_difficulty_samples(samples: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    valid_samples = []
    for sample in samples:
        baseline_loss = _num(sample.get("baseline_cross_entropy"))
        routed_loss = _num(sample.get("routed_cross_entropy"))
        route_steps = _num(sample.get("route_steps"))
        if baseline_loss is not None and routed_loss is not None and route_steps is not None:
            valid_samples.append((baseline_loss, routed_loss, route_steps))
    baseline_losses = [baseline_loss for baseline_loss, _, _ in valid_samples]
    routed_losses = [routed_loss for _, routed_loss, _ in valid_samples]
    route_steps = [steps for _, _, steps in valid_samples]
    count = len(valid_samples)
    if count == 0:
        return {
            "sample_count": 0,
            "mean_baseline_cross_entropy": None,
            "mean_routed_cross_entropy": None,
            "mean_route_steps": None,
            "mean_loss_delta": None,
            "difficulty_step_correlation": None,
        }
    mean_baseline = sum(baseline_losses) / count
    mean_routed = sum(routed_losses) / count
    return {
        "sample_count": count,
        "mean_baseline_cross_entropy": mean_baseline,
        "mean_routed_cross_entropy": mean_routed,
        "mean_route_steps": sum(route_steps) / count,
        "mean_loss_delta": mean_routed - mean_baseline,
        "difficulty_step_correlation": difficulty_step_correlation(baseline_losses, route_steps),
    }


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
