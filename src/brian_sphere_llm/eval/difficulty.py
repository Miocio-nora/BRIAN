from __future__ import annotations

from brian_sphere_llm.eval.stage_gate_report import pearson_correlation


def difficulty_step_correlation(baseline_losses: list[float], route_steps: list[float]) -> float | None:
    """Critical diagnostic: corr(baseline_sample_loss, route_steps)."""
    return pearson_correlation(baseline_losses, route_steps)


def summarize_difficulty_samples(samples: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    baseline_losses = [float(sample["baseline_cross_entropy"]) for sample in samples]
    routed_losses = [float(sample["routed_cross_entropy"]) for sample in samples]
    route_steps = [float(sample["route_steps"]) for sample in samples]
    count = len(samples)
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
