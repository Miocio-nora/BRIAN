from __future__ import annotations

from brian_sphere_llm.eval.stage_gate_report import pearson_correlation


def difficulty_step_correlation(baseline_losses: list[float], route_steps: list[float]) -> float | None:
    """Critical diagnostic: corr(baseline_sample_loss, route_steps)."""
    return pearson_correlation(baseline_losses, route_steps)
