from __future__ import annotations

from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.compute_report import summarize_run
from brian_sphere_llm.utils.logging import write_json


def make_hard_exit_comparison_report(
    baseline_run: str | Path,
    candidate_run_dirs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    max_validation_loss_delta: float = 0.0,
    max_latency_ratio: float = 1.0,
    max_inference_time_ratio: float = 1.0,
    max_route_step_ratio: float = 1.0,
) -> Path:
    baseline = summarize_run(baseline_run)
    rows = [
        _compare_candidate(
            baseline,
            summarize_run(candidate_run, baseline=baseline),
            max_validation_loss_delta=max_validation_loss_delta,
            max_latency_ratio=max_latency_ratio,
            max_inference_time_ratio=max_inference_time_ratio,
            max_route_step_ratio=max_route_step_ratio,
        )
        for candidate_run in candidate_run_dirs
    ]
    report = {
        "baseline_run": str(baseline_run),
        "baseline": _summary_row(baseline),
        "candidate_count": len(rows),
        "comparisons": rows,
        "thresholds": {
            "max_validation_loss_delta": max_validation_loss_delta,
            "max_latency_ratio": max_latency_ratio,
            "max_inference_time_ratio": max_inference_time_ratio,
            "max_route_step_ratio": max_route_step_ratio,
        },
        "overall_status": _overall_status(rows),
    }
    output_path = Path(output_path or Path("reports") / "hard_exit_compare.json")
    write_json(report, output_path)
    return output_path


def _compare_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_validation_loss_delta: float,
    max_latency_ratio: float,
    max_inference_time_ratio: float,
    max_route_step_ratio: float,
) -> dict[str, Any]:
    comparison = candidate.get("baseline_comparison", {})
    baseline_routing = baseline.get("routing", {}) if isinstance(baseline.get("routing"), dict) else {}
    candidate_routing = candidate.get("routing", {}) if isinstance(candidate.get("routing"), dict) else {}
    validation_delta = _num(comparison.get("validation_loss_delta"))
    latency_ratio = _num(comparison.get("inference_latency_ms_per_token_ratio"))
    inference_time_ratio = _ratio(
        candidate.get("inference_time_seconds_latest"),
        baseline.get("inference_time_seconds_latest"),
    )
    route_step_ratio = _ratio(candidate_routing.get("average_route_steps"), baseline_routing.get("average_route_steps"))
    checks = {
        "baseline_stage4_scheduled_free_routing": baseline.get("stage") == "stage4_scheduled_free_routing",
        "baseline_without_hard_exit": baseline.get("hard_exit_enabled") is False,
        "candidate_stage4_output_action": candidate.get("stage") == "stage4_output_action",
        "candidate_with_hard_exit": candidate.get("hard_exit_enabled") is True,
        "candidate_top1_hard_exit_rule": _top1_hard_exit_rule(candidate),
        "inference_timing_present": _num(baseline.get("inference_time_seconds_latest")) is not None
        and _num(candidate.get("inference_time_seconds_latest")) is not None
        and _num(baseline.get("inference_latency_ms_per_token_latest")) is not None
        and _num(candidate.get("inference_latency_ms_per_token_latest")) is not None,
        "latency_ratio_within_threshold": latency_ratio is not None and latency_ratio <= max_latency_ratio,
        "inference_time_ratio_within_threshold": inference_time_ratio is not None
        and inference_time_ratio <= max_inference_time_ratio,
        "route_steps_not_increasing": route_step_ratio is not None and route_step_ratio <= max_route_step_ratio,
        "validation_loss_not_worse": validation_delta is not None and validation_delta <= max_validation_loss_delta,
    }
    return {
        "candidate_run": candidate.get("run_dir"),
        "candidate": _summary_row(candidate),
        "baseline_comparison": {
            "validation_loss_delta": validation_delta,
            "validation_loss_ratio": _num(comparison.get("validation_loss_ratio")),
            "inference_latency_ms_per_token_ratio": latency_ratio,
            "inference_time_seconds_ratio": inference_time_ratio,
            "average_route_steps_ratio": route_step_ratio,
        },
        "checks": checks,
        "status": _status(checks),
    }


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    routing = summary.get("routing", {}) if isinstance(summary.get("routing"), dict) else {}
    return {
        "run_dir": summary.get("run_dir"),
        "stage": summary.get("stage"),
        "model_name": summary.get("model_name"),
        "route_mode": summary.get("route_mode"),
        "hard_exit_enabled": summary.get("hard_exit_enabled"),
        "hard_exit_top1_rule": _top1_hard_exit_rule(summary),
        "top_k": summary.get("top_k"),
        "parallel_passing_enabled": summary.get("parallel_passing_enabled"),
        "parallel_exit_policy": summary.get("parallel_exit_policy"),
        "validation_loss": summary.get("validation_loss"),
        "inference_time_seconds_latest": summary.get("inference_time_seconds_latest"),
        "inference_tokens_per_second_latest": summary.get("inference_tokens_per_second_latest"),
        "inference_latency_ms_per_token_latest": summary.get("inference_latency_ms_per_token_latest"),
        "average_route_steps": routing.get("average_route_steps"),
        "active_internal_decision_fraction": routing.get("active_internal_decision_fraction"),
    }


def _top1_hard_exit_rule(summary: dict[str, Any]) -> bool:
    if summary.get("hard_exit_enabled") is not True:
        return False
    if summary.get("parallel_passing_enabled") is True:
        return summary.get("parallel_exit_policy") == "top1"
    return True


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    statuses = [row["status"] for row in rows]
    if all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "fail" for status in statuses):
        return "fail"
    return "warn"


def _status(checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "pass"
    if any(checks.values()):
        return "warn"
    return "fail"


def _ratio(value: Any, baseline: Any) -> float | None:
    left = _num(value)
    right = _num(baseline)
    if left is None or right is None or right == 0.0:
        return None
    return left / right


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
