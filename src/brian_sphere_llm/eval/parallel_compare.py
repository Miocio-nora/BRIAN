from __future__ import annotations

from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.compute_report import summarize_run
from brian_sphere_llm.utils.logging import write_json


def make_parallel_comparison_report(
    baseline_run: str | Path,
    candidate_run_dirs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    max_validation_loss_delta: float = 0.0,
    max_active_layer_eval_ratio: float = 2.0,
    max_estimated_flops_ratio: float = 2.0,
    min_throughput_ratio: float = 0.0,
    min_parallel_branch_count: float = 1.5,
) -> Path:
    baseline = summarize_run(baseline_run)
    rows = [
        _compare_candidate(
            baseline,
            summarize_run(candidate_run, baseline=baseline),
            max_validation_loss_delta=max_validation_loss_delta,
            max_active_layer_eval_ratio=max_active_layer_eval_ratio,
            max_estimated_flops_ratio=max_estimated_flops_ratio,
            min_throughput_ratio=min_throughput_ratio,
            min_parallel_branch_count=min_parallel_branch_count,
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
            "max_active_layer_eval_ratio": max_active_layer_eval_ratio,
            "max_estimated_flops_ratio": max_estimated_flops_ratio,
            "min_throughput_ratio": min_throughput_ratio,
            "min_parallel_branch_count": min_parallel_branch_count,
        },
        "overall_status": _overall_status(rows),
    }
    if output_path is None:
        output_path = Path("reports") / "parallel_compare.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _compare_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_validation_loss_delta: float,
    max_active_layer_eval_ratio: float,
    max_estimated_flops_ratio: float,
    min_throughput_ratio: float,
    min_parallel_branch_count: float,
) -> dict[str, Any]:
    comparison = candidate.get("baseline_comparison", {})
    baseline_routing = baseline.get("routing", {})
    routing = candidate.get("routing", {})
    baseline_top_k = _num(baseline.get("top_k"))
    baseline_weighted_fusion_ratio = _num(baseline_routing.get("weighted_fusion_ratio"))
    branch_count = _num(routing.get("parallel_branch_count_mean"))
    score_margin = _num(routing.get("parallel_score_margin_mean"))
    validation_delta = _num(comparison.get("validation_loss_delta"))
    active_ratio = _num(comparison.get("active_layer_eval_ratio"))
    flops_ratio = _num(comparison.get("estimated_flops_per_token_ratio"))
    throughput_ratio = _num(comparison.get("tokens_per_second_ratio"))
    baseline_stage5 = baseline.get("stage") == "stage5_global_kv"
    baseline_scheduled = baseline.get("route_mode") == "scheduled"
    baseline_global_kv = baseline.get("global_kv_enabled") is True
    baseline_non_parallel = baseline.get("parallel_passing_enabled") is False
    baseline_topk_weighted = (
        baseline_top_k is not None
        and baseline_top_k > 1.0
        and baseline_weighted_fusion_ratio is not None
        and baseline_weighted_fusion_ratio > 0.0
    )
    candidate_parallel_stage = candidate.get("stage") in {"stage6_parallel_passing", "stage7_parallel_passing"}
    candidate_parallel_route = candidate.get("route_mode") == "parallel"
    candidate_parallel_enabled = candidate.get("parallel_passing_enabled") is True
    candidate_global_kv = candidate.get("global_kv_enabled") is True
    quality_not_worse = validation_delta is not None and validation_delta <= max_validation_loss_delta
    active_compute_bounded = active_ratio is not None and active_ratio <= max_active_layer_eval_ratio
    flops_bounded = flops_ratio is not None and flops_ratio <= max_estimated_flops_ratio
    throughput_not_collapsed = min_throughput_ratio <= 0.0 or (
        throughput_ratio is not None and throughput_ratio >= min_throughput_ratio
    )
    parallel_branch_active = branch_count is not None and branch_count >= min_parallel_branch_count
    score_margin_present = score_margin is not None
    branch_benefit_proxy = quality_not_worse and (
        validation_delta < 0.0
        or (active_ratio is not None and active_ratio <= 1.0)
        or (flops_ratio is not None and flops_ratio <= 1.0)
        or (throughput_ratio is not None and throughput_ratio >= 1.0)
    )
    checks = {
        "baseline_stage5_global_kv": baseline_stage5,
        "baseline_scheduled_route_mode": baseline_scheduled,
        "baseline_global_kv_enabled": baseline_global_kv,
        "baseline_parallel_passing_disabled": baseline_non_parallel,
        "baseline_topk_weighted_fusion": baseline_topk_weighted,
        "candidate_parallel_stage": candidate_parallel_stage,
        "candidate_parallel_route_mode": candidate_parallel_route,
        "candidate_parallel_passing_enabled": candidate_parallel_enabled,
        "candidate_global_kv_enabled": candidate_global_kv,
        "parallel_branch_active": parallel_branch_active,
        "parallel_score_margin_present": score_margin_present,
        "quality_not_worse": quality_not_worse,
        "active_compute_bounded": active_compute_bounded,
        "estimated_flops_bounded": flops_bounded,
        "throughput_not_collapsed": throughput_not_collapsed,
        "parallel_branch_benefit_proxy": branch_benefit_proxy,
    }
    return {
        "candidate_run": candidate["run_dir"],
        "candidate_stage": candidate.get("stage"),
        "baseline_stage": baseline.get("stage"),
        "baseline_top_k": baseline_top_k,
        "baseline_weighted_fusion_ratio": baseline_weighted_fusion_ratio,
        "candidate_route_mode": candidate.get("route_mode"),
        "baseline_route_mode": baseline.get("route_mode"),
        "candidate": _summary_row(candidate),
        "baseline_comparison": {
            "validation_loss_delta": validation_delta,
            "validation_loss_ratio": _num(comparison.get("validation_loss_ratio")),
            "active_layer_eval_ratio": active_ratio,
            "estimated_flops_per_token_ratio": flops_ratio,
            "tokens_per_second_ratio": throughput_ratio,
        },
        "parallel": {
            "parallel_branch_count_mean": branch_count,
            "parallel_score_margin_mean": score_margin,
        },
        "checks": checks,
        "status": _status(checks),
    }


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    routing = summary.get("routing", {})
    return {
        "run_dir": summary.get("run_dir"),
        "stage": summary.get("stage"),
        "route_mode": summary.get("route_mode"),
        "top_k": summary.get("top_k"),
        "global_kv_enabled": summary.get("global_kv_enabled"),
        "parallel_passing_enabled": summary.get("parallel_passing_enabled"),
        "validation_loss": summary.get("validation_loss"),
        "active_layer_evals_per_token": summary.get("active_layer_evals_per_token"),
        "active_layer_ratio": summary.get("active_layer_ratio"),
        "estimated_flops_per_token": summary.get("estimated_flops_per_token"),
        "tokens_per_second_mean": summary.get("tokens_per_second_mean"),
        "train_latency_ms_per_token_mean": summary.get("train_latency_ms_per_token_mean"),
        "inference_latency_ms_per_token_latest": summary.get("inference_latency_ms_per_token_latest"),
        "routing": {
            "average_route_steps": routing.get("average_route_steps"),
            "active_internal_decision_fraction": routing.get("active_internal_decision_fraction"),
            "weighted_fusion_ratio": routing.get("weighted_fusion_ratio"),
            "parallel_branch_count_mean": routing.get("parallel_branch_count_mean"),
            "parallel_score_margin_mean": routing.get("parallel_score_margin_mean"),
        },
    }


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


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
