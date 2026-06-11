from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.logging import write_json


DEFAULT_THRESHOLDS = {
    "fixed_route_loss_ratio_max": 1.03,
    "stage3_loss_ratio_max": 1.20,
    "route_imitation_fixed_min": 0.98,
    "route_imitation_mixed_min": 0.90,
    "route_entropy_min": 0.05,
    "block_load_entropy_min": 0.05,
    "route_path_diversity_min": 0.05,
    "global_attention_mass_min": 1e-6,
    "global_read_gate_min": 1e-6,
}


def make_stage_gate_report(
    run_dirs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
    cost_control_report_path: str | Path | None = None,
    out_by_difficulty_report_path: str | Path | None = None,
    global_kv_retention_report_path: str | Path | None = None,
    long_context_compare_report_path: str | Path | None = None,
    parallel_passing_report_path: str | Path | None = None,
    parallel_compare_report_path: str | Path | None = None,
) -> Path:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    summaries = [_summarize_run(Path(run_dir)) for run_dir in run_dirs]
    by_stage = {summary["stage"]: summary for summary in summaries if summary["stage"]}
    cost_control_report = _read_json_if_exists(Path(cost_control_report_path)) if cost_control_report_path else {}
    out_by_difficulty_report = _read_json_if_exists(Path(out_by_difficulty_report_path)) if out_by_difficulty_report_path else {}
    long_context_compare_report = (
        _read_json_if_exists(Path(long_context_compare_report_path)) if long_context_compare_report_path else {}
    )
    global_kv_retention_report = (
        _read_json_if_exists(Path(global_kv_retention_report_path)) if global_kv_retention_report_path else {}
    )
    parallel_passing_report = (
        _read_json_if_exists(Path(parallel_passing_report_path)) if parallel_passing_report_path else {}
    )
    parallel_compare_report = _read_json_if_exists(Path(parallel_compare_report_path)) if parallel_compare_report_path else {}
    parallel_stage = by_stage.get("stage6_parallel_passing") or by_stage.get("stage7_parallel_passing")
    gates = {
        "stage0_to_1": _gate_stage0(by_stage.get("stage0_baseline")),
        "stage1_to_2": _gate_stage1(by_stage.get("stage1_fixed_route"), by_stage.get("stage0_baseline"), thresholds),
        "stage2_to_3": _gate_stage2(by_stage.get("stage2_router_imitation"), thresholds),
        "stage3_to_4": _gate_stage3(by_stage.get("stage3_scheduled_free_routing"), by_stage.get("stage1_fixed_route"), thresholds),
        "stage4_to_5": _gate_stage4(by_stage.get("stage4_output_action"), cost_control_report, out_by_difficulty_report),
        "stage5_to_6": _gate_stage5(
            by_stage.get("stage5_global_kv"),
            thresholds,
            long_context_compare_report,
            global_kv_retention_report,
        ),
        "stage6_to_scale": _gate_stage6(
            parallel_stage,
            parallel_compare_report,
            parallel_passing_report,
        ),
    }
    report = {
        "run_count": len(summaries),
        "runs": summaries,
        "gates": gates,
        "overall_status": _overall_status(gates),
        "thresholds": thresholds,
        "supplemental_reports": {
            "cost_control_report": str(cost_control_report_path) if cost_control_report_path else None,
            "out_by_difficulty_report": str(out_by_difficulty_report_path) if out_by_difficulty_report_path else None,
            "global_kv_retention_report": str(global_kv_retention_report_path)
            if global_kv_retention_report_path
            else None,
            "long_context_compare_report": str(long_context_compare_report_path) if long_context_compare_report_path else None,
            "parallel_passing_report": str(parallel_passing_report_path) if parallel_passing_report_path else None,
            "parallel_compare_report": str(parallel_compare_report_path) if parallel_compare_report_path else None,
        },
    }
    if output_path is None:
        output_path = Path("reports") / "stage_gate_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_dev = [x - x_mean for x in x_values]
    y_dev = [y - y_mean for y in y_values]
    denom_x = math.sqrt(sum(value * value for value in x_dev))
    denom_y = math.sqrt(sum(value * value for value in y_dev))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(x_dev, y_dev)) / (denom_x * denom_y)


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "routing_report.json").exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    config = _read_yaml_if_exists(run_dir / "config_resolved.yaml")
    model_stats = _read_json_if_exists(run_dir / "model_stats.json")
    routing_report = _read_json_if_exists(run_dir / "routing_report.json")
    baseline_difficulty_report = _read_json_if_exists(run_dir / "baseline_difficulty_report.json")
    fixed_route_stability_report = _read_json_if_exists(run_dir / "fixed_route_stability_report.json")
    pseudo_route_curriculum_report = _read_json_if_exists(run_dir / "pseudo_route_curriculum_report.json")
    scheduled_routing_report = _read_json_if_exists(run_dir / "scheduled_routing_report.json")
    difficulty_report = _read_json_if_exists(run_dir / "difficulty_step_report.json")
    determinism_report = _read_json_if_exists(run_dir / "eval_determinism_report.json")
    data_manifest_ref = _read_json_if_exists(run_dir / "data_manifest_ref.json")
    global_kv_retention_report = _read_json_if_exists(run_dir / "global_kv_retention_report.json")
    parallel_passing_report = _read_json_if_exists(run_dir / "parallel_passing_report.json")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    train_rows = _read_jsonl(run_dir / "train_log.jsonl")
    resume_rows = _read_jsonl(run_dir / "resume_events.jsonl")
    stage = str(config.get("stage", "")) if config else _stage_from_name(run_dir.name)
    final_eval = eval_rows[-1] if eval_rows else {}
    final_train = train_rows[-1] if train_rows else {}
    difficulty_corr = _num(difficulty_report.get("difficulty_step_correlation"))
    if difficulty_corr is None:
        difficulty_corr = _difficulty_step_corr(train_rows)
    summary = {
        "run_dir": str(run_dir),
        "stage": stage,
        "model_name": model_stats.get("model_name", ""),
        "has_checkpoint_latest": (run_dir / "checkpoint_latest" / "state.pt").exists(),
        "has_checkpoint_best": (run_dir / "checkpoint_best" / "state.pt").exists(),
        "has_eval_log": bool(eval_rows),
        "has_train_log": bool(train_rows),
        "has_resume_event": bool(resume_rows),
        "latest_resume_event": resume_rows[-1] if resume_rows else {},
        "data_manifest_ref": data_manifest_ref,
        "data_manifest_ref_present": bool(data_manifest_ref),
        "data_manifest_ref_checks": _data_manifest_ref_checks(data_manifest_ref),
        "validation_loss": _num(final_eval.get("validation_loss")),
        "perplexity": _num(final_eval.get("perplexity")),
        "train_loss": _num(final_train.get("loss")),
        "routing": routing_report.get("summary", {}),
        "latest_eval": final_eval,
        "baseline_difficulty_report_present": bool(baseline_difficulty_report),
        "baseline_difficulty_sample_count": _num(baseline_difficulty_report.get("sample_count")),
        "baseline_difficulty_bin_count": _num(baseline_difficulty_report.get("difficulty_bin_count")),
        "baseline_difficulty_by_bin": baseline_difficulty_report.get("by_difficulty", {}),
        "fixed_route_stability_report_present": bool(fixed_route_stability_report),
        "fixed_route_stability_status": fixed_route_stability_report.get("overall_status"),
        "fixed_route_stability_checks": fixed_route_stability_report.get("checks", {}),
        "pseudo_route_curriculum_report_present": bool(pseudo_route_curriculum_report),
        "pseudo_route_curriculum_status": pseudo_route_curriculum_report.get("overall_status"),
        "pseudo_route_curriculum_checks": pseudo_route_curriculum_report.get("checks", {}),
        "pseudo_route_curriculum_by_difficulty": pseudo_route_curriculum_report.get("by_difficulty", {}),
        "scheduled_routing_report_present": bool(scheduled_routing_report),
        "scheduled_routing_status": scheduled_routing_report.get("overall_status"),
        "scheduled_routing_checks": scheduled_routing_report.get("checks", {}),
        "scheduled_routing_logged_values": scheduled_routing_report.get("logged_schedule_values", []),
        "global_kv_retention_report": global_kv_retention_report,
        "global_kv_retention_report_present": bool(global_kv_retention_report),
        "global_kv_retention_status": global_kv_retention_report.get("overall_status"),
        "global_kv_retention_checks": global_kv_retention_report.get("checks", {}),
        "global_kv_retention_metrics": global_kv_retention_report.get("metrics", {}),
        "global_kv_retention_model": global_kv_retention_report.get("model", {}),
        "parallel_passing_report": parallel_passing_report,
        "parallel_passing_report_present": bool(parallel_passing_report),
        "parallel_passing_status": parallel_passing_report.get("overall_status"),
        "parallel_passing_checks": parallel_passing_report.get("checks", {}),
        "parallel_passing_model": parallel_passing_report.get("model", {}),
        "parallel_passing_routing": parallel_passing_report.get("routing", {}),
        "difficulty_step_correlation": difficulty_corr,
        "difficulty_sample_count": _num(difficulty_report.get("sample_count")),
        "eval_determinism_status": determinism_report.get("overall_status"),
        "eval_determinism_checks": determinism_report.get("checks", {}),
        "eval_determinism_report_present": bool(determinism_report),
    }
    return summary


def _gate_stage0(stage0: dict[str, Any] | None) -> dict[str, Any]:
    resume_event = stage0.get("latest_resume_event") if stage0 else {}
    resume_event_checks = _resume_event_checks(resume_event)
    determinism_checks = stage0.get("eval_determinism_checks", {}) if stage0 else {}
    checks = {
        "checkpoint_resume_artifact": bool(stage0 and stage0["has_checkpoint_latest"]),
        "checkpoint_best_artifact": bool(stage0 and stage0["has_checkpoint_best"]),
        "checkpoint_resume_event": bool(stage0 and stage0.get("has_resume_event")),
        "checkpoint_resume_event_valid": all(resume_event_checks.values()),
        "eval_log_present": bool(stage0 and stage0["has_eval_log"]),
        "validation_loss_finite": _finite(stage0.get("validation_loss") if stage0 else None),
        "baseline_difficulty_report_present": bool(stage0 and stage0.get("baseline_difficulty_report_present")),
        "baseline_difficulty_samples_present": bool(
            stage0
            and _num(stage0.get("baseline_difficulty_sample_count")) is not None
            and stage0["baseline_difficulty_sample_count"] >= 1
        ),
        "baseline_difficulty_bins_present": bool(
            stage0
            and _num(stage0.get("baseline_difficulty_bin_count")) is not None
            and stage0["baseline_difficulty_bin_count"] >= 3
        ),
        "eval_determinism_report_present": bool(stage0 and stage0.get("eval_determinism_report_present")),
        "eval_deterministic": bool(stage0 and stage0.get("eval_determinism_status") == "pass"),
        "eval_determinism_checks_passed": _eval_determinism_checks_passed(determinism_checks),
        **_data_manifest_gate_checks(stage0),
    }
    return _gate(
        "Stage 0 baseline trains, checkpoints, and evaluates deterministically",
        checks,
        {
            "eval_determinism_status": stage0.get("eval_determinism_status") if stage0 else None,
            "eval_determinism_checks": determinism_checks,
            "latest_resume_event": stage0.get("latest_resume_event") if stage0 else {},
            "resume_event_checks": resume_event_checks,
            **_data_manifest_gate_extras(stage0),
            "baseline_difficulty_by_bin": stage0.get("baseline_difficulty_by_bin") if stage0 else {},
        },
    )


def _gate_stage1(stage1: dict[str, Any] | None, stage0: dict[str, Any] | None, thresholds: dict[str, float]) -> dict[str, Any]:
    ratio = None
    if stage1 and stage0 and _finite(stage1.get("validation_loss")) and _finite(stage0.get("validation_loss")):
        ratio = stage1["validation_loss"] / max(1e-9, stage0["validation_loss"])
    checks = {
        "loss_within_1_to_3_percent": ratio is not None and ratio <= thresholds["fixed_route_loss_ratio_max"],
        "route_imitation_accuracy": _metric_at_least(stage1, "route_imitation_accuracy", thresholds["route_imitation_fixed_min"]),
        "position_state_finite": _finite(_routing_metric(stage1, "position_norm_mean")),
        "fixed_route_stability_report_present": bool(stage1 and stage1.get("fixed_route_stability_report_present")),
        "fixed_route_stability_passed": bool(stage1 and stage1.get("fixed_route_stability_status") == "pass"),
        "checkpoint_present": bool(stage1 and stage1["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage1),
    }
    return _gate(
        "Stage 1 fixed route wrapper matches baseline and router imitates fixed path",
        checks,
        {
            "loss_ratio": ratio,
            "fixed_route_stability_status": stage1.get("fixed_route_stability_status") if stage1 else None,
            "fixed_route_stability_checks": stage1.get("fixed_route_stability_checks") if stage1 else {},
            **_data_manifest_gate_extras(stage1),
        },
    )


def _gate_stage2(stage2: dict[str, Any] | None, thresholds: dict[str, float]) -> dict[str, Any]:
    checks = {
        "route_imitation_accuracy": _metric_at_least(stage2, "route_imitation_accuracy", thresholds["route_imitation_mixed_min"]),
        "lm_loss_finite": _finite(stage2.get("validation_loss") if stage2 else None),
        "block_usage_non_degenerate": _block_usage_non_degenerate(stage2),
        "block_load_entropy_present": _metric_at_least(stage2, "block_load_entropy", thresholds["block_load_entropy_min"]),
        "pseudo_route_curriculum_report_present": bool(stage2 and stage2.get("pseudo_route_curriculum_report_present")),
        "pseudo_route_curriculum_passed": bool(stage2 and stage2.get("pseudo_route_curriculum_status") == "pass"),
        "checkpoint_present": bool(stage2 and stage2["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage2),
    }
    return _gate(
        "Stage 2 mixed pseudo routing is stable and non-degenerate",
        checks,
        {
            "pseudo_route_curriculum_status": stage2.get("pseudo_route_curriculum_status") if stage2 else None,
            "pseudo_route_curriculum_checks": stage2.get("pseudo_route_curriculum_checks") if stage2 else {},
            "pseudo_route_curriculum_by_difficulty": stage2.get("pseudo_route_curriculum_by_difficulty") if stage2 else {},
            **_data_manifest_gate_extras(stage2),
        },
    )


def _gate_stage3(stage3: dict[str, Any] | None, stage1: dict[str, Any] | None, thresholds: dict[str, float]) -> dict[str, Any]:
    ratio = None
    if stage3 and stage1 and _finite(stage3.get("validation_loss")) and _finite(stage1.get("validation_loss")):
        ratio = stage3["validation_loss"] / max(1e-9, stage1["validation_loss"])
    checks = {
        "validation_loss_not_collapsed": ratio is not None and ratio <= thresholds["stage3_loss_ratio_max"],
        "route_entropy_present": _metric_at_least(stage3, "route_entropy", thresholds["route_entropy_min"]),
        "block_load_entropy_present": _metric_at_least(stage3, "block_load_entropy", thresholds["block_load_entropy_min"]),
        "route_path_diversity_present": _metric_at_least(stage3, "route_path_diversity", thresholds["route_path_diversity_min"]),
        "average_route_steps_present": _finite(_routing_metric(stage3, "average_route_steps")),
        "difficulty_report_present": _num(stage3.get("difficulty_sample_count") if stage3 else None) is not None
        and float(stage3["difficulty_sample_count"]) >= 1.0,
        "difficulty_step_correlation_finite": _finite(stage3.get("difficulty_step_correlation") if stage3 else None),
        "scheduled_routing_report_present": bool(stage3 and stage3.get("scheduled_routing_report_present")),
        "scheduled_routing_passed": bool(stage3 and stage3.get("scheduled_routing_status") == "pass"),
        "checkpoint_present": bool(stage3 and stage3["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage3),
    }
    return _gate(
        "Stage 3 scheduled free routing remains stable",
        checks,
        {
            "loss_ratio_vs_stage1": ratio,
            "scheduled_routing_status": stage3.get("scheduled_routing_status") if stage3 else None,
            "scheduled_routing_checks": stage3.get("scheduled_routing_checks") if stage3 else {},
            "scheduled_routing_logged_values": stage3.get("scheduled_routing_logged_values") if stage3 else [],
            **_data_manifest_gate_extras(stage3),
        },
    )


def _gate_stage4(
    stage4: dict[str, Any] | None,
    cost_control_report: dict[str, Any] | None = None,
    out_by_difficulty_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exit_hist = _latest_exit_hist(stage4)
    total_exits = sum(exit_hist.values()) if exit_hist else 0
    immediate = exit_hist.get("1", 0) if exit_hist else 0
    nonzero_exits = sum(count for step, count in exit_hist.items() if step != "0") if exit_hist else 0
    cost_analysis = cost_control_report.get("analysis", {}) if cost_control_report else {}
    cost_checks = cost_analysis.get("checks", {}) if isinstance(cost_analysis.get("checks"), dict) else {}
    out_checks = out_by_difficulty_report.get("checks", {}) if isinstance(out_by_difficulty_report.get("checks"), dict) else {}
    checks = {
        "exit_distribution_present": bool(exit_hist),
        "not_all_immediate_exit": bool(exit_hist) and immediate < total_exits,
        "not_never_exit": bool(exit_hist) and nonzero_exits > 0,
        "average_route_steps_present": _finite(_routing_metric(stage4, "average_route_steps")),
        "cost_control_report_present": bool(cost_control_report),
        "cost_control_active_range_present": bool(cost_checks.get("active_compute_range_present", False)),
        "cost_control_active_not_increasing": bool(cost_checks.get("active_compute_not_increasing_with_cost", False)),
        "cost_control_output_not_decreasing": bool(cost_checks.get("output_probability_not_decreasing_with_cost", False)),
        "out_by_difficulty_report_present": bool(out_by_difficulty_report),
        "out_by_difficulty_passed": bool(out_by_difficulty_report.get("overall_status") == "pass"),
        "hard_compute_not_below_easy": bool(out_checks.get("route_steps_non_decreasing_with_difficulty", False))
        and bool(out_checks.get("active_compute_non_decreasing_with_difficulty", False)),
        "easy_output_probability_not_below_hard": bool(out_checks.get("easy_output_probability_at_least_hard", False)),
        "checkpoint_present": bool(stage4 and stage4["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage4),
    }
    return _gate(
        "Stage 4 hard OUT produces controllable exits and difficulty-conditioned compute",
        checks,
        {
            "first_exit_step_histogram": exit_hist,
            "cost_control_status": cost_analysis.get("status") if cost_analysis else None,
            "cost_control_active_block_evals_range": cost_analysis.get("active_block_evals_range") if cost_analysis else None,
            "cost_control_report_run_count": cost_control_report.get("run_count") if cost_control_report else None,
            "out_by_difficulty_status": out_by_difficulty_report.get("overall_status") if out_by_difficulty_report else None,
            "out_by_difficulty_checks": out_checks,
            "out_by_difficulty_deltas": out_by_difficulty_report.get("deltas", {}) if out_by_difficulty_report else {},
            **_data_manifest_gate_extras(stage4),
        },
    )


def _gate_stage5(
    stage5: dict[str, Any] | None,
    thresholds: dict[str, float],
    long_context_compare_report: dict[str, Any] | None = None,
    global_kv_retention_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparisons = long_context_compare_report.get("comparisons", []) if long_context_compare_report else []
    long_context_key_checks = [
        "global_kv_active",
        "quality_not_worse",
        "memory_budget_present",
        "global_budget_below_local_context",
    ]
    any_long_context_pass = _any_passing_comparison_with_checks(comparisons, long_context_key_checks)
    retention_report = global_kv_retention_report or (stage5.get("global_kv_retention_report") if stage5 else None) or {}
    retention_checks = retention_report.get("checks", {}) if isinstance(retention_report.get("checks"), dict) else {}
    checks = {
        "global_attention_mass_nonzero": _metric_at_least(stage5, "global_attention_mass", thresholds["global_attention_mass_min"]),
        "global_read_gate_nonzero": _metric_at_least(stage5, "global_read_gate_mean", thresholds["global_read_gate_min"]),
        "global_cache_slots_present": _metric_at_least(stage5, "global_cache_slots_mean", 1.0),
        "global_kv_retention_report_present": bool(retention_report),
        "global_kv_retention_passed": bool(retention_report.get("overall_status") == "pass"),
        "sink_window_retention_configured": bool(retention_checks.get("sink_slots_configured", False))
        and bool(retention_checks.get("window_slots_configured", False)),
        "sink_window_attention_measured": bool(retention_checks.get("sink_attention_mass_measured", False))
        and bool(retention_checks.get("window_attention_mass_measured", False)),
        "cache_slots_within_retention_capacity": bool(retention_checks.get("cache_slots_within_retention_capacity", False)),
        "long_context_compare_report_present": bool(long_context_compare_report),
        "long_context_compare_passed": bool(
            long_context_compare_report and long_context_compare_report.get("overall_status") == "pass"
        ),
        "long_context_global_kv_benefit_proxy": any_long_context_pass,
        "checkpoint_present": bool(stage5 and stage5["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage5),
    }
    return _gate(
        "Stage 5 Global KV retention is active and has long-context comparison evidence",
        checks,
        {
            "global_kv_retention_status": retention_report.get("overall_status") if retention_report else None,
            "global_kv_retention_checks": retention_checks,
            "global_kv_retention_metrics": retention_report.get("metrics", {}) if retention_report else {},
            "global_kv_retention_model": retention_report.get("model", {}) if retention_report else {},
            "long_context_compare_status": long_context_compare_report.get("overall_status") if long_context_compare_report else None,
            "long_context_compare_candidate_count": long_context_compare_report.get("candidate_count")
            if long_context_compare_report
            else None,
            **_data_manifest_gate_extras(stage5),
        },
    )


def _gate_stage6(
    stage6: dict[str, Any] | None,
    parallel_compare_report: dict[str, Any] | None = None,
    parallel_passing_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparisons = parallel_compare_report.get("comparisons", []) if parallel_compare_report else []
    parallel_key_checks = [
        "parallel_branch_active",
        "parallel_score_margin_present",
        "quality_not_worse",
        "active_compute_bounded",
        "estimated_flops_bounded",
        "parallel_branch_benefit_proxy",
    ]
    any_parallel_pass = _any_passing_comparison_with_checks(comparisons, parallel_key_checks)
    passing_report = parallel_passing_report or (stage6.get("parallel_passing_report") if stage6 else None) or {}
    passing_checks = passing_report.get("checks", {}) if isinstance(passing_report.get("checks"), dict) else {}
    checks = {
        "parallel_branch_count_present": _metric_at_least(stage6, "parallel_branch_count_mean", 1.0),
        "parallel_score_margin_present": _finite(_routing_metric(stage6, "parallel_score_margin_mean")),
        "parallel_passing_report_present": bool(passing_report),
        "parallel_passing_report_passed": bool(passing_report.get("overall_status") == "pass"),
        "parallel_beam_bounded": bool(passing_checks.get("beam_size_within_limit", False)),
        "parallel_branch_count_bounded_by_beam": bool(passing_checks.get("branch_count_bounded_by_beam", False)),
        "parallel_branch_cost_enabled": bool(passing_checks.get("branch_cost_enabled", False)),
        "parallel_delta_cache_bounded": bool(passing_checks.get("delta_cache_bounded_by_window", False)),
        "global_cache_or_local_route_present": _finite(_routing_metric(stage6, "global_cache_slots_mean"))
        or _finite(_routing_metric(stage6, "average_route_steps")),
        "parallel_compare_report_present": bool(parallel_compare_report),
        "parallel_compare_passed": bool(
            parallel_compare_report and parallel_compare_report.get("overall_status") == "pass"
        ),
        "parallel_branch_benefit_proxy": any_parallel_pass,
        "checkpoint_present": bool(stage6 and stage6["has_checkpoint_latest"]),
        **_data_manifest_gate_checks(stage6),
    }
    return _gate(
        "Stage 6 parallel passing is bounded and has comparison evidence",
        checks,
        {
            "parallel_passing_status": passing_report.get("overall_status") if passing_report else None,
            "parallel_passing_checks": passing_checks,
            "parallel_passing_model": passing_report.get("model", {}) if passing_report else {},
            "parallel_passing_routing": passing_report.get("routing", {}) if passing_report else {},
            "parallel_compare_status": parallel_compare_report.get("overall_status") if parallel_compare_report else None,
            "parallel_compare_candidate_count": parallel_compare_report.get("candidate_count")
            if parallel_compare_report
            else None,
            **_data_manifest_gate_extras(stage6),
        },
    )


def _gate(description: str, checks: dict[str, bool], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    if not checks:
        status = "unknown"
    elif all(checks.values()):
        status = "pass"
    elif any(checks.values()):
        status = "warn"
    else:
        status = "fail"
    return {"status": status, "description": description, "checks": checks, **(extras or {})}


def _overall_status(gates: dict[str, dict[str, Any]]) -> str:
    statuses = [gate["status"] for gate in gates.values()]
    if all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "fail" for status in statuses):
        return "fail"
    return "warn"


def _any_passing_comparison_with_checks(comparisons: Any, required_checks: list[str]) -> bool:
    if not isinstance(comparisons, list):
        return False
    for item in comparisons:
        if not isinstance(item, dict) or item.get("status") != "pass":
            continue
        checks = item.get("checks")
        if not isinstance(checks, dict):
            continue
        if all(checks.get(check) is True for check in required_checks):
            return True
    return False


def _resume_event_checks(event: Any) -> dict[str, bool]:
    if not isinstance(event, dict):
        event = {}
    resumed_from_step = _int_like(event.get("resumed_from_step"))
    target_max_steps = _int_like(event.get("target_max_steps"))
    checkpoint = event.get("checkpoint")
    return {
        "checkpoint_points_to_latest": isinstance(checkpoint, str) and Path(checkpoint).name == "checkpoint_latest",
        "resumed_from_positive_step": resumed_from_step is not None and resumed_from_step >= 1,
        "target_after_resume_step": resumed_from_step is not None
        and target_max_steps is not None
        and target_max_steps > resumed_from_step,
        "optimizer_state_loaded": event.get("optimizer_state_loaded") is True,
    }


def _data_manifest_ref_checks(ref: Any) -> dict[str, bool]:
    if not isinstance(ref, dict):
        ref = {}
    return {
        "path_present": _nonempty_string(ref.get("path")),
        "tokenized_dir_present": _nonempty_string(ref.get("tokenized_dir")),
        "stats_path_present": _nonempty_string(ref.get("stats_path")),
        "recipe_name_present": _nonempty_string(ref.get("recipe_name")),
        "sequence_length_positive": _positive_number(ref.get("sequence_length")),
        "num_tokens_train_positive": _positive_number(ref.get("num_tokens_train")),
        "num_tokens_val_positive": _positive_number(ref.get("num_tokens_val")),
        "sha256_manifest_present": _nonempty_string(ref.get("sha256_manifest")),
        "source_mixture_present": isinstance(ref.get("source_mixture_realized"), dict)
        and bool(ref.get("source_mixture_realized")),
    }


def _data_manifest_gate_checks(summary: dict[str, Any] | None) -> dict[str, bool]:
    checks = summary.get("data_manifest_ref_checks", {}) if summary else {}
    return {
        "data_manifest_ref_present": bool(summary and summary.get("data_manifest_ref_present")),
        "data_manifest_ref_valid": isinstance(checks, dict) and all(checks.values()),
    }


def _data_manifest_gate_extras(summary: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "data_manifest_ref": summary.get("data_manifest_ref") if summary else {},
        "data_manifest_ref_checks": summary.get("data_manifest_ref_checks") if summary else {},
    }


def _eval_determinism_checks_passed(checks: Any) -> bool:
    required = [
        "checkpoint_loaded",
        "two_eval_passes_completed",
        "compared_numeric_metrics_present",
        "numeric_metrics_within_tolerance",
    ]
    return isinstance(checks, dict) and all(checks.get(key) is True for key in required)


def _difficulty_step_corr(rows: list[dict[str, Any]]) -> float | None:
    difficulty = []
    steps = []
    for row in rows:
        if isinstance(row.get("baseline_sample_loss"), (int, float)) and isinstance(row.get("average_route_steps"), (int, float)):
            difficulty.append(float(row["baseline_sample_loss"]))
            steps.append(float(row["average_route_steps"]))
    return pearson_correlation(difficulty, steps)


def _routing_metric(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary:
        return None
    value = summary.get("routing", {}).get(key)
    return _num(value)


def _metric_at_least(summary: dict[str, Any] | None, key: str, minimum: float) -> bool:
    value = _routing_metric(summary, key)
    return value is not None and value >= minimum


def _block_usage_non_degenerate(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    report = _read_json_if_exists(Path(summary["run_dir"]) / "routing_report.json")
    hist = report.get("latest_block_histogram", {})
    if not hist:
        return False
    out_action = max(int(action) for action in hist)
    internal_counts = [int(count) for action, count in hist.items() if int(action) != out_action]
    total = sum(internal_counts)
    if total <= 0:
        return False
    return max(internal_counts) / total < 0.95


def _latest_exit_hist(summary: dict[str, Any] | None) -> dict[str, int]:
    if not summary:
        return {}
    rows = _read_jsonl(Path(summary["run_dir"]) / "train_log.jsonl")
    for row in reversed(rows):
        hist = row.get("first_exit_step_histogram")
        if isinstance(hist, dict):
            return {str(key): int(value) for key, value in hist.items()}
    return {}


def _finite(value: Any) -> bool:
    numeric = _num(value)
    return numeric is not None and math.isfinite(numeric)


def _int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    numeric = _num(value)
    return numeric is not None and math.isfinite(numeric) and numeric > 0


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _stage_from_name(name: str) -> str:
    for part in name.split("_"):
        if part.startswith("stage"):
            return part
    return ""
