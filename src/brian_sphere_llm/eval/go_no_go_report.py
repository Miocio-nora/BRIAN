from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


def make_go_no_go_report(
    *,
    stage_gate_report_path: str | Path,
    output_path: str | Path | None = None,
    phase: str = "all",
    compute_report_path: str | Path | None = None,
    position_ablation_report_path: str | Path | None = None,
    out_by_difficulty_report_path: str | Path | None = None,
    reasoning_baseline_report_path: str | Path | None = None,
    reasoning_candidate_report_paths: list[str | Path] | None = None,
    long_context_compare_report_path: str | Path | None = None,
    global_kv_ablation_report_path: str | Path | None = None,
    parallel_compare_report_path: str | Path | None = None,
    min_difficulty_step_correlation: float = 0.0,
    min_reasoning_delta: float = 0.0,
    max_compute_adjusted_loss_delta: float = 0.0,
    min_visible_cot_reduction: float = 1.0,
    max_reasoning_drop_for_cot: float = 0.0,
    max_global_kv_cache_capacity_ratio: float = 1.0,
    max_inference_latency_ratio: float = 2.0,
) -> Path:
    stage_gate_report = _read_json(Path(stage_gate_report_path))
    compute_report = _read_json_if_present(compute_report_path)
    position_ablation_report = _read_json_if_present(position_ablation_report_path)
    out_by_difficulty_report = _read_json_if_present(out_by_difficulty_report_path)
    long_context_compare_report = _read_json_if_present(long_context_compare_report_path)
    global_kv_ablation_report = _read_json_if_present(global_kv_ablation_report_path)
    parallel_compare_report = _read_json_if_present(parallel_compare_report_path)
    reasoning_baseline_report = _read_json_if_present(reasoning_baseline_report_path)
    reasoning_candidates = [_read_json(Path(path)) for path in (reasoning_candidate_report_paths or [])]

    phases: dict[str, Any] = {}
    if phase in {"all", "r125_to_r350"}:
        phases["r125_to_r350"] = _r125_decision(
            stage_gate_report,
            position_ablation_report=position_ablation_report,
        )
    if phase in {"all", "r350_to_1b"}:
        phases["r350_to_1b"] = _r350_decision(
            stage_gate_report,
            compute_report=compute_report,
            reasoning_baseline_report=reasoning_baseline_report,
            reasoning_candidates=reasoning_candidates,
            out_by_difficulty_report=out_by_difficulty_report,
            long_context_compare_report=long_context_compare_report,
            global_kv_ablation_report=global_kv_ablation_report,
            min_difficulty_step_correlation=min_difficulty_step_correlation,
            min_reasoning_delta=min_reasoning_delta,
        )
    if phase in {"all", "r1b_success"}:
        phases["r1b_success"] = _r1b_success_decision(
            stage_gate_report,
            compute_report=compute_report,
            reasoning_baseline_report=reasoning_baseline_report,
            reasoning_candidates=reasoning_candidates,
            long_context_compare_report=long_context_compare_report,
            global_kv_ablation_report=global_kv_ablation_report,
            max_compute_adjusted_loss_delta=max_compute_adjusted_loss_delta,
            min_reasoning_delta=min_reasoning_delta,
            min_visible_cot_reduction=min_visible_cot_reduction,
            max_reasoning_drop_for_cot=max_reasoning_drop_for_cot,
            max_global_kv_cache_capacity_ratio=max_global_kv_cache_capacity_ratio,
            max_inference_latency_ratio=max_inference_latency_ratio,
        )
    if not phases:
        raise ValueError(f"Unsupported go/no-go phase: {phase}")

    report = {
        "phase": phase,
        "overall_status": _overall_status([item["status"] for item in phases.values()]),
        "recommendation": _recommendation([item["recommendation"] for item in phases.values()]),
        "phases": phases,
        "inputs": {
            "stage_gate_report": str(stage_gate_report_path),
            "compute_report": str(compute_report_path) if compute_report_path else None,
            "position_ablation_report": str(position_ablation_report_path) if position_ablation_report_path else None,
            "out_by_difficulty_report": str(out_by_difficulty_report_path) if out_by_difficulty_report_path else None,
            "reasoning_baseline_report": str(reasoning_baseline_report_path) if reasoning_baseline_report_path else None,
            "reasoning_candidate_reports": [str(path) for path in (reasoning_candidate_report_paths or [])],
            "long_context_compare_report": str(long_context_compare_report_path) if long_context_compare_report_path else None,
            "global_kv_ablation_report": str(global_kv_ablation_report_path) if global_kv_ablation_report_path else None,
            "parallel_compare_report": str(parallel_compare_report_path) if parallel_compare_report_path else None,
        },
        "thresholds": {
            "min_difficulty_step_correlation": min_difficulty_step_correlation,
            "min_reasoning_delta": min_reasoning_delta,
            "max_compute_adjusted_loss_delta": max_compute_adjusted_loss_delta,
            "min_visible_cot_reduction": min_visible_cot_reduction,
            "max_reasoning_drop_for_cot": max_reasoning_drop_for_cot,
            "max_global_kv_cache_capacity_ratio": max_global_kv_cache_capacity_ratio,
            "max_inference_latency_ratio": max_inference_latency_ratio,
        },
        "optional_evidence": {
            "parallel_compare": _parallel_compare_evidence(parallel_compare_report),
        },
    }
    if output_path is None:
        output_path = Path("reports") / "go_no_go_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _r125_decision(stage_gate_report: dict[str, Any], *, position_ablation_report: dict[str, Any]) -> dict[str, Any]:
    gates = stage_gate_report.get("gates", {})
    criteria = [
        _criterion(
            "fixed_route_wrapper_loss_within_threshold",
            _gate_check(gates, "stage1_to_2", "loss_within_1_to_3_percent"),
            _gate_evidence(gates, "stage1_to_2", ["status", "loss_ratio"]),
        ),
        _criterion(
            "router_imitation_accuracy_above_threshold",
            _gate_check(gates, "stage2_to_3", "route_imitation_accuracy"),
            _gate_evidence(gates, "stage2_to_3", ["status"]),
        ),
        _criterion(
            "scheduled_free_routing_not_collapsed",
            _gate_check(gates, "stage3_to_4", "validation_loss_not_collapsed"),
            _gate_evidence(gates, "stage3_to_4", ["status", "loss_ratio_vs_stage1"]),
        ),
        _criterion(
            "route_steps_controlled_by_cost_loss",
            _all_gate_checks(
                gates,
                "stage4_to_5",
                [
                    "cost_control_report_present",
                    "cost_control_active_range_present",
                    "cost_control_active_not_increasing",
                    "cost_control_average_steps_not_increasing",
                    "cost_control_output_not_decreasing",
                ],
            ),
            _gate_evidence(
                gates,
                "stage4_to_5",
                ["status", "cost_control_status", "cost_control_active_block_evals_range"],
            ),
        ),
        _criterion(
            "block_load_not_collapsed",
            _any_gate_checks(
                gates,
                ["stage2_to_3", "stage3_to_4"],
                ["block_usage_non_degenerate", "block_load_entropy_present", "route_path_diversity_present"],
            ),
            {
                "stage2_to_3": _gate_evidence(gates, "stage2_to_3", ["status"]),
                "stage3_to_4": _gate_evidence(gates, "stage3_to_4", ["status"]),
            },
        ),
        _criterion(
            "block_position_ablation_measurable_difference",
            _position_ablation_passed(position_ablation_report),
            _position_ablation_evidence(position_ablation_report),
        ),
        _criterion(
            "output_action_not_always_early_or_never_used",
            _all_gate_checks(gates, "stage4_to_5", ["exit_distribution_present", "not_all_immediate_exit", "not_never_exit"]),
            _gate_evidence(gates, "stage4_to_5", ["status", "first_exit_step_histogram"]),
        ),
    ]
    return _phase("Proceed from BRIAN-R125 route-core to BRIAN-R350", criteria)


def _r350_decision(
    stage_gate_report: dict[str, Any],
    *,
    compute_report: dict[str, Any],
    reasoning_baseline_report: dict[str, Any],
    reasoning_candidates: list[dict[str, Any]],
    out_by_difficulty_report: dict[str, Any],
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    min_difficulty_step_correlation: float,
    min_reasoning_delta: float,
) -> dict[str, Any]:
    criteria = [
        _criterion(
            "same_active_compute_routed_not_worse_than_baseline",
            _compute_report_has_not_worse_candidate(compute_report),
            _compute_report_evidence(compute_report),
        ),
        _criterion(
            "reasoning_or_synthetic_multistep_improves",
            _reasoning_improved(reasoning_baseline_report, reasoning_candidates, min_reasoning_delta),
            _reasoning_evidence(reasoning_baseline_report, reasoning_candidates),
        ),
        _criterion(
            "difficulty_step_correlation_positive",
            _difficulty_positive(stage_gate_report, min_difficulty_step_correlation),
            _difficulty_evidence(stage_gate_report, min_difficulty_step_correlation),
        ),
        _criterion(
            "out_action_reduces_compute_on_easy_samples",
            _out_by_difficulty_passed(out_by_difficulty_report),
            _out_by_difficulty_evidence(out_by_difficulty_report),
        ),
        _criterion(
            "global_kv_long_context_benefit_if_tested",
            _global_kv_memory_quality_benefit(long_context_compare_report, global_kv_ablation_report),
            _global_kv_evidence(long_context_compare_report, global_kv_ablation_report),
        ),
    ]
    return _phase("Proceed from BRIAN-R350 to 1B/global serious validation", criteria)


def _r1b_success_decision(
    stage_gate_report: dict[str, Any],
    *,
    compute_report: dict[str, Any],
    reasoning_baseline_report: dict[str, Any],
    reasoning_candidates: list[dict[str, Any]],
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    max_compute_adjusted_loss_delta: float,
    min_reasoning_delta: float,
    min_visible_cot_reduction: float,
    max_reasoning_drop_for_cot: float,
    max_global_kv_cache_capacity_ratio: float,
    max_inference_latency_ratio: float,
) -> dict[str, Any]:
    core_advantages = _r1b_core_advantages(
        compute_report=compute_report,
        reasoning_baseline_report=reasoning_baseline_report,
        reasoning_candidates=reasoning_candidates,
        long_context_compare_report=long_context_compare_report,
        global_kv_ablation_report=global_kv_ablation_report,
        max_compute_adjusted_loss_delta=max_compute_adjusted_loss_delta,
        min_reasoning_delta=min_reasoning_delta,
        min_visible_cot_reduction=min_visible_cot_reduction,
        max_reasoning_drop_for_cot=max_reasoning_drop_for_cot,
    )
    criteria = [
        _criterion(
            "routing_does_not_collapse",
            _routing_not_collapsed(stage_gate_report),
            _routing_not_collapsed_evidence(stage_gate_report),
        ),
        _criterion(
            "compute_adjusted_eval_present",
            _compute_adjusted_eval_present(compute_report),
            _compute_adjusted_evidence(compute_report, max_compute_adjusted_loss_delta),
        ),
        _criterion(
            "kv_memory_remains_controlled",
            _kv_memory_controlled(
                long_context_compare_report,
                global_kv_ablation_report,
                max_global_kv_cache_capacity_ratio,
            ),
            _kv_memory_evidence(
                long_context_compare_report,
                global_kv_ablation_report,
                max_global_kv_cache_capacity_ratio,
            ),
        ),
        _criterion(
            "inference_latency_remains_acceptable",
            _inference_latency_acceptable(compute_report, max_inference_latency_ratio),
            _inference_latency_evidence(compute_report, max_inference_latency_ratio),
        ),
        _criterion(
            "at_least_one_core_advantage_stable",
            _at_least_one_core_advantage(core_advantages),
            core_advantages,
        ),
    ]
    return _phase("Declare BRIAN-R1B/core validation successful", criteria)


def _phase(description: str, criteria: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [item["status"] for item in criteria]
    status = _overall_status(statuses)
    recommendation = {"pass": "proceed", "warn": "hold", "fail": "stop"}[status]
    return {
        "description": description,
        "status": status,
        "recommendation": recommendation,
        "criteria": criteria,
    }


def _criterion(name: str, passed: bool | None, evidence: Any) -> dict[str, Any]:
    if passed is True:
        status = "pass"
    elif passed is False:
        status = "fail"
    else:
        status = "missing"
    return {"name": name, "status": status, "passed": passed, "evidence": evidence}


def _gate_check(gates: dict[str, Any], gate_name: str, check_name: str) -> bool | None:
    gate = gates.get(gate_name)
    if not isinstance(gate, dict):
        return None
    checks = gate.get("checks", {})
    if not isinstance(checks, dict) or check_name not in checks:
        return None
    return bool(checks[check_name])


def _all_gate_checks(gates: dict[str, Any], gate_name: str, check_names: list[str]) -> bool | None:
    values = [_gate_check(gates, gate_name, check_name) for check_name in check_names]
    if any(value is None for value in values):
        return None
    return all(bool(value) for value in values)


def _any_gate_checks(gates: dict[str, Any], gate_names: list[str], check_names: list[str]) -> bool | None:
    values = []
    for gate_name in gate_names:
        for check_name in check_names:
            value = _gate_check(gates, gate_name, check_name)
            if value is not None:
                values.append(value)
    if not values:
        return None
    return any(values)


def _gate_evidence(gates: dict[str, Any], gate_name: str, keys: list[str]) -> dict[str, Any]:
    gate = gates.get(gate_name)
    if not isinstance(gate, dict):
        return {}
    evidence = {key: gate.get(key) for key in keys}
    if isinstance(gate.get("checks"), dict):
        evidence["checks"] = gate["checks"]
    return evidence


def _report_passed(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    if report.get("overall_status") == "pass" or report.get("status") == "pass":
        return True
    if report.get("overall_status") in {"fail", "warn"} or report.get("status") in {"fail", "warn"}:
        return False
    checks = report.get("checks")
    if isinstance(checks, dict):
        return all(bool(value) for value in checks.values())
    return None


def _report_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "overall_status": report.get("overall_status"),
        "status": report.get("status"),
        "candidate_count": report.get("candidate_count"),
        "checks": report.get("checks"),
    }


def _out_by_difficulty_passed(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    if report.get("overall_status") != "pass":
        return False
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return False
    required = [
        "easy_and_hard_present",
        "route_steps_non_decreasing_with_difficulty",
        "active_compute_non_decreasing_with_difficulty",
        "easy_output_probability_at_least_hard",
    ]
    return all(checks.get(key) is True for key in required)


def _out_by_difficulty_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        **_report_evidence(report),
        "deltas": report.get("deltas"),
        "by_difficulty": report.get("by_difficulty"),
    }


def _position_ablation_passed(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    if report.get("overall_status") != "pass":
        return False
    checks = report.get("checks")
    if not isinstance(checks, dict):
        return False
    return checks.get("candidate_present") is True and checks.get("any_measurable_difference") is True


def _position_ablation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        **_report_evidence(report),
        "comparisons": report.get("comparisons"),
    }


def _global_kv_memory_quality_benefit(
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
) -> bool | None:
    outcomes = [
        _long_context_compare_has_memory_quality_candidate(long_context_compare_report),
        _global_kv_ablation_has_memory_quality_candidate(global_kv_ablation_report),
    ]
    outcomes = [outcome for outcome in outcomes if outcome is not None]
    if not outcomes:
        return None
    return any(outcomes)


def _global_kv_evidence(
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "long_context_compare": {
            **_report_evidence(long_context_compare_report),
            "benefit_candidates": _long_context_benefit_candidates(long_context_compare_report),
        },
        "global_kv_ablation": {
            **_report_evidence(global_kv_ablation_report),
            "benefit_candidates": _global_kv_ablation_benefit_candidates(global_kv_ablation_report),
        },
    }


def _long_context_compare_has_memory_quality_candidate(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    if report.get("overall_status") != "pass":
        return False
    candidates = _long_context_benefit_candidates(report)
    if not candidates:
        return False
    return any(candidate["passes_memory_quality_proxy"] for candidate in candidates)


def _long_context_benefit_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("comparisons", []) if report else []
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        checks = row.get("checks", {}) if isinstance(row.get("checks"), dict) else {}
        passes = (
            row.get("status") == "pass"
            and checks.get("global_kv_active") is True
            and checks.get("quality_not_worse") is True
            and checks.get("memory_budget_present") is True
            and checks.get("global_budget_below_local_context") is True
        )
        candidates.append(
            {
                "candidate_report": row.get("candidate_report"),
                "candidate_run_dir": row.get("candidate_run_dir"),
                "status": row.get("status"),
                "global_kv_active": checks.get("global_kv_active"),
                "quality_not_worse": checks.get("quality_not_worse"),
                "memory_budget_present": checks.get("memory_budget_present"),
                "global_budget_below_local_context": checks.get("global_budget_below_local_context"),
                "passes_memory_quality_proxy": passes,
            }
        )
    return candidates


def _global_kv_ablation_has_memory_quality_candidate(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    if report.get("overall_status") != "pass":
        return False
    candidates = _global_kv_ablation_benefit_candidates(report)
    if not candidates:
        return False
    return any(candidate["passes_memory_quality_proxy"] for candidate in candidates)


def _global_kv_ablation_benefit_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = report.get("comparisons", {}) if report else {}
    rows = comparisons.get("local_vs_global", []) if isinstance(comparisons, dict) else []
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        capacity_ratio = _num(row.get("global_cache_capacity_ratio"))
        exact_delta = _num(row.get("exact_match_delta_vs_local"))
        teacher_delta = _num(row.get("teacher_forced_token_accuracy_delta_vs_local"))
        passes = (
            capacity_ratio is not None
            and capacity_ratio < 1.0
            and exact_delta is not None
            and exact_delta >= 0.0
            and teacher_delta is not None
            and teacher_delta >= 0.0
        )
        candidates.append(
            {
                "entry_id": row.get("entry_id"),
                "entry_name": row.get("entry_name"),
                "run_dir": row.get("run_dir"),
                "global_cache_capacity_ratio": capacity_ratio,
                "exact_match_delta_vs_local": exact_delta,
                "teacher_forced_token_accuracy_delta_vs_local": teacher_delta,
                "passes_memory_quality_proxy": passes,
            }
        )
    return candidates


def _r1b_core_advantages(
    *,
    compute_report: dict[str, Any],
    reasoning_baseline_report: dict[str, Any],
    reasoning_candidates: list[dict[str, Any]],
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    max_compute_adjusted_loss_delta: float,
    min_reasoning_delta: float,
    min_visible_cot_reduction: float,
    max_reasoning_drop_for_cot: float,
) -> dict[str, Any]:
    return {
        "better_compute_adjusted_perplexity": {
            "passed": _better_compute_adjusted_perplexity(compute_report, max_compute_adjusted_loss_delta),
            "evidence": _compute_adjusted_evidence(compute_report, max_compute_adjusted_loss_delta),
        },
        "better_reasoning_accuracy": {
            "passed": _reasoning_improved(reasoning_baseline_report, reasoning_candidates, min_reasoning_delta),
            "evidence": _reasoning_evidence(reasoning_baseline_report, reasoning_candidates),
        },
        "better_long_context_memory_efficiency": {
            "passed": _global_kv_memory_quality_benefit(long_context_compare_report, global_kv_ablation_report),
            "evidence": _global_kv_evidence(long_context_compare_report, global_kv_ablation_report),
        },
        "less_visible_cot_for_similar_reasoning": {
            "passed": _visible_cot_reduced(
                reasoning_baseline_report,
                reasoning_candidates,
                min_visible_cot_reduction=min_visible_cot_reduction,
                max_reasoning_drop=max_reasoning_drop_for_cot,
            ),
            "evidence": _visible_cot_evidence(reasoning_baseline_report, reasoning_candidates),
        },
    }


def _at_least_one_core_advantage(core_advantages: dict[str, Any]) -> bool | None:
    values = [item.get("passed") for item in core_advantages.values() if isinstance(item, dict)]
    if any(value is True for value in values):
        return True
    if any(value is False for value in values):
        return False
    return None


def _routing_not_collapsed(stage_gate_report: dict[str, Any]) -> bool | None:
    gates = stage_gate_report.get("gates", {})
    checks = [
        _gate_check(gates, "stage2_to_3", "block_usage_non_degenerate"),
        _gate_check(gates, "stage2_to_3", "block_load_entropy_present"),
        _gate_check(gates, "stage3_to_4", "route_entropy_present"),
        _gate_check(gates, "stage3_to_4", "route_path_diversity_present"),
        _gate_check(gates, "stage5_to_6", "global_attention_mass_nonzero"),
        _gate_check(gates, "stage6_to_scale", "parallel_branch_count_present"),
    ]
    checks = [check for check in checks if check is not None]
    if not checks:
        return None
    return all(checks)


def _routing_not_collapsed_evidence(stage_gate_report: dict[str, Any]) -> dict[str, Any]:
    gates = stage_gate_report.get("gates", {})
    return {
        "stage2_to_3": _gate_evidence(gates, "stage2_to_3", ["status"]),
        "stage3_to_4": _gate_evidence(gates, "stage3_to_4", ["status"]),
        "stage5_to_6": _gate_evidence(gates, "stage5_to_6", ["status"]),
        "stage6_to_scale": _gate_evidence(gates, "stage6_to_scale", ["status"]),
    }


def _kv_memory_controlled(
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    max_global_kv_cache_capacity_ratio: float,
) -> bool | None:
    candidates = [
        *_long_context_memory_candidates(long_context_compare_report, max_global_kv_cache_capacity_ratio),
        *_global_kv_ablation_memory_candidates(global_kv_ablation_report, max_global_kv_cache_capacity_ratio),
    ]
    if not candidates:
        return None
    if all(candidate["global_cache_capacity_ratio"] is None for candidate in candidates):
        return None
    return any(candidate["passes_memory_control_proxy"] for candidate in candidates)


def _kv_memory_evidence(
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    max_global_kv_cache_capacity_ratio: float,
) -> dict[str, Any]:
    return {
        "max_global_kv_cache_capacity_ratio": max_global_kv_cache_capacity_ratio,
        "long_context_compare": {
            **_report_evidence(long_context_compare_report),
            "memory_candidates": _long_context_memory_candidates(
                long_context_compare_report,
                max_global_kv_cache_capacity_ratio,
            ),
        },
        "global_kv_ablation": {
            **_report_evidence(global_kv_ablation_report),
            "memory_candidates": _global_kv_ablation_memory_candidates(
                global_kv_ablation_report,
                max_global_kv_cache_capacity_ratio,
            ),
        },
    }


def _long_context_memory_candidates(
    report: dict[str, Any],
    max_global_kv_cache_capacity_ratio: float,
) -> list[dict[str, Any]]:
    rows = report.get("comparisons", []) if report else []
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        memory_budget = row.get("memory_budget", {})
        candidate_memory = memory_budget.get("candidate", {}) if isinstance(memory_budget, dict) else {}
        checks = row.get("checks", {}) if isinstance(row.get("checks"), dict) else {}
        capacity_ratio = _num(candidate_memory.get("estimated_global_cache_capacity_to_local_context_ratio"))
        passes = (
            capacity_ratio is not None
            and capacity_ratio <= max_global_kv_cache_capacity_ratio
            and bool(checks.get("memory_budget_present", True))
        )
        candidates.append(
            {
                "candidate_report": row.get("candidate_report"),
                "candidate_run_dir": row.get("candidate_run_dir"),
                "status": row.get("status"),
                "global_cache_capacity_ratio": capacity_ratio,
                "memory_budget_present": checks.get("memory_budget_present"),
                "global_budget_below_local_context": checks.get("global_budget_below_local_context"),
                "passes_memory_control_proxy": passes,
            }
        )
    return candidates


def _global_kv_ablation_memory_candidates(
    report: dict[str, Any],
    max_global_kv_cache_capacity_ratio: float,
) -> list[dict[str, Any]]:
    comparisons = report.get("comparisons", {}) if report else {}
    rows = comparisons.get("local_vs_global", []) if isinstance(comparisons, dict) else []
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        capacity_ratio = _num(row.get("global_cache_capacity_ratio"))
        candidates.append(
            {
                "entry_id": row.get("entry_id"),
                "entry_name": row.get("entry_name"),
                "run_dir": row.get("run_dir"),
                "global_cache_capacity_ratio": capacity_ratio,
                "passes_memory_control_proxy": capacity_ratio is not None
                and capacity_ratio <= max_global_kv_cache_capacity_ratio,
            }
        )
    return candidates


def _inference_latency_acceptable(report: dict[str, Any], max_inference_latency_ratio: float) -> bool | None:
    candidates = _inference_latency_candidates(report, max_inference_latency_ratio)
    if not candidates:
        return None
    if all(candidate["inference_latency_ms_per_token_ratio"] is None for candidate in candidates):
        return None
    return any(candidate["passes_latency_proxy"] for candidate in candidates)


def _inference_latency_evidence(report: dict[str, Any], max_inference_latency_ratio: float) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "baseline_run": report.get("baseline_run"),
        "max_inference_latency_ratio": max_inference_latency_ratio,
        "candidates": _inference_latency_candidates(report, max_inference_latency_ratio),
    }


def _inference_latency_candidates(
    report: dict[str, Any],
    max_inference_latency_ratio: float,
) -> list[dict[str, Any]]:
    if not report:
        return []
    baseline = _baseline_run_summary(report)
    baseline_latency = _num(baseline.get("inference_latency_ms_per_token_latest")) if baseline else None
    rows = report.get("runs", [])
    if not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("baseline_comparison"), dict):
            continue
        comparison = row["baseline_comparison"]
        latency = _num(row.get("inference_latency_ms_per_token_latest"))
        latency_ratio = _num(comparison.get("inference_latency_ms_per_token_ratio"))
        if latency_ratio is None:
            latency_ratio = _ratio(latency, baseline_latency)
        candidates.append(
            {
                "run_dir": row.get("run_dir"),
                "stage": row.get("stage"),
                "inference_latency_ms_per_token": latency,
                "baseline_inference_latency_ms_per_token": baseline_latency,
                "inference_latency_ms_per_token_ratio": latency_ratio,
                "tokens_per_second_ratio": _num(comparison.get("tokens_per_second_ratio")),
                "passes_latency_proxy": latency_ratio is not None and latency_ratio <= max_inference_latency_ratio,
            }
        )
    return candidates


def _compute_adjusted_eval_present(report: dict[str, Any]) -> bool | None:
    candidates = _compute_adjusted_candidates(report, max_compute_adjusted_loss_delta=0.0)
    if not candidates:
        return None
    return any(candidate["compute_adjusted_loss_delta"] is not None for candidate in candidates)


def _better_compute_adjusted_perplexity(report: dict[str, Any], max_compute_adjusted_loss_delta: float) -> bool | None:
    candidates = _compute_adjusted_candidates(report, max_compute_adjusted_loss_delta=max_compute_adjusted_loss_delta)
    if not candidates:
        return None
    return any(candidate["passes_compute_adjusted_loss_proxy"] for candidate in candidates)


def _compute_adjusted_evidence(report: dict[str, Any], max_compute_adjusted_loss_delta: float) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "run_count": report.get("run_count"),
        "baseline_run": report.get("baseline_run"),
        "max_compute_adjusted_loss_delta": max_compute_adjusted_loss_delta,
        "candidates": _compute_adjusted_candidates(
            report,
            max_compute_adjusted_loss_delta=max_compute_adjusted_loss_delta,
        ),
    }


def _compute_adjusted_candidates(
    report: dict[str, Any],
    *,
    max_compute_adjusted_loss_delta: float,
) -> list[dict[str, Any]]:
    if not report:
        return []
    baseline = _baseline_run_summary(report)
    baseline_loss = _num(baseline.get("validation_loss")) if baseline else None
    rows = report.get("runs", [])
    if baseline_loss is None or not isinstance(rows, list):
        return []
    candidates = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("baseline_comparison"), dict):
            continue
        comparison = row["baseline_comparison"]
        flops_ratio = _num(comparison.get("estimated_flops_per_token_ratio"))
        validation_loss = _num(row.get("validation_loss"))
        adjusted_delta = None
        if validation_loss is not None and flops_ratio is not None:
            adjusted_delta = validation_loss * flops_ratio - baseline_loss
        candidates.append(
            {
                "run_dir": row.get("run_dir"),
                "stage": row.get("stage"),
                "validation_loss": validation_loss,
                "baseline_validation_loss": baseline_loss,
                "estimated_flops_per_token_ratio": flops_ratio,
                "validation_loss_delta": _num(comparison.get("validation_loss_delta")),
                "compute_adjusted_loss_delta": adjusted_delta,
                "passes_compute_adjusted_loss_proxy": adjusted_delta is not None
                and adjusted_delta <= max_compute_adjusted_loss_delta,
            }
        )
    return candidates


def _baseline_run_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("runs", [])
    if not isinstance(rows, list):
        return {}
    baseline_run = report.get("baseline_run")
    for row in rows:
        if isinstance(row, dict) and baseline_run and row.get("run_dir") == baseline_run:
            return row
    for row in rows:
        if isinstance(row, dict) and not isinstance(row.get("baseline_comparison"), dict):
            return row
    return {}


def _visible_cot_reduced(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    min_visible_cot_reduction: float,
    max_reasoning_drop: float,
) -> bool | None:
    evidence = _visible_cot_evidence(baseline, candidates)
    rows = evidence.get("candidate_comparisons", [])
    if not rows:
        return None
    return any(
        row.get("visible_cot_token_delta") is not None
        and row["visible_cot_token_delta"] <= -min_visible_cot_reduction
        and row.get("reasoning_score_delta") is not None
        and row["reasoning_score_delta"] >= -max_reasoning_drop
        for row in rows
    )


def _visible_cot_evidence(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_score = _reasoning_score(baseline) if baseline else None
    baseline_cot = _visible_cot_tokens(baseline) if baseline else None
    comparisons = []
    for candidate in candidates:
        candidate_score = _reasoning_score(candidate)
        candidate_cot = _visible_cot_tokens(candidate)
        comparisons.append(
            {
                "run_dir": candidate.get("run_dir"),
                "baseline_reasoning_score": baseline_score,
                "candidate_reasoning_score": candidate_score,
                "reasoning_score_delta": _delta(candidate_score, baseline_score),
                "baseline_visible_cot_tokens": baseline_cot,
                "candidate_visible_cot_tokens": candidate_cot,
                "visible_cot_token_delta": _delta(candidate_cot, baseline_cot),
            }
        )
    return {
        "baseline_visible_cot_tokens": baseline_cot,
        "candidate_comparisons": comparisons,
    }


def _visible_cot_tokens(report: dict[str, Any]) -> float | None:
    overall = report.get("overall", {}) if isinstance(report, dict) else {}
    if not isinstance(overall, dict):
        return None
    for key in (
        "visible_cot_tokens_mean",
        "mean_visible_cot_tokens",
        "visible_cot_token_count_mean",
        "cot_tokens_mean",
        "visible_reasoning_tokens_mean",
    ):
        value = _num(overall.get(key))
        if value is not None:
            return value
    return None


def _compute_report_has_not_worse_candidate(report: dict[str, Any]) -> bool | None:
    runs = report.get("runs") if report else None
    if not isinstance(runs, list):
        return None
    found = False
    for run in runs:
        comparison = run.get("baseline_comparison") if isinstance(run, dict) else None
        if not isinstance(comparison, dict):
            continue
        found = True
        same_parameter_count = comparison.get("same_parameter_count_view") is True
        same_active = comparison.get("same_active_compute_view") is True
        similar_flops = comparison.get("similar_training_flops_view") is True
        loss_delta = _num(comparison.get("validation_loss_delta"))
        if same_parameter_count and same_active and similar_flops and loss_delta is not None and loss_delta <= 0.0:
            return True
    return False if found else None


def _compute_report_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    rows = []
    for run in report.get("runs", []):
        if isinstance(run, dict) and isinstance(run.get("baseline_comparison"), dict):
            rows.append(
                {
                    "run_dir": run.get("run_dir"),
                    "stage": run.get("stage"),
                    "validation_loss": run.get("validation_loss"),
                    "baseline_comparison": run.get("baseline_comparison"),
                }
            )
    return {"run_count": report.get("run_count"), "baseline_run": report.get("baseline_run"), "comparisons": rows}


def _parallel_compare_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    comparisons = []
    for row in report.get("comparisons", []):
        if not isinstance(row, dict):
            continue
        comparisons.append(
            {
                "candidate_run": row.get("candidate_run"),
                "status": row.get("status"),
                "checks": row.get("checks"),
                "parallel": row.get("parallel"),
                "baseline_comparison": row.get("baseline_comparison"),
            }
        )
    return {
        "overall_status": report.get("overall_status"),
        "candidate_count": report.get("candidate_count"),
        "comparisons": comparisons,
    }


def _reasoning_improved(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    min_delta: float,
) -> bool | None:
    if not baseline or not candidates:
        return None
    baseline_score = _reasoning_score(baseline)
    candidate_scores = [_reasoning_score(candidate) for candidate in candidates]
    candidate_scores = [score for score in candidate_scores if score is not None]
    if baseline_score is None or not candidate_scores:
        return None
    return max(candidate_scores) - baseline_score >= min_delta


def _reasoning_score(report: dict[str, Any]) -> float | None:
    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        return None
    exact = overall.get("exact_match_accuracy")
    teacher = overall.get("teacher_forced_token_accuracy")
    exact_score = _num(exact)
    teacher_score = _num(teacher)
    if exact_score is not None:
        return exact_score
    if teacher_score is not None:
        return teacher_score
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _delta(value: Any, baseline: Any) -> float | None:
    left = _num(value)
    right = _num(baseline)
    if left is None or right is None:
        return None
    return left - right


def _ratio(value: Any, baseline: Any) -> float | None:
    left = _num(value)
    right = _num(baseline)
    if left is None or right is None or right == 0.0:
        return None
    return left / right


def _reasoning_evidence(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "baseline_score": _reasoning_score(baseline) if baseline else None,
        "candidate_scores": [_reasoning_score(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
    }


def _difficulty_positive(stage_gate_report: dict[str, Any], minimum: float) -> bool | None:
    values = _difficulty_values(stage_gate_report)
    if not values:
        return None
    return any(row["difficulty_step_correlation"] > minimum for row in values)


def _difficulty_evidence(stage_gate_report: dict[str, Any], minimum: float) -> dict[str, Any]:
    return {
        "threshold": minimum,
        "runs": _difficulty_values(stage_gate_report),
    }


def _difficulty_values(stage_gate_report: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for run in stage_gate_report.get("runs", []):
        if not isinstance(run, dict):
            continue
        value = _num(run.get("difficulty_step_correlation"))
        if value is not None:
            values.append(
                {
                    "run_dir": run.get("run_dir"),
                    "stage": run.get("stage"),
                    "difficulty_step_correlation": value,
                }
            )
    return values


def _overall_status(statuses: list[str]) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status in {"missing", "warn"} for status in statuses):
        return "warn"
    return "pass"


def _recommendation(recommendations: list[str]) -> str:
    if "stop" in recommendations:
        return "stop"
    if "hold" in recommendations:
        return "hold"
    return "proceed"


def _read_json_if_present(path: str | Path | None) -> dict[str, Any]:
    return _read_json(Path(path)) if path else {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data
