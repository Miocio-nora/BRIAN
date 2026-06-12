from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


DEFAULT_THRESHOLDS = {
    "min_block_load_entropy_normalized": 0.05,
    "min_route_path_diversity": 0.05,
    "max_exit_dominance": 0.95,
    "min_output_probability": 0.01,
    "max_free_route_loss_ratio": 1.20,
    "min_position_trajectory_range": 1e-6,
    "min_location_distance_range": 1e-6,
    "min_location_distance_mean": 1e-6,
    "min_global_attention_mass": 1e-6,
    "min_global_kv_quality_delta": 1e-9,
    "max_global_kv_loss_delta": 0.0,
}


PARALLEL_COMPARE_ROLE_CHECKS = [
    "baseline_stage5_global_kv",
    "baseline_scheduled_route_mode",
    "baseline_global_kv_enabled",
    "baseline_parallel_passing_disabled",
    "baseline_topk_weighted_fusion",
    "candidate_parallel_stage",
    "candidate_parallel_route_mode",
    "candidate_parallel_passing_enabled",
    "candidate_global_kv_enabled",
]


LONG_CONTEXT_STAGE5_ROLE_CHECKS = [
    "baseline_stage4_output_action",
    "baseline_scheduled_route_mode",
    "baseline_local_kv",
    "candidate_stage5_global_kv",
    "candidate_scheduled_route_mode",
    "candidate_global_kv_enabled",
]

LONG_CONTEXT_COVERAGE_CHECKS = [
    "baseline_task_family_coverage",
    "baseline_difficulty_coverage",
    "candidate_task_family_coverage",
    "candidate_difficulty_coverage",
]


MITIGATIONS = {
    "router_collapse": [
        "extend pseudo-route imitation",
        "increase balance loss",
        "lower cost loss if early exit",
        "increase cost loss if never exit",
        "delay hard OUT behavior",
        "increase location bias early",
    ],
    "block_position_state_no_effect": [
        "feed position to both router and blocks",
        "increase position adapter strength gradually",
        "add location loss",
        "compare random vs open-arc initialization",
        "check whether normalization layers erase position signal",
    ],
    "free_routing_lm_loss_degrades": [
        "slower scheduled routing",
        "teacher/logit distillation from fixed baseline",
        "top-1 before top-2",
        "fixed min/max route steps",
        "lower router temperature gradually",
    ],
    "global_kv_noise": [
        "initialize global read gate near zero",
        "train global adapters after route core",
        "limit write frequency",
        "use simple window + sink first",
        "start with per-block adapters before full per-head adapters",
    ],
    "parallel_passing_cost_explosion": [
        "postpone until route core works",
        "beam <= 2 initially",
        "branch score decay",
        "branch cost loss",
        "shared base memory + branch delta memory",
    ],
}


def make_risk_audit_report(
    *,
    output_path: str | Path | None = None,
    stage_gate_report_path: str | Path | None = None,
    routing_report_path: str | Path | None = None,
    position_ablation_report_path: str | Path | None = None,
    compute_report_path: str | Path | None = None,
    long_context_compare_report_path: str | Path | None = None,
    global_kv_retention_report_path: str | Path | None = None,
    global_kv_ablation_report_path: str | Path | None = None,
    parallel_compare_report_path: str | Path | None = None,
    parallel_passing_report_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> Path:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    report_paths = {
        "stage_gate_report": stage_gate_report_path,
        "routing_report": routing_report_path,
        "position_ablation_report": position_ablation_report_path,
        "compute_report": compute_report_path,
        "long_context_compare_report": long_context_compare_report_path,
        "global_kv_retention_report": global_kv_retention_report_path,
        "global_kv_ablation_report": global_kv_ablation_report_path,
        "parallel_compare_report": parallel_compare_report_path,
        "parallel_passing_report": parallel_passing_report_path,
    }
    reports = {name: _read_json_if_present(path) for name, path in report_paths.items()}
    risks = {
        "router_collapse": _router_collapse(reports, thresholds),
        "block_position_state_no_effect": _block_position_state_no_effect(reports, thresholds),
        "free_routing_lm_loss_degrades": _free_routing_lm_loss_degrades(reports, thresholds),
        "global_kv_noise": _global_kv_noise(reports, thresholds),
        "parallel_passing_cost_explosion": _parallel_passing_cost_explosion(reports),
    }
    report = {
        "source_plan_section": "20 Risk Register",
        "overall_status": _overall_status(risks),
        "risks": risks,
        "thresholds": thresholds,
        "inputs": {
            name: {"path": str(path) if path else None, "present": bool(reports[name])}
            for name, path in report_paths.items()
        },
    }
    if output_path is None:
        output_path = Path("reports") / "risk_audit_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _router_collapse(reports: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    routing_report = reports["routing_report"]
    stage_gate_report = reports["stage_gate_report"]
    summary = _dict(routing_report.get("summary"))
    exit_hist = _first_exit_histogram(routing_report, stage_gate_report)

    block_entropy = _num(summary.get("block_load_entropy_normalized"))
    block_check = _gate_check(stage_gate_report, "stage2_to_3", "block_usage_non_degenerate")
    same_block = _below_threshold_or_failed_check(
        block_entropy,
        thresholds["min_block_load_entropy_normalized"],
        block_check,
    )

    path_diversity = _num(summary.get("route_path_diversity"))
    path_check = _gate_check(stage_gate_report, "stage3_to_4", "route_path_diversity_present")
    original_sequence = _below_threshold_or_failed_check(
        path_diversity,
        thresholds["min_route_path_diversity"],
        path_check,
    )

    early_fraction = _hist_fraction(exit_hist, "1")
    not_immediate = _gate_check(stage_gate_report, "stage4_to_5", "not_all_immediate_exit")
    if early_fraction is not None:
        exits_early = early_fraction >= thresholds["max_exit_dominance"]
    elif not_immediate is not None:
        exits_early = not not_immediate
    else:
        exits_early = None

    never_exit_fraction = _hist_fraction(exit_hist, "0")
    p_output = _num(summary.get("p_output_mean"))
    not_never_exit = _gate_check(stage_gate_report, "stage4_to_5", "not_never_exit")
    if never_exit_fraction is not None:
        never_exits = never_exit_fraction >= thresholds["max_exit_dominance"]
    elif not_never_exit is not None:
        never_exits = not not_never_exit
    elif p_output is not None:
        never_exits = p_output <= thresholds["min_output_probability"]
    else:
        never_exits = None

    symptoms = [
        _symptom(
            "always_selects_same_block",
            same_block,
            {
                "block_load_entropy_normalized": block_entropy,
                "threshold": thresholds["min_block_load_entropy_normalized"],
                "stage2_block_usage_non_degenerate": block_check,
            },
        ),
        _symptom(
            "always_follows_original_sequence",
            original_sequence,
            {
                "route_path_diversity": path_diversity,
                "threshold": thresholds["min_route_path_diversity"],
                "stage3_route_path_diversity_present": path_check,
            },
        ),
        _symptom(
            "always_exits_early",
            exits_early,
            {
                "first_exit_step_histogram": exit_hist,
                "step_1_fraction": early_fraction,
                "threshold": thresholds["max_exit_dominance"],
                "stage4_not_all_immediate_exit": not_immediate,
            },
        ),
        _symptom(
            "never_exits",
            never_exits,
            {
                "first_exit_step_histogram": exit_hist,
                "step_0_fraction": never_exit_fraction,
                "p_output_mean": p_output,
                "min_output_probability": thresholds["min_output_probability"],
                "stage4_not_never_exit": not_never_exit,
            },
        ),
    ]
    return _risk("Router collapse", symptoms, MITIGATIONS["router_collapse"])


def _block_position_state_no_effect(
    reports: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    routing_report = reports["routing_report"]
    position_report = reports["position_ablation_report"]
    summary = _dict(routing_report.get("summary"))

    no_position_equals = _no_position_ablation_equals_main(position_report)
    position_values = _num_list(routing_report.get("latest_position_norm_trajectory"))
    position_range = _range(position_values)
    position_constant = (
        position_range <= thresholds["min_position_trajectory_range"]
        if position_range is not None and len(position_values) >= 2
        else None
    )

    location_values = _num_list(routing_report.get("latest_location_distance_trajectory"))
    location_range = _range(location_values)
    location_mean = _num(summary.get("location_distance_mean"))
    if location_range is not None and len(location_values) >= 2:
        location_no_structure = location_range <= thresholds["min_location_distance_range"]
    elif location_mean is not None:
        location_no_structure = location_mean <= thresholds["min_location_distance_mean"]
    else:
        location_no_structure = None

    symptoms = [
        _symptom(
            "no_position_ablation_equals_main_model",
            no_position_equals,
            {
                "overall_status": position_report.get("overall_status") if position_report else None,
                "candidate_count": position_report.get("candidate_count") if position_report else None,
                "checks": position_report.get("checks", {}) if position_report else {},
            },
        ),
        _symptom(
            "position_state_becomes_constant",
            position_constant,
            {
                "position_norm_trajectory_count": len(position_values),
                "position_norm_range": position_range,
                "threshold": thresholds["min_position_trajectory_range"],
            },
        ),
        _symptom(
            "location_distance_has_no_structure",
            location_no_structure,
            {
                "location_distance_trajectory_count": len(location_values),
                "location_distance_range": location_range,
                "location_distance_mean": location_mean,
                "range_threshold": thresholds["min_location_distance_range"],
                "mean_threshold": thresholds["min_location_distance_mean"],
            },
        ),
    ]
    return _risk(
        "Block-position state has no effect",
        symptoms,
        MITIGATIONS["block_position_state_no_effect"],
    )


def _free_routing_lm_loss_degrades(
    reports: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    stage_gate_report = reports["stage_gate_report"]
    fixed_route_works = _fixed_route_works(stage_gate_report)
    free_route_spike = _free_route_loss_spike(stage_gate_report, thresholds["max_free_route_loss_ratio"])
    gate = _gate(stage_gate_report, "stage3_to_4")
    symptoms = [
        _symptom(
            "fixed_route_works_but_free_route_validation_loss_spikes",
            free_route_spike,
            {
                "fixed_route_works": fixed_route_works,
                "stage3_validation_loss_not_collapsed": _gate_check(
                    stage_gate_report,
                    "stage3_to_4",
                    "validation_loss_not_collapsed",
                ),
                "loss_ratio_vs_stage1": _num(gate.get("loss_ratio_vs_stage1")),
                "max_free_route_loss_ratio": thresholds["max_free_route_loss_ratio"],
            },
        )
    ]
    return _risk(
        "Free routing degrades LM loss",
        symptoms,
        MITIGATIONS["free_routing_lm_loss_degrades"],
    )


def _global_kv_noise(reports: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    routing_report = reports["routing_report"]
    stage_gate_report = reports["stage_gate_report"]
    retention_report = reports["global_kv_retention_report"]
    long_context_compare_report = reports["long_context_compare_report"]
    ablation_report = reports["global_kv_ablation_report"]
    routing_summary = _dict(routing_report.get("summary"))
    retention_metrics = _dict(retention_report.get("metrics"))
    attention_mass = _first_num(
        retention_metrics.get("global_attention_mass"),
        routing_summary.get("global_attention_mass"),
    )
    retention_check = _report_check(retention_report, "global_attention_mass_nonzero")
    stage_check = _gate_check(stage_gate_report, "stage5_to_6", "global_attention_mass_nonzero")
    if attention_mass is not None:
        mass_near_zero = attention_mass < thresholds["min_global_attention_mass"]
    elif retention_check is not None:
        mass_near_zero = not retention_check
    elif stage_check is not None:
        mass_near_zero = not stage_check
    else:
        mass_near_zero = None

    no_difference, no_difference_evidence = _global_on_off_no_difference(
        long_context_compare_report,
        ablation_report,
        thresholds["min_global_kv_quality_delta"],
    )
    worsens_loss, worsens_loss_evidence = _global_cache_worsens_loss(
        ablation_report,
        thresholds["max_global_kv_loss_delta"],
    )

    symptoms = [
        _symptom(
            "global_attention_mass_near_zero",
            mass_near_zero,
            {
                "global_attention_mass": attention_mass,
                "threshold": thresholds["min_global_attention_mass"],
                "retention_global_attention_mass_nonzero": retention_check,
                "stage5_global_attention_mass_nonzero": stage_check,
            },
        ),
        _symptom("global_on_off_no_difference", no_difference, no_difference_evidence),
        _symptom("global_cache_worsens_loss", worsens_loss, worsens_loss_evidence),
    ]
    return _risk("Global KV becomes noise", symptoms, MITIGATIONS["global_kv_noise"])


def _parallel_passing_cost_explosion(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing_report = reports["parallel_passing_report"]
    compare_report = reports["parallel_compare_report"]
    branch_bounded = _report_check(passing_report, "branch_count_bounded_by_beam")
    delta_cache_bounded = _report_check(passing_report, "delta_cache_bounded_by_window")
    score_margin_measured = _report_check(passing_report, "score_margin_measured")
    score_decay_configured = _report_check(passing_report, "branch_score_decay_configured")
    compare_benefit, compare_benefit_evidence = _parallel_compare_benefit(compare_report)

    if score_decay_configured is False:
        credit_unstable = True
    elif score_margin_measured is False:
        credit_unstable = True
    elif compare_benefit is False:
        credit_unstable = True
    elif score_decay_configured is True and (score_margin_measured is True or compare_benefit is True):
        credit_unstable = False
    else:
        credit_unstable = None

    symptoms = [
        _symptom(
            "branch_count_grows",
            None if branch_bounded is None else not branch_bounded,
            {
                "branch_count_bounded_by_beam": branch_bounded,
                "model": passing_report.get("model", {}) if passing_report else {},
                "routing": passing_report.get("routing", {}) if passing_report else {},
            },
        ),
        _symptom(
            "memory_usage_grows_uncontrollably",
            None if delta_cache_bounded is None else not delta_cache_bounded,
            {
                "delta_cache_bounded_by_window": delta_cache_bounded,
                "model": passing_report.get("model", {}) if passing_report else {},
                "routing": passing_report.get("routing", {}) if passing_report else {},
            },
        ),
        _symptom(
            "branch_credit_assignment_unstable",
            credit_unstable,
            {
                "branch_score_decay_configured": score_decay_configured,
                "score_margin_measured": score_margin_measured,
                "parallel_compare_benefit_proxy": compare_benefit,
                "parallel_compare_status": compare_report.get("overall_status") if compare_report else None,
                "parallel_compare_candidates": compare_benefit_evidence,
            },
        ),
    ]
    return _risk(
        "Parallel passing explodes cost",
        symptoms,
        MITIGATIONS["parallel_passing_cost_explosion"],
    )


def _risk(description: str, symptoms: list[dict[str, Any]], mitigations: list[str]) -> dict[str, Any]:
    return {
        "description": description,
        "status": _risk_status(symptoms),
        "symptoms": symptoms,
        "mitigations": mitigations,
    }


def _symptom(name: str, triggered: bool | None, evidence: dict[str, Any]) -> dict[str, Any]:
    if triggered is True:
        status = "triggered"
    elif triggered is False:
        status = "clear"
    else:
        status = "missing"
    return {
        "name": name,
        "status": status,
        "triggered": triggered,
        "evidence": evidence,
    }


def _risk_status(symptoms: list[dict[str, Any]]) -> str:
    triggered = [symptom.get("triggered") for symptom in symptoms]
    if any(value is True for value in triggered):
        return "fail"
    if any(value is None for value in triggered):
        return "warn"
    return "pass"


def _overall_status(risks: dict[str, dict[str, Any]]) -> str:
    statuses = [risk["status"] for risk in risks.values()]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _below_threshold_or_failed_check(value: float | None, threshold: float, check: bool | None) -> bool | None:
    if value is not None:
        return value < threshold
    if check is not None:
        return not check
    return None


def _no_position_ablation_equals_main(report: dict[str, Any]) -> bool | None:
    if not report:
        return None
    checks = _dict(report.get("checks"))
    candidate_count = _num(report.get("candidate_count"))
    candidate_present = checks.get("candidate_present")
    if candidate_present is False or candidate_count == 0:
        return None
    reference_position_enabled = checks.get("reference_position_enabled")
    no_position_present = checks.get("no_position_candidate_present")
    if reference_position_enabled is False or no_position_present is False:
        return None
    if reference_position_enabled is not True or no_position_present is not True:
        return None
    valid_no_position_measurable = checks.get("any_valid_no_position_measurable_difference")
    if isinstance(valid_no_position_measurable, bool):
        return not valid_no_position_measurable
    return None


def _fixed_route_works(stage_gate_report: dict[str, Any]) -> bool | None:
    check = _gate_check(stage_gate_report, "stage1_to_2", "loss_within_1_to_3_percent")
    if check is not None:
        return check
    status = _gate(stage_gate_report, "stage1_to_2").get("status")
    if status == "pass":
        return True
    if status == "fail":
        return False
    return None


def _free_route_loss_spike(stage_gate_report: dict[str, Any], max_loss_ratio: float) -> bool | None:
    check = _gate_check(stage_gate_report, "stage3_to_4", "validation_loss_not_collapsed")
    if check is not None:
        return not check
    ratio = _num(_gate(stage_gate_report, "stage3_to_4").get("loss_ratio_vs_stage1"))
    if ratio is not None:
        return ratio > max_loss_ratio
    return None


def _global_on_off_no_difference(
    long_context_compare_report: dict[str, Any],
    global_kv_ablation_report: dict[str, Any],
    min_delta: float,
) -> tuple[bool | None, dict[str, Any]]:
    comparisons = _list(long_context_compare_report.get("comparisons"))
    if comparisons:
        candidates = _long_context_compare_candidates(comparisons)
        if any(candidate["passes_stage5_long_context_contract"] is True for candidate in candidates):
            return False, {
                "source": "long_context_compare_report",
                "overall_status": long_context_compare_report.get("overall_status"),
                "candidate_count": long_context_compare_report.get("candidate_count"),
                "passing_comparison_count": sum(
                    1 for candidate in candidates if candidate["passes_stage5_long_context_contract"] is True
                ),
                "comparison_candidates": candidates,
            }
        return True, {
            "source": "long_context_compare_report",
            "overall_status": long_context_compare_report.get("overall_status"),
            "candidate_count": long_context_compare_report.get("candidate_count"),
            "passing_comparison_count": 0,
            "comparison_candidates": candidates,
        }

    local_vs_global = _list(_dict(global_kv_ablation_report.get("comparisons")).get("local_vs_global"))
    deltas: list[float] = []
    for row in local_vs_global:
        if not isinstance(row, dict):
            continue
        deltas.extend(
            value
            for value in [
                _num(row.get("exact_match_delta_vs_local")),
                _num(row.get("teacher_forced_token_accuracy_delta_vs_local")),
            ]
            if value is not None
        )
    if deltas:
        flat = all(abs(value) <= min_delta for value in deltas)
        return flat, {
            "source": "global_kv_ablation_report",
            "quality_deltas": deltas,
            "min_global_kv_quality_delta": min_delta,
        }
    return None, {
        "source": None,
        "long_context_compare_present": bool(long_context_compare_report),
        "global_kv_ablation_present": bool(global_kv_ablation_report),
    }


def _long_context_compare_candidates(comparisons: list[Any]) -> list[dict[str, Any]]:
    candidates = []
    for row in comparisons:
        if not isinstance(row, dict):
            continue
        checks = _dict(row.get("checks"))
        role_checks = _selected_checks(checks, LONG_CONTEXT_STAGE5_ROLE_CHECKS)
        role_contract_passed = all(value is True for value in role_checks.values())
        coverage_checks = _selected_checks(checks, LONG_CONTEXT_COVERAGE_CHECKS)
        coverage_contract_passed = all(value is True for value in coverage_checks.values())
        stage5_contract = (
            row.get("status") == "pass"
            and role_contract_passed
            and coverage_contract_passed
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
                "role_checks": role_checks,
                "role_contract_passed": role_contract_passed,
                "coverage_checks": coverage_checks,
                "coverage_contract_passed": coverage_contract_passed,
                "global_kv_active": checks.get("global_kv_active"),
                "quality_not_worse": checks.get("quality_not_worse"),
                "memory_budget_present": checks.get("memory_budget_present"),
                "global_budget_below_local_context": checks.get("global_budget_below_local_context"),
                "passes_stage5_long_context_contract": stage5_contract,
            }
        )
    return candidates


def _global_cache_worsens_loss(
    global_kv_ablation_report: dict[str, Any],
    max_loss_delta: float,
) -> tuple[bool | None, dict[str, Any]]:
    local_vs_global = _list(_dict(global_kv_ablation_report.get("comparisons")).get("local_vs_global"))
    deltas = [
        value
        for row in local_vs_global
        if isinstance(row, dict)
        for value in [_num(row.get("validation_loss_delta_vs_local"))]
        if value is not None
    ]
    if not deltas:
        return None, {
            "global_kv_ablation_present": bool(global_kv_ablation_report),
            "validation_loss_deltas_vs_local": [],
            "max_global_kv_loss_delta": max_loss_delta,
        }
    return any(value > max_loss_delta for value in deltas), {
        "validation_loss_deltas_vs_local": deltas,
        "max_global_kv_loss_delta": max_loss_delta,
    }


def _parallel_compare_benefit(report: dict[str, Any]) -> tuple[bool | None, list[dict[str, Any]]]:
    comparisons = _list(report.get("comparisons"))
    if not comparisons:
        return None, []
    values = []
    evidence = []
    for row in comparisons:
        if not isinstance(row, dict):
            continue
        checks = _dict(row.get("checks"))
        role_checks = _selected_checks(checks, PARALLEL_COMPARE_ROLE_CHECKS)
        role_contract_passed = all(value is True for value in role_checks.values())
        benefit = checks.get("parallel_branch_benefit_proxy")
        value = benefit is True and role_contract_passed
        if isinstance(benefit, bool):
            values.append(value)
        elif row.get("status") == "fail":
            values.append(False)
        else:
            evidence.append(
                {
                    "candidate_run": row.get("candidate_run"),
                    "status": row.get("status"),
                    "role_checks": role_checks,
                    "role_contract_passed": role_contract_passed,
                    "parallel_branch_benefit_proxy": benefit,
                    "passes_parallel_compare_contract": None,
                }
            )
            continue
        evidence.append(
            {
                "candidate_run": row.get("candidate_run"),
                "status": row.get("status"),
                "role_checks": role_checks,
                "role_contract_passed": role_contract_passed,
                "parallel_branch_benefit_proxy": benefit,
                "passes_parallel_compare_contract": value,
            }
        )
    if not values:
        return None, evidence
    return any(values), evidence


def _first_exit_histogram(
    routing_report: dict[str, Any],
    stage_gate_report: dict[str, Any],
) -> dict[str, int]:
    hist = _hist(routing_report.get("latest_first_exit_step_histogram"))
    if hist:
        return hist
    stage_hist = _hist(_gate(stage_gate_report, "stage4_to_5").get("first_exit_step_histogram"))
    if stage_hist:
        return stage_hist
    return {}


def _hist(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    hist: dict[str, int] = {}
    for key, count in value.items():
        number = _num(count)
        if number is not None:
            hist[str(key)] = int(number)
    return hist


def _hist_fraction(hist: dict[str, int], key: str) -> float | None:
    if not hist:
        return None
    total = sum(count for count in hist.values() if count > 0)
    if total <= 0:
        return None
    return max(0, hist.get(key, 0)) / total


def _gate_check(report: dict[str, Any], gate_name: str, check_name: str) -> bool | None:
    value = _dict(_gate(report, gate_name).get("checks")).get(check_name)
    return value if isinstance(value, bool) else None


def _gate(report: dict[str, Any], gate_name: str) -> dict[str, Any]:
    return _dict(_dict(report.get("gates")).get(gate_name))


def _report_check(report: dict[str, Any], check_name: str) -> bool | None:
    value = _dict(report.get("checks")).get(check_name)
    return value if isinstance(value, bool) else None


def _selected_checks(checks: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: checks.get(name) for name in names}


def _first_num(*values: Any) -> float | None:
    for value in values:
        number = _num(value)
        if number is not None:
            return number
    return None


def _range(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _num_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    numbers = [_num(item) for item in value]
    return [float(number) for number in numbers if number is not None]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _read_json_if_present(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    report_path = Path(path)
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}
