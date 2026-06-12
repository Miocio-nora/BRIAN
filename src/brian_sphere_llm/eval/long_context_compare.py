from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


def make_long_context_comparison_report(
    baseline_report_path: str | Path,
    candidate_report_paths: list[str | Path],
    *,
    output_path: str | Path | None = None,
    min_global_attention_mass: float = 1e-6,
    min_global_read_gate: float = 1e-6,
    quality_tolerance: float = 0.0,
) -> Path:
    baseline = _read_json(Path(baseline_report_path))
    rows = [
        _compare_candidate(
            baseline,
            _read_json(Path(candidate_path)),
            baseline_report_path=Path(baseline_report_path),
            candidate_report_path=Path(candidate_path),
            min_global_attention_mass=min_global_attention_mass,
            min_global_read_gate=min_global_read_gate,
            quality_tolerance=quality_tolerance,
        )
        for candidate_path in candidate_report_paths
    ]
    report = {
        "baseline_report": str(baseline_report_path),
        "candidate_count": len(rows),
        "comparisons": rows,
        "thresholds": {
            "min_global_attention_mass": min_global_attention_mass,
            "min_global_read_gate": min_global_read_gate,
            "quality_tolerance": quality_tolerance,
        },
        "overall_status": _overall_status(rows),
    }
    if output_path is None:
        output_path = Path("reports") / "long_context_compare.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _compare_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    min_global_attention_mass: float,
    min_global_read_gate: float,
    quality_tolerance: float,
) -> dict[str, Any]:
    baseline_overall = baseline.get("overall", {})
    candidate_overall = candidate.get("overall", {})
    baseline_teacher = _num(baseline_overall.get("teacher_forced_token_accuracy"))
    candidate_teacher = _num(candidate_overall.get("teacher_forced_token_accuracy"))
    baseline_exact = _num(baseline_overall.get("exact_match_accuracy"))
    candidate_exact = _num(candidate_overall.get("exact_match_accuracy"))
    global_kv = candidate.get("global_kv", {})
    attention_mass = _num(global_kv.get("global_attention_mass"))
    sink_attention_mass = _num(global_kv.get("global_sink_attention_mass"))
    window_attention_mass = _num(global_kv.get("global_window_attention_mass"))
    read_gate = _num(global_kv.get("global_read_gate_mean"))
    cache_slots = _num(global_kv.get("global_cache_slots_mean"))
    teacher_delta = _delta(candidate_teacher, baseline_teacher)
    exact_delta = _delta(candidate_exact, baseline_exact)
    candidate_memory = candidate.get("memory_budget", {})
    baseline_memory = baseline.get("memory_budget", {})
    baseline_coverage = baseline.get("coverage", {})
    candidate_coverage = candidate.get("coverage", {})
    baseline_global_enabled = _bool_value(baseline_memory.get("global_kv_enabled"))
    candidate_global_enabled = _bool_value(candidate_memory.get("global_kv_enabled"))
    candidate_capacity_ratio = _num(candidate_memory.get("estimated_global_cache_capacity_to_local_context_ratio"))
    baseline_stage4 = baseline.get("stage") == "stage4_output_action"
    candidate_stage5 = candidate.get("stage") == "stage5_global_kv"
    baseline_scheduled = baseline.get("route_mode") == "scheduled"
    candidate_scheduled = candidate.get("route_mode") == "scheduled"
    baseline_local_kv = baseline_global_enabled is False
    candidate_global_kv = candidate_global_enabled is True
    baseline_report_status = baseline.get("overall_status")
    candidate_report_status = candidate.get("overall_status")
    baseline_report_passed = _report_passed(baseline)
    candidate_report_passed = _report_passed(candidate)
    global_active = (
        attention_mass is not None
        and attention_mass >= min_global_attention_mass
        and read_gate is not None
        and read_gate >= min_global_read_gate
        and cache_slots is not None
        and cache_slots > 0.0
    )
    quality_metrics_present = teacher_delta is not None and exact_delta is not None
    quality_not_worse = quality_metrics_present and teacher_delta >= -quality_tolerance and exact_delta >= -quality_tolerance
    memory_budget_present = candidate_global_kv and candidate_capacity_ratio is not None
    global_budget_below_local_context = memory_budget_present and candidate_capacity_ratio < 1.0
    baseline_task_family_coverage = _coverage_passed(baseline_coverage, "task_family_coverage_passed")
    baseline_difficulty_coverage = _coverage_passed(baseline_coverage, "difficulty_coverage_passed")
    candidate_task_family_coverage = _coverage_passed(candidate_coverage, "task_family_coverage_passed")
    candidate_difficulty_coverage = _coverage_passed(candidate_coverage, "difficulty_coverage_passed")
    checks = {
        "baseline_report_passed": baseline_report_passed,
        "candidate_report_passed": candidate_report_passed,
        "baseline_stage4_output_action": baseline_stage4,
        "baseline_scheduled_route_mode": baseline_scheduled,
        "baseline_local_kv": baseline_local_kv,
        "candidate_stage5_global_kv": candidate_stage5,
        "candidate_scheduled_route_mode": candidate_scheduled,
        "candidate_global_kv_enabled": candidate_global_kv,
        "baseline_task_family_coverage": baseline_task_family_coverage,
        "baseline_difficulty_coverage": baseline_difficulty_coverage,
        "candidate_task_family_coverage": candidate_task_family_coverage,
        "candidate_difficulty_coverage": candidate_difficulty_coverage,
        "global_kv_active": global_active,
        "quality_metrics_present": quality_metrics_present,
        "quality_not_worse": quality_not_worse,
        "memory_budget_present": memory_budget_present,
        "global_budget_below_local_context": global_budget_below_local_context,
    }
    return {
        "candidate_report": str(candidate_report_path),
        "baseline_report": str(baseline_report_path),
        "candidate_report_status": candidate_report_status,
        "baseline_report_status": baseline_report_status,
        "candidate_stage": candidate.get("stage"),
        "baseline_stage": baseline.get("stage"),
        "candidate_route_mode": candidate.get("route_mode"),
        "baseline_route_mode": baseline.get("route_mode"),
        "candidate_run_dir": candidate.get("run_dir"),
        "baseline_run_dir": baseline.get("run_dir"),
        "baseline_sample_count": _num(baseline.get("sample_count")),
        "candidate_sample_count": _num(candidate.get("sample_count")),
        "baseline_exact_match_accuracy": baseline_exact,
        "candidate_exact_match_accuracy": candidate_exact,
        "exact_match_delta": exact_delta,
        "baseline_teacher_forced_token_accuracy": baseline_teacher,
        "candidate_teacher_forced_token_accuracy": candidate_teacher,
        "teacher_forced_token_accuracy_delta": teacher_delta,
        "baseline_truncation_rate": _num(baseline_overall.get("truncation_rate")),
        "candidate_truncation_rate": _num(candidate_overall.get("truncation_rate")),
        "global_kv": {
            "global_attention_mass": attention_mass,
            "global_sink_attention_mass": sink_attention_mass,
            "global_window_attention_mass": window_attention_mass,
            "global_read_gate_mean": read_gate,
            "global_cache_slots_mean": cache_slots,
        },
        "memory_budget": {
            "baseline": _memory_summary(baseline_memory),
            "candidate": _memory_summary(candidate_memory),
        },
        "coverage": {
            "baseline": _coverage_summary(baseline_coverage),
            "candidate": _coverage_summary(candidate_coverage),
        },
        "checks": checks,
        "status": _status(checks),
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
    if all(value is True for value in checks.values()):
        return "pass"
    if any(value is True for value in checks.values()):
        return "warn"
    return "fail"


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _memory_summary(memory_budget: Any) -> dict[str, Any]:
    if not isinstance(memory_budget, dict):
        return {}
    keys = [
        "global_kv_enabled",
        "estimated_local_raw_kv_context_bytes_fp16",
        "estimated_global_cache_capacity_bytes_fp16",
        "estimated_global_cache_mean_bytes_fp16",
        "estimated_global_cache_capacity_to_local_context_ratio",
        "estimated_global_cache_mean_to_local_context_ratio",
    ]
    return {key: memory_budget.get(key) for key in keys}


def _coverage_passed(coverage: Any, key: str) -> bool:
    if not isinstance(coverage, dict):
        return False
    return coverage.get(key) is True


def _report_passed(report: Any) -> bool:
    if not isinstance(report, dict) or report.get("overall_status") != "pass":
        return False
    checks = report.get("checks")
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def _coverage_summary(coverage: Any) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    keys = [
        "expected_task_families",
        "observed_task_families",
        "missing_task_families",
        "task_family_coverage_passed",
        "expected_difficulties",
        "observed_difficulties",
        "missing_difficulties",
        "difficulty_coverage_passed",
    ]
    return {key: coverage.get(key) for key in keys}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data
