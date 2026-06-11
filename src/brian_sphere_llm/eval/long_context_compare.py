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
    candidate_capacity_ratio = _num(candidate_memory.get("estimated_global_cache_capacity_to_local_context_ratio"))
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
    memory_budget_present = bool(candidate_memory.get("global_kv_enabled")) and candidate_capacity_ratio is not None
    global_budget_below_local_context = memory_budget_present and candidate_capacity_ratio < 1.0
    checks = {
        "global_kv_active": global_active,
        "quality_metrics_present": quality_metrics_present,
        "quality_not_worse": quality_not_worse,
        "memory_budget_present": memory_budget_present,
        "global_budget_below_local_context": global_budget_below_local_context,
    }
    return {
        "candidate_report": str(candidate_report_path),
        "baseline_report": str(baseline_report_path),
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
    if all(checks.values()):
        return "pass"
    if any(checks.values()):
        return "warn"
    return "fail"


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data
