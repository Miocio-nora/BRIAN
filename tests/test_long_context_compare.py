import json
from pathlib import Path

import pytest

from brian_sphere_llm.eval.long_context_compare import make_long_context_comparison_report


def _write_long_context_report(
    path: Path,
    *,
    run_dir: str,
    exact_match: float,
    teacher_accuracy: float,
    attention_mass: float | None = None,
    read_gate: float | None = None,
    cache_slots: float | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "run_dir": run_dir,
                "sample_count": 4,
                "overall": {
                    "exact_match_accuracy": exact_match,
                    "teacher_forced_token_accuracy": teacher_accuracy,
                    "truncation_rate": 0.5,
                },
                "global_kv": {
                    "global_attention_mass": attention_mass,
                    "global_read_gate_mean": read_gate,
                    "global_cache_slots_mean": cache_slots,
                },
                "memory_budget": {
                    "global_kv_enabled": attention_mass is not None,
                    "estimated_local_raw_kv_context_bytes_fp16": 8192.0,
                    "estimated_global_cache_capacity_bytes_fp16": 128.0 if attention_mass is not None else 0.0,
                    "estimated_global_cache_mean_bytes_fp16": 64.0 if attention_mass is not None else 0.0,
                    "estimated_global_cache_capacity_to_local_context_ratio": 128.0 / 8192.0
                    if attention_mass is not None
                    else 0.0,
                    "estimated_global_cache_mean_to_local_context_ratio": 64.0 / 8192.0
                    if attention_mass is not None
                    else 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_long_context_compare_passes_active_global_kv_not_worse(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        exact_match=0.25,
        teacher_accuracy=0.50,
    )
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        exact_match=0.25,
        teacher_accuracy=0.55,
        attention_mass=0.1,
        read_gate=0.2,
        cache_slots=3.0,
    )
    output = make_long_context_comparison_report(baseline, [candidate], output_path=tmp_path / "compare.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]
    assert report["overall_status"] == "pass"
    assert row["candidate_report"] == str(candidate)
    assert row["baseline_report"] == str(baseline)
    assert row["checks"]["global_kv_active"] is True
    assert row["checks"]["quality_metrics_present"] is True
    assert row["checks"]["quality_not_worse"] is True
    assert row["checks"]["memory_budget_present"] is True
    assert row["checks"]["global_budget_below_local_context"] is True
    assert row["memory_budget"]["candidate"]["estimated_global_cache_capacity_bytes_fp16"] == 128.0
    assert row["teacher_forced_token_accuracy_delta"] == pytest.approx(0.05)


def test_long_context_compare_warns_for_inactive_and_worse_candidate(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        exact_match=0.5,
        teacher_accuracy=0.8,
    )
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        exact_match=0.25,
        teacher_accuracy=0.6,
        attention_mass=0.0,
        read_gate=0.0,
        cache_slots=0.0,
    )
    output = make_long_context_comparison_report(
        baseline,
        [candidate],
        output_path=tmp_path / "compare.json",
        quality_tolerance=0.05,
    )
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]
    assert row["status"] == "warn"
    assert row["checks"]["global_kv_active"] is False
    assert row["checks"]["quality_metrics_present"] is True
    assert row["checks"]["quality_not_worse"] is False
