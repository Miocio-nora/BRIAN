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
