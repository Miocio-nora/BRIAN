import json
from pathlib import Path

import pytest

from brian_sphere_llm.eval.long_context_compare import make_long_context_comparison_report


def _write_long_context_report(
    path: Path,
    *,
    run_dir: str,
    stage: str,
    exact_match: float,
    teacher_accuracy: float,
    route_mode: str = "scheduled",
    attention_mass: float | None = None,
    read_gate: float | None = None,
    cache_slots: float | None = None,
    sink_attention_mass: float | None = None,
    window_attention_mass: float | None = None,
    global_kv_enabled: bool | None = None,
    coverage: dict | None = None,
) -> Path:
    if global_kv_enabled is None:
        global_kv_enabled = attention_mass is not None
    if coverage is None:
        coverage = _passing_coverage()
    path.write_text(
        json.dumps(
            {
                "run_dir": run_dir,
                "stage": stage,
                "route_mode": route_mode,
                "sample_count": 4,
                "overall": {
                    "exact_match_accuracy": exact_match,
                    "teacher_forced_token_accuracy": teacher_accuracy,
                    "truncation_rate": 0.5,
                },
                "coverage": coverage,
                "global_kv": {
                    "global_attention_mass": attention_mass,
                    "global_sink_attention_mass": sink_attention_mass,
                    "global_window_attention_mass": window_attention_mass,
                    "global_read_gate_mean": read_gate,
                    "global_cache_slots_mean": cache_slots,
                },
                "memory_budget": {
                    "global_kv_enabled": global_kv_enabled,
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


def _passing_coverage() -> dict:
    return {
        "expected_task_families": ["needle_retrieval", "two_hop_tracing"],
        "observed_task_families": ["needle_retrieval", "two_hop_tracing"],
        "missing_task_families": [],
        "task_family_coverage_passed": True,
        "expected_difficulties": ["near", "middle", "far"],
        "observed_difficulties": ["near", "middle", "far"],
        "missing_difficulties": [],
        "difficulty_coverage_passed": True,
    }


def test_long_context_compare_passes_active_global_kv_not_worse(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        stage="stage4_output_action",
        exact_match=0.25,
        teacher_accuracy=0.50,
    )
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        stage="stage5_global_kv",
        exact_match=0.25,
        teacher_accuracy=0.55,
        attention_mass=0.1,
        sink_attention_mass=0.03,
        window_attention_mass=0.07,
        read_gate=0.2,
        cache_slots=3.0,
    )
    output = make_long_context_comparison_report(baseline, [candidate], output_path=tmp_path / "compare.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]
    assert report["overall_status"] == "pass"
    assert row["candidate_report"] == str(candidate)
    assert row["baseline_report"] == str(baseline)
    assert row["baseline_stage"] == "stage4_output_action"
    assert row["candidate_stage"] == "stage5_global_kv"
    assert row["checks"]["baseline_stage4_output_action"] is True
    assert row["checks"]["baseline_scheduled_route_mode"] is True
    assert row["checks"]["baseline_local_kv"] is True
    assert row["checks"]["candidate_stage5_global_kv"] is True
    assert row["checks"]["candidate_scheduled_route_mode"] is True
    assert row["checks"]["candidate_global_kv_enabled"] is True
    assert row["checks"]["baseline_task_family_coverage"] is True
    assert row["checks"]["baseline_difficulty_coverage"] is True
    assert row["checks"]["candidate_task_family_coverage"] is True
    assert row["checks"]["candidate_difficulty_coverage"] is True
    assert row["checks"]["global_kv_active"] is True
    assert row["checks"]["quality_metrics_present"] is True
    assert row["checks"]["quality_not_worse"] is True
    assert row["checks"]["memory_budget_present"] is True
    assert row["checks"]["global_budget_below_local_context"] is True
    assert row["global_kv"]["global_sink_attention_mass"] == 0.03
    assert row["global_kv"]["global_window_attention_mass"] == 0.07
    assert row["memory_budget"]["candidate"]["estimated_global_cache_capacity_bytes_fp16"] == 128.0
    assert row["coverage"]["candidate"]["task_family_coverage_passed"] is True
    assert row["coverage"]["candidate"]["difficulty_coverage_passed"] is True
    assert row["teacher_forced_token_accuracy_delta"] == pytest.approx(0.05)


def test_long_context_compare_warns_for_inactive_and_worse_candidate(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        stage="stage4_output_action",
        exact_match=0.5,
        teacher_accuracy=0.8,
    )
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        stage="stage5_global_kv",
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


def test_long_context_compare_requires_stage4_local_to_stage5_global_roles(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "wrong_baseline.json",
        run_dir="runs/wrong-baseline",
        stage="stage5_global_kv",
        route_mode="parallel",
        exact_match=0.25,
        teacher_accuracy=0.50,
        global_kv_enabled=True,
    )
    candidate = _write_long_context_report(
        tmp_path / "wrong_candidate.json",
        run_dir="runs/wrong-candidate",
        stage="stage4_output_action",
        route_mode="fixed",
        exact_match=0.25,
        teacher_accuracy=0.55,
        attention_mass=0.1,
        sink_attention_mass=0.03,
        window_attention_mass=0.07,
        read_gate=0.2,
        cache_slots=3.0,
        global_kv_enabled=False,
    )

    output = make_long_context_comparison_report(baseline, [candidate], output_path=tmp_path / "compare.json")
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]

    assert row["status"] == "warn"
    assert row["checks"]["baseline_stage4_output_action"] is False
    assert row["checks"]["baseline_scheduled_route_mode"] is False
    assert row["checks"]["baseline_local_kv"] is False
    assert row["checks"]["candidate_stage5_global_kv"] is False
    assert row["checks"]["candidate_scheduled_route_mode"] is False
    assert row["checks"]["candidate_global_kv_enabled"] is False
    assert row["checks"]["global_kv_active"] is True
    assert row["checks"]["memory_budget_present"] is False


def test_long_context_compare_requires_task_family_and_difficulty_coverage(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        stage="stage4_output_action",
        exact_match=0.25,
        teacher_accuracy=0.50,
    )
    incomplete_coverage = _passing_coverage() | {
        "observed_task_families": ["needle_retrieval"],
        "missing_task_families": ["two_hop_tracing"],
        "task_family_coverage_passed": False,
        "observed_difficulties": ["near", "middle"],
        "missing_difficulties": ["far"],
        "difficulty_coverage_passed": False,
    }
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        stage="stage5_global_kv",
        exact_match=0.25,
        teacher_accuracy=0.55,
        attention_mass=0.1,
        sink_attention_mass=0.03,
        window_attention_mass=0.07,
        read_gate=0.2,
        cache_slots=3.0,
        coverage=incomplete_coverage,
    )

    output = make_long_context_comparison_report(baseline, [candidate], output_path=tmp_path / "compare.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]

    assert report["overall_status"] == "warn"
    assert row["status"] == "warn"
    assert row["checks"]["global_kv_active"] is True
    assert row["checks"]["quality_not_worse"] is True
    assert row["checks"]["global_budget_below_local_context"] is True
    assert row["checks"]["baseline_task_family_coverage"] is True
    assert row["checks"]["baseline_difficulty_coverage"] is True
    assert row["checks"]["candidate_task_family_coverage"] is False
    assert row["checks"]["candidate_difficulty_coverage"] is False
    assert row["coverage"]["candidate"]["missing_task_families"] == ["two_hop_tracing"]
    assert row["coverage"]["candidate"]["missing_difficulties"] == ["far"]


def test_long_context_compare_rejects_boolean_metrics(tmp_path: Path) -> None:
    baseline = _write_long_context_report(
        tmp_path / "local.json",
        run_dir="runs/local",
        stage="stage4_output_action",
        exact_match=False,
        teacher_accuracy=False,
    )
    candidate = _write_long_context_report(
        tmp_path / "global.json",
        run_dir="runs/global",
        stage="stage5_global_kv",
        exact_match=True,
        teacher_accuracy=True,
        attention_mass=True,
        sink_attention_mass=True,
        window_attention_mass=True,
        read_gate=True,
        cache_slots=True,
    )

    output = make_long_context_comparison_report(baseline, [candidate], output_path=tmp_path / "compare.json")
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]

    assert row["status"] == "warn"
    assert row["baseline_exact_match_accuracy"] is None
    assert row["candidate_exact_match_accuracy"] is None
    assert row["global_kv"]["global_attention_mass"] is None
    assert row["checks"]["global_kv_active"] is False
    assert row["checks"]["quality_metrics_present"] is False
