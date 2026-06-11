import json
from pathlib import Path

import yaml
import pytest

from brian_sphere_llm.eval.difficulty import difficulty_step_correlation
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report, pearson_correlation


def _write_run(
    root: Path,
    name: str,
    *,
    stage: str,
    val_loss: float,
    train_row: dict,
    determinism_status: str | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_latest").mkdir()
    (run_dir / "checkpoint_latest" / "state.pt").write_bytes(b"stub")
    (run_dir / "checkpoint_best").mkdir()
    (run_dir / "checkpoint_best" / "state.pt").write_bytes(b"stub")
    (run_dir / "model_stats.json").write_text(json.dumps({"model_name": name}), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump({"stage": stage}), encoding="utf-8")
    (run_dir / "eval_log.jsonl").write_text(json.dumps({"validation_loss": val_loss, "perplexity": 1.0}) + "\n", encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(json.dumps(train_row | {"loss": val_loss}) + "\n", encoding="utf-8")
    if determinism_status is not None:
        (run_dir / "eval_determinism_report.json").write_text(
            json.dumps({"overall_status": determinism_status, "checks": {"numeric_metrics_within_tolerance": True}}),
            encoding="utf-8",
        )
    return run_dir


def test_pearson_and_difficulty_step_correlation() -> None:
    assert pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert difficulty_step_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert difficulty_step_correlation([1.0], [1.0]) is None


def test_stage_gate_report_writes_json(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
    )
    stage5 = _write_run(
        tmp_path,
        "global",
        stage="stage5_global_kv",
        val_loss=11.0,
        train_row={
            "global_attention_mass": 1.0,
            "global_read_gate_mean": 0.01,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    long_context_compare = tmp_path / "long_context_compare.json"
    long_context_compare.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [{"status": "pass", "checks": {"global_kv_active": True, "quality_not_worse": True}}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gate.json"
    report_path = make_stage_gate_report(
        [baseline, fixed, stage5],
        output_path=output,
        long_context_compare_report_path=long_context_compare,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_count"] == 3
    assert report["gates"]["stage0_to_1"]["status"] == "pass"
    assert report["gates"]["stage1_to_2"]["status"] == "pass"
    assert report["gates"]["stage5_to_6"]["status"] == "pass"
    assert report["supplemental_reports"]["long_context_compare_report"] == str(long_context_compare)


def test_stage_gate_report_uses_cost_control_report(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 2.0,
            "first_exit_step_histogram": {"1": 1, "2": 2},
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    cost_report = tmp_path / "cost.json"
    cost_report.write_text(
        json.dumps(
            {
                "run_count": 4,
                "analysis": {
                    "status": "pass",
                    "active_block_evals_range": 0.4,
                    "checks": {
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "output_probability_not_decreasing_with_cost": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = make_stage_gate_report([stage4], output_path=tmp_path / "gate.json", cost_control_report_path=cost_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["gates"]["stage4_to_5"]
    assert gate["checks"]["cost_control_report_present"] is True
    assert gate["checks"]["cost_control_active_range_present"] is True
    assert gate["cost_control_status"] == "pass"
    assert report["supplemental_reports"]["cost_control_report"] == str(cost_report)


def test_stage5_gate_warns_without_long_context_compare_report(tmp_path: Path) -> None:
    stage5 = _write_run(
        tmp_path,
        "global",
        stage="stage5_global_kv",
        val_loss=10.0,
        train_row={
            "global_attention_mass": 1.0,
            "global_read_gate_mean": 0.01,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    report_path = make_stage_gate_report([stage5], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]
    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_report_present"] is False
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage6_gate_uses_parallel_compare_report(tmp_path: Path) -> None:
    stage6 = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [{"status": "pass", "checks": {"parallel_branch_benefit_proxy": True}}],
            }
        ),
        encoding="utf-8",
    )
    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["gates"]["stage6_to_scale"]
    assert gate["status"] == "pass"
    assert gate["checks"]["parallel_compare_report_present"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is True
    assert report["supplemental_reports"]["parallel_compare_report"] == str(compare_report)


def test_stage6_gate_warns_without_parallel_compare_report(tmp_path: Path) -> None:
    stage6 = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    report_path = make_stage_gate_report([stage6], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]
    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_compare_report_present"] is False
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False
