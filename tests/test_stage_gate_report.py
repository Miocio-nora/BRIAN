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
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
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
    output = tmp_path / "gate.json"
    report_path = make_stage_gate_report([baseline, fixed, stage5], output_path=output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_count"] == 3
    assert report["gates"]["stage0_to_1"]["status"] == "pass"
    assert report["gates"]["stage1_to_2"]["status"] == "pass"
    assert report["gates"]["stage5_to_6"]["status"] == "pass"
