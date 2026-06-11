import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.cost_control_report import make_cost_control_report
from brian_sphere_llm.utils.config import load_config


def _write_run(
    root: Path,
    name: str,
    *,
    cost: float,
    active: float,
    steps: float,
    p_output: float | None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump({"stage": "stage4_output_action", "loss_weights": {"cost": cost}}),
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"validation_loss": 10.0 + cost, "perplexity": 123.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train_log.jsonl").write_text(
        json.dumps({"first_exit_step_histogram": {"1": 1, "2": 1}}) + "\n",
        encoding="utf-8",
    )
    summary = {
        "average_route_steps": steps,
        "active_block_evals_per_token": active,
        "route_entropy": 1.0,
    }
    if p_output is not None:
        summary["p_output_mean"] = p_output
    (run_dir / "routing_report.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
    return run_dir


def test_make_cost_control_report_orders_and_scores_sweep(tmp_path: Path) -> None:
    high = _write_run(tmp_path, "high", cost=0.05, active=0.2, steps=1.0, p_output=0.8)
    low = _write_run(tmp_path, "low", cost=0.0, active=0.8, steps=3.0, p_output=0.2)
    mid = _write_run(tmp_path, "mid", cost=0.01, active=0.4, steps=2.0, p_output=0.6)
    output = tmp_path / "cost.json"
    report_path = make_cost_control_report([high, low, mid], output_path=output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [row["cost_weight"] for row in report["runs"]] == [0.0, 0.01, 0.05]
    assert report["analysis"]["status"] == "pass"
    assert report["analysis"]["active_block_evals_range"] == 0.6000000000000001
    assert report["analysis"]["cost_vs_active_block_evals_correlation"] < 0.0
    assert report["analysis"]["checks"]["average_steps_not_increasing_with_cost"] is True
    assert report["analysis"]["average_route_steps_monotonic_nonincreasing"] is True
    assert report["analysis"]["cost_vs_p_output_correlation"] > 0.0


def test_cost_control_report_requires_output_probability_trend(tmp_path: Path) -> None:
    low = _write_run(tmp_path, "low", cost=0.0, active=0.8, steps=3.0, p_output=None)
    high = _write_run(tmp_path, "high", cost=0.05, active=0.2, steps=1.0, p_output=None)
    report_path = make_cost_control_report([high, low], output_path=tmp_path / "cost.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["analysis"]["status"] == "warn"
    assert report["analysis"]["cost_vs_p_output_correlation"] is None
    assert report["analysis"]["checks"]["active_compute_not_increasing_with_cost"] is True
    assert report["analysis"]["checks"]["output_probability_not_decreasing_with_cost"] is False


def test_cost_control_report_rejects_boolean_metrics(tmp_path: Path) -> None:
    low = _write_run(tmp_path, "low", cost=False, active=True, steps=True, p_output=True)
    high = _write_run(tmp_path, "high", cost=True, active=False, steps=False, p_output=False)
    for run in [low, high]:
        (run / "train_log.jsonl").write_text(
            json.dumps({"first_exit_step_histogram": {"1": True, "2": False}}) + "\n",
            encoding="utf-8",
        )
    report_path = make_cost_control_report([high, low], output_path=tmp_path / "cost.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert [row["cost_weight"] for row in report["runs"]] == [None, None]
    assert report["runs"][0]["active_block_evals_per_token"] is None
    assert report["runs"][0]["p_output_mean"] is None
    assert report["runs"][0]["first_exit_step_histogram"] == {}
    assert report["analysis"]["status"] == "fail"
    assert report["analysis"]["checks"]["has_multiple_cost_weights"] is False
    assert report["analysis"]["checks"]["active_compute_range_present"] is False


def test_cost_control_configs_resolve() -> None:
    for path in [
        "configs/train/stage4_cost0_tiny_debug.yaml",
        "configs/train/stage4_cost001_tiny_debug.yaml",
        "configs/train/stage4_cost01_tiny_debug.yaml",
        "configs/train/stage4_cost05_tiny_debug.yaml",
        "configs/train/ablation_c0_cost0.yaml",
        "configs/train/ablation_c1_cost001.yaml",
        "configs/train/ablation_c2_cost01.yaml",
        "configs/train/ablation_c3_cost05.yaml",
        "configs/eval/cost_control_report.yaml",
    ]:
        cfg = load_config(path)
        assert cfg


def test_cost_control_experiment_manifest_resolves() -> None:
    manifest = load_config("configs/experiments/route_core_cost_control.yaml")
    assert manifest["experiment_name"] == "route_core_cost_control"
    assert len(manifest["ablations"]) == 4
