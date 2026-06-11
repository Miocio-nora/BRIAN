import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.position_ablation import make_position_ablation_report
from brian_sphere_llm.utils.config import load_config


def test_position_ablation_report_passes_on_metric_delta(tmp_path: Path) -> None:
    reference = _write_run(
        tmp_path,
        "main_position",
        validation_loss=10.0,
        routing={"average_route_steps": 2.0, "route_entropy": 0.5, "position_norm_mean": 1.0},
    )
    candidate = _write_run(
        tmp_path,
        "no_position",
        validation_loss=10.0,
        routing={"average_route_steps": 1.5, "route_entropy": 0.5, "position_norm_mean": 0.0},
    )

    output = make_position_ablation_report(
        reference,
        [candidate],
        output_path=tmp_path / "position.json",
        min_routing_metric_delta=0.1,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["checks"]["any_measurable_difference"] is True
    comparison = report["comparisons"][0]
    assert comparison["checks"]["routing_metric_delta_measurable"] is True
    assert comparison["routing_metric_deltas"]["average_route_steps"] == -0.5
    assert comparison["measurable_routing_metric_deltas"]["position_norm_mean"] == -1.0


def test_position_ablation_report_fails_without_difference(tmp_path: Path) -> None:
    reference = _write_run(
        tmp_path,
        "main_position",
        validation_loss=10.0,
        routing={"average_route_steps": 2.0, "route_entropy": 0.5, "position_norm_mean": 1.0},
    )
    candidate = _write_run(
        tmp_path,
        "same_position",
        validation_loss=10.0,
        routing={"average_route_steps": 2.0, "route_entropy": 0.5, "position_norm_mean": 1.0},
    )

    output = make_position_ablation_report(reference, [candidate], output_path=tmp_path / "position.json")

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "fail"
    assert report["checks"]["any_measurable_difference"] is False
    assert report["comparisons"][0]["status"] == "fail"


def test_position_ablation_report_rejects_boolean_metrics(tmp_path: Path) -> None:
    reference = _write_run(
        tmp_path,
        "main_position",
        validation_loss=False,
        routing={"average_route_steps": False, "route_entropy": False, "position_norm_mean": False},
    )
    candidate = _write_run(
        tmp_path,
        "no_position",
        validation_loss=True,
        routing={"average_route_steps": True, "route_entropy": True, "position_norm_mean": True},
    )

    output = make_position_ablation_report(reference, [candidate], output_path=tmp_path / "position.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    comparison = report["comparisons"][0]

    assert report["overall_status"] == "fail"
    assert comparison["validation_loss_delta"] is None
    assert comparison["routing_metric_deltas"]["average_route_steps"] is None
    assert comparison["checks"]["measurable_difference"] is False


def test_position_ablation_eval_config_resolves() -> None:
    cfg = load_config("configs/eval/position_ablation.yaml")
    assert cfg["eval_name"] == "position_ablation_report"


def _write_run(root: Path, name: str, *, validation_loss: float, routing: dict[str, float]) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump({"stage": "stage3_scheduled_free_routing"}),
        encoding="utf-8",
    )
    (run_dir / "model_stats.json").write_text(json.dumps({"model_name": name}), encoding="utf-8")
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"validation_loss": validation_loss, "perplexity": 100.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(json.dumps({"summary": routing}), encoding="utf-8")
    return run_dir
