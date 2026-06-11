import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.scheduled_routing import make_scheduled_routing_report
from brian_sphere_llm.utils.config import load_config


def test_scheduled_routing_report_checks_schedule_and_logged_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3"
    run_dir.mkdir()
    schedule = [
        {"max_step": 1, "router_probability": 0.1, "lambda_route": 1.0},
        {"max_step": 2, "router_probability": 0.5, "lambda_route": 0.5},
        {"max_step": 3, "router_probability": 1.0, "lambda_route": 0.05},
    ]
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_scheduled_free_routing",
                "loss_weights": {"route": 1.0},
                "routing": {"mode": "scheduled", "schedule": schedule},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {"step": 1, "scheduled_router_probability": 0.1, "scheduled_lambda_route": 1.0},
            {"step": 2, "scheduled_router_probability": 0.5, "scheduled_lambda_route": 0.5},
            {"step": 3, "scheduled_router_probability": 1.0, "scheduled_lambda_route": 0.05},
        ],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [{"step": 3, "scheduled_router_probability": 1.0, "scheduled_lambda_route": 0.05}],
    )

    output = make_scheduled_routing_report(run_dir, output_path=tmp_path / "scheduled.json")

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["checks"]["router_probability_increases"] is True
    assert report["checks"]["lambda_route_decays"] is True
    assert report["checks"]["schedule_values_numeric"] is True
    assert report["checks"]["logged_router_probability_matches_schedule"] is True
    assert report["logged_schedule_values"][-1]["scheduled_router_probability"] == 1.0
    assert report["latest_eval_schedule_values"]["scheduled_lambda_route"] == 0.05


def test_scheduled_routing_report_rejects_boolean_schedule_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3"
    run_dir.mkdir()
    schedule = [
        {"max_step": True, "router_probability": True, "lambda_route": False},
        {"max_step": 2, "router_probability": 1.0, "lambda_route": 0.05},
    ]
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_scheduled_free_routing",
                "loss_weights": {"route": 1.0},
                "routing": {"mode": "scheduled", "schedule": schedule},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [{"step": 1, "scheduled_router_probability": 0.1, "scheduled_lambda_route": 1.0}],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [{"step": 1, "scheduled_router_probability": 0.1, "scheduled_lambda_route": 1.0}],
    )

    output = make_scheduled_routing_report(run_dir, output_path=tmp_path / "scheduled.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["schedule_present"] is True
    assert report["checks"]["schedule_values_numeric"] is False
    assert len(report["schedule"]) == 1


def test_scheduled_routing_report_rejects_boolean_logged_schedule_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3"
    run_dir.mkdir()
    schedule = [
        {"max_step": 1, "router_probability": 0.1, "lambda_route": 1.0},
        {"max_step": 2, "router_probability": 1.0, "lambda_route": 0.05},
    ]
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_scheduled_free_routing",
                "loss_weights": {"route": True},
                "routing": {"mode": "scheduled", "schedule": schedule},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [{"step": 1, "scheduled_router_probability": True, "scheduled_lambda_route": False}],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [{"step": 1, "scheduled_router_probability": True, "scheduled_lambda_route": False}],
    )

    output = make_scheduled_routing_report(run_dir, output_path=tmp_path / "scheduled.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["logged_schedule_step_count"] == 0
    assert report["latest_eval_schedule_values"] is None
    assert report["checks"]["logged_schedule_values_present"] is False


def test_scheduled_routing_eval_config_resolves() -> None:
    cfg = load_config("configs/eval/scheduled_routing.yaml")
    assert cfg["eval_name"] == "scheduled_routing_report"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
