import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.parallel_passing_report import make_parallel_passing_report
from brian_sphere_llm.utils.config import load_config


def test_parallel_passing_report_passes_bounded_beam_and_cost(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        beam_size=2,
        branch_cost=0.01,
        train_rows=[
            {"parallel_branch_count_mean": 2.0, "parallel_score_margin_mean": 0.2},
            {"parallel_branch_count_mean": 2.0, "parallel_score_margin_mean": 0.1},
        ],
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["checks"]["parallel_passing_enabled"] is True
    assert report["checks"]["parallel_route_selected"] is True
    assert report["checks"]["beam_size_within_limit"] is True
    assert report["checks"]["branch_cost_enabled"] is True
    assert report["checks"]["branch_count_bounded_by_beam"] is True
    assert report["routing"]["parallel_branch_count"]["max"] == 2.0


def test_parallel_passing_report_fails_unbounded_branch_count(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        beam_size=2,
        branch_cost=0.0,
        train_rows=[
            {"parallel_branch_count_mean": 3.0, "parallel_score_margin_mean": 0.2},
        ],
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["branch_cost_enabled"] is False
    assert report["checks"]["branch_count_bounded_by_beam"] is False


def test_parallel_passing_eval_config_resolves() -> None:
    config = load_config("configs/eval/parallel_passing.yaml")
    assert config["eval_name"] == "parallel_passing_report"


def _write_run(
    root: Path,
    *,
    beam_size: int,
    branch_cost: float,
    train_rows: list[dict],
) -> Path:
    run_dir = root / "parallel"
    run_dir.mkdir()
    config = {
        "stage": "stage6_parallel_passing",
        "routing": {"mode": "parallel"},
        "model_config_resolved": {
            "parallel_passing": True,
            "beam_size": beam_size,
            "branch_cost": branch_cost,
            "global_kv": True,
            "global_sink_slots": 1,
            "global_window_slots": 3,
        },
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps({"step": index + 1, **row}) for index, row in enumerate(train_rows)) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"step": len(train_rows), "parallel_branch_count_mean": 2.0, "parallel_score_margin_mean": 0.1})
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(
        json.dumps({"summary": {"parallel_branch_count_mean": 2.0, "parallel_score_margin_mean": 0.1}}),
        encoding="utf-8",
    )
    return run_dir
