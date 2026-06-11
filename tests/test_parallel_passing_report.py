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
        window_slots=3,
        train_rows=[
            {
                "parallel_branch_count_mean": 2.0,
                "parallel_score_margin_mean": 0.2,
                "parallel_delta_cache_slots_max": 1.0,
            },
            {
                "parallel_branch_count_mean": 2.0,
                "parallel_score_margin_mean": 0.1,
                "parallel_delta_cache_slots_max": 2.0,
            },
        ],
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["checks"]["parallel_passing_enabled"] is True
    assert report["checks"]["parallel_route_selected"] is True
    assert report["checks"]["shared_base_global_memory_enabled"] is True
    assert report["checks"]["beam_size_within_limit"] is True
    assert report["checks"]["branch_cost_enabled"] is True
    assert report["checks"]["branch_count_bounded_by_beam"] is True
    assert report["checks"]["score_margin_nonnegative"] is True
    assert report["checks"]["branch_delta_memory_measured"] is True
    assert report["checks"]["delta_cache_nonnegative"] is True
    assert report["checks"]["delta_memory_policy_present"] is True
    assert report["checks"]["delta_cache_bounded_by_window"] is True
    assert report["model"]["memory_policy"] == "shared_base_global_kv_with_branch_delta"
    assert report["model"]["parallel_exit_policy"] == "branch"
    assert report["routing"]["parallel_branch_count"]["max"] == 2.0
    assert report["routing"]["parallel_delta_cache_slots"]["max"] == 2.0


def test_parallel_passing_report_fails_unbounded_branch_and_delta_cache(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        beam_size=2,
        branch_cost=0.0,
        window_slots=2,
        train_rows=[
            {
                "parallel_branch_count_mean": 3.0,
                "parallel_score_margin_mean": 0.2,
                "parallel_delta_cache_slots_max": 4.0,
            },
        ],
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["branch_cost_enabled"] is False
    assert report["checks"]["branch_count_bounded_by_beam"] is False
    assert report["checks"]["delta_cache_bounded_by_window"] is False


def test_parallel_passing_report_fails_negative_margin_or_delta_cache(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        beam_size=2,
        branch_cost=0.01,
        window_slots=3,
        train_rows=[
            {
                "parallel_branch_count_mean": 2.0,
                "parallel_score_margin_mean": -0.1,
                "parallel_delta_cache_slots_max": -1.0,
            },
        ],
        include_eval_delta=False,
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["score_margin_measured"] is True
    assert report["checks"]["score_margin_nonnegative"] is False
    assert report["checks"]["branch_delta_memory_measured"] is True
    assert report["checks"]["delta_cache_nonnegative"] is False


def test_parallel_passing_report_fails_missing_branch_delta_memory(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        beam_size=2,
        branch_cost=0.01,
        window_slots=3,
        train_rows=[
            {
                "parallel_branch_count_mean": 2.0,
                "parallel_score_margin_mean": 0.2,
            },
        ],
        include_eval_delta=False,
    )

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["shared_base_global_memory_enabled"] is True
    assert report["checks"]["branch_delta_memory_measured"] is False


def test_parallel_passing_report_rejects_boolean_numeric_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "parallel_bool"
    run_dir.mkdir()
    config = {
        "stage": "stage6_parallel_passing",
        "routing": {"mode": "parallel"},
        "model_config_resolved": {
            "parallel_passing": True,
            "beam_size": True,
            "branch_cost": True,
            "global_kv": True,
            "global_sink_slots": True,
            "global_window_slots": True,
        },
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    train_row = {
        "parallel_branch_count_mean": True,
        "parallel_score_margin_mean": True,
        "parallel_delta_cache_slots_max": True,
    }
    (run_dir / "train_log.jsonl").write_text(json.dumps({"step": 1, **train_row}) + "\n", encoding="utf-8")
    (run_dir / "eval_log.jsonl").write_text(json.dumps({"step": 1, **train_row}) + "\n", encoding="utf-8")
    (run_dir / "routing_report.json").write_text(json.dumps({"summary": train_row}), encoding="utf-8")

    report_path = make_parallel_passing_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["model"]["beam_size"] == 0
    assert report["model"]["branch_cost"] is None
    assert report["routing"]["parallel_branch_count"]["count"] == 0
    assert report["checks"]["beam_size_present"] is False
    assert report["checks"]["branch_cost_enabled"] is False
    assert report["checks"]["branch_metrics_present"] is False
    assert report["checks"]["score_margin_measured"] is False
    assert report["checks"]["branch_delta_memory_measured"] is False


def test_parallel_passing_eval_config_resolves() -> None:
    config = load_config("configs/eval/parallel_passing.yaml")
    assert config["eval_name"] == "parallel_passing_report"


def _write_run(
    root: Path,
    *,
    beam_size: int,
    branch_cost: float,
    window_slots: int,
    train_rows: list[dict],
    include_eval_delta: bool = True,
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
            "global_window_slots": window_slots,
        },
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps({"step": index + 1, **row}) for index, row in enumerate(train_rows)) + "\n",
        encoding="utf-8",
    )
    eval_row = {
        "step": len(train_rows),
        "parallel_branch_count_mean": 2.0,
        "parallel_score_margin_mean": 0.1,
    }
    routing_summary = {
        "parallel_branch_count_mean": 2.0,
        "parallel_score_margin_mean": 0.1,
    }
    if include_eval_delta:
        eval_row["parallel_delta_cache_slots_max"] = min(float(window_slots), 2.0)
        routing_summary["parallel_delta_cache_slots_max"] = min(float(window_slots), 2.0)
    (run_dir / "eval_log.jsonl").write_text(json.dumps(eval_row) + "\n", encoding="utf-8")
    (run_dir / "routing_report.json").write_text(json.dumps({"summary": routing_summary}), encoding="utf-8")
    return run_dir
