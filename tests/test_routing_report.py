import json
from pathlib import Path

import pytest

from brian_sphere_llm.eval.routing_report import make_routing_report


def test_routing_report_preserves_latest_route_examples_and_trajectories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {
                "step": 1,
                "loss": 3.0,
                "route_entropy": 0.1,
                "active_block_evals_per_token": 0.5,
                "average_route_steps": 2.0,
                "p_output_mean": 0.25,
                "tokens_per_second": 10,
                "train_step_time_seconds": 0.2,
                "train_latency_ms_per_token": 20.0,
                "top1_block_histogram": {"0": 1, "1": 1, "2": 0},
                "exit_step_distribution": [0, 1],
                "first_exit_step_histogram": {"2": 1},
                "route_path_examples": [{"sample_index": 0, "actions": [0, 1]}],
                "position_norm_trajectory": [1.0, 0.75],
                "location_distance_trajectory": [0.5, 0.25],
            },
            {
                "step": 2,
                "loss": 2.0,
                "route_entropy": 0.3,
                "active_block_evals_per_token": 0.25,
                "average_route_steps": 1.0,
                "p_output_mean": 0.75,
                "tokens_per_second": 20,
                "train_step_time_seconds": 0.1,
                "train_latency_ms_per_token": 5.0,
                "top1_block_histogram": {"0": 0, "1": 1, "2": 1},
                "exit_step_distribution": [1, 1],
                "first_exit_step_histogram": {"1": 1},
                "route_path_examples": [{"sample_index": 0, "actions": [2]}],
                "position_norm_trajectory": [0.5],
                "location_distance_trajectory": [0.1],
            },
        ],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [
            {
                "step": 2,
                "validation_loss": 2.0,
                "perplexity": 7.4,
                "inference_time_seconds": 0.2,
                "inference_tokens_per_second": 10.0,
                "inference_latency_ms_per_token": 100.0,
            }
        ],
    )

    output = make_routing_report(run_dir)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["summary"]["route_entropy"] == 0.2
    assert report["latest_block_histogram"] == {"0": 0, "1": 1, "2": 1}
    assert report["latest_exit_step_distribution"] == [1, 1]
    assert report["latest_first_exit_step_histogram"] == {"1": 1}
    assert report["latest_route_path_examples"] == [{"sample_index": 0, "actions": [2]}]
    assert report["latest_position_norm_trajectory"] == [0.5]
    assert report["latest_location_distance_trajectory"] == [0.1]
    assert report["latest_eval"]["validation_loss"] == 2.0
    assert report["cost_quality_curve"]["train_points"] == [
        {
            "step": 1.0,
            "train_loss": 3.0,
            "active_block_evals_per_token": 0.5,
            "average_route_steps": 2.0,
            "p_output_mean": 0.25,
            "tokens_per_second": 10.0,
            "train_step_time_seconds": 0.2,
            "train_latency_ms_per_token": 20.0,
        },
        {
            "step": 2.0,
            "train_loss": 2.0,
            "active_block_evals_per_token": 0.25,
            "average_route_steps": 1.0,
            "p_output_mean": 0.75,
            "tokens_per_second": 20.0,
            "train_step_time_seconds": 0.1,
            "train_latency_ms_per_token": 5.0,
        },
    ]
    assert report["cost_quality_curve"]["eval_points"] == [
        {
            "step": 2.0,
            "validation_loss": 2.0,
            "perplexity": 7.4,
            "inference_time_seconds": 0.2,
            "inference_tokens_per_second": 10.0,
            "inference_latency_ms_per_token": 100.0,
            "active_block_evals_per_token": 0.25,
            "average_route_steps": 1.0,
            "p_output_mean": 0.75,
            "tokens_per_second": 20.0,
            "train_step_time_seconds": 0.1,
            "train_latency_ms_per_token": 5.0,
        }
    ]
    assert report["cost_quality_curve"]["summary"]["active_compute_range"] == 0.25
    assert report["cost_quality_curve"]["summary"]["train_loss_vs_active_compute_correlation"] == pytest.approx(1.0)


def test_routing_report_matches_eval_curve_to_previous_train_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {"step": 1, "loss": 3.0, "active_block_evals_per_token": 0.1, "average_route_steps": 1.0},
            {"step": 3, "loss": 2.5, "active_block_evals_per_token": 0.3, "average_route_steps": 2.0},
            {"step": 5, "loss": 2.0, "active_block_evals_per_token": 0.5, "average_route_steps": 3.0},
        ],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [
            {"step": 4, "validation_loss": 2.4, "perplexity": 11.0},
            {"step": 5, "validation_loss": 2.0, "perplexity": 7.4},
        ],
    )

    report = json.loads(make_routing_report(run_dir).read_text(encoding="utf-8"))
    eval_points = report["cost_quality_curve"]["eval_points"]

    assert eval_points[0]["active_block_evals_per_token"] == 0.3
    assert eval_points[0]["average_route_steps"] == 2.0
    assert eval_points[1]["active_block_evals_per_token"] == 0.5
    assert report["cost_quality_curve"]["summary"]["eval_point_count"] == 2
    assert report["cost_quality_curve"]["summary"]["validation_loss_vs_active_compute_correlation"] == pytest.approx(-1.0)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
