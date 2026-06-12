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
                "block_load_entropy": 0.2,
                "route_path_diversity": 0.4,
                "active_block_evals_per_token": 0.5,
                "average_route_steps": 2.0,
                "advance_ratio": 0.5,
                "skip_ratio": 0.25,
                "recur_ratio": 0.25,
                "position_norm_mean": 1.0,
                "location_distance_mean": 0.5,
                "p_output_mean": 0.25,
                "cost_loss": 0.02,
                "balance_loss": 0.03,
                "location_loss": 0.04,
                "global_read_gate_mean": 0.25,
                "local_read_fraction_mean": 0.75,
                "global_to_local_read_ratio": 1 / 3,
                "local_to_global_read_ratio": 3.0,
                "max_route_steps": 2,
                "forced_max_step_exit_count": 1,
                "forced_max_step_exit_fraction": 0.5,
                "tokens_per_second": 10,
                "train_step_time_seconds": 0.2,
                "train_latency_ms_per_token": 20.0,
                "top1_block_histogram": {"0": 1, "1": 1, "2": 0},
                "topk_block_histogram": {"0": 2, "1": 1, "2": 1},
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
                "block_load_entropy": 0.4,
                "route_path_diversity": 0.6,
                "active_block_evals_per_token": 0.25,
                "average_route_steps": 1.0,
                "advance_ratio": 0.25,
                "skip_ratio": 0.5,
                "recur_ratio": 0.25,
                "position_norm_mean": 0.5,
                "location_distance_mean": 0.1,
                "p_output_mean": 0.75,
                "cost_loss": 0.01,
                "balance_loss": 0.02,
                "location_loss": 0.03,
                "global_read_gate_mean": 0.75,
                "local_read_fraction_mean": 0.25,
                "global_to_local_read_ratio": 3.0,
                "local_to_global_read_ratio": 1 / 3,
                "max_route_steps": 2,
                "forced_max_step_exit_count": 0,
                "forced_max_step_exit_fraction": 0.0,
                "tokens_per_second": 20,
                "train_step_time_seconds": 0.1,
                "train_latency_ms_per_token": 5.0,
                "top1_block_histogram": {"0": 0, "1": 1, "2": 1},
                "topk_block_histogram": {"0": 1, "1": 1, "2": 2},
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
    assert report["overall_status"] == "pass"
    assert report["checks"]["core_route_metrics_present"] is True
    assert report["checks"]["route_transition_ratios_present"] is True
    assert report["checks"]["position_location_metrics_present"] is True
    assert report["checks"]["exit_distribution_present"] is True
    assert report["checks"]["topk_block_histogram_present"] is True
    assert report["checks"]["output_probability_present"] is True
    assert report["checks"]["route_path_examples_present"] is True
    assert report["checks"]["position_trajectory_present"] is True
    assert report["checks"]["location_trajectory_present"] is True
    assert report["checks"]["cost_quality_train_points_present"] is True
    assert report["checks"]["cost_quality_eval_points_present"] is True
    assert report["checks"]["training_timing_metrics_present"] is True
    assert report["checks"]["inference_timing_metrics_present"] is True
    assert report["checks"]["route_loss_terms_present"] is True
    assert report["summary"]["global_read_gate_mean"] == 0.5
    assert report["summary"]["local_read_fraction_mean"] == 0.5
    assert report["summary"]["global_to_local_read_ratio"] == pytest.approx((1 / 3 + 3.0) / 2)
    assert report["summary"]["local_to_global_read_ratio"] == pytest.approx((3.0 + 1 / 3) / 2)
    assert report["summary"]["max_route_steps"] == 2.0
    assert report["summary"]["forced_max_step_exit_count"] == 0.5
    assert report["summary"]["forced_max_step_exit_fraction"] == 0.25
    assert report["latest_block_histogram"] == {"0": 0, "1": 1, "2": 1}
    assert report["latest_topk_block_histogram"] == {"0": 1, "1": 1, "2": 2}
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


def test_routing_report_warns_when_route_behavior_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "train_log.jsonl", [{"step": 1, "loss": 3.0}])
    _write_jsonl(run_dir / "eval_log.jsonl", [{"step": 1, "validation_loss": 2.0, "perplexity": 7.4}])

    report = json.loads(make_routing_report(run_dir).read_text(encoding="utf-8"))

    assert report["overall_status"] == "warn"
    assert report["checks"]["train_log_present"] is True
    assert report["checks"]["eval_log_present"] is True
    assert report["checks"]["core_route_metrics_present"] is False
    assert report["checks"]["route_transition_ratios_present"] is False
    assert report["checks"]["position_location_metrics_present"] is False
    assert report["checks"]["block_histogram_present"] is False
    assert report["checks"]["topk_block_histogram_present"] is False
    assert report["checks"]["output_probability_present"] is False
    assert report["checks"]["exit_distribution_present"] is False
    assert report["checks"]["route_path_examples_present"] is False
    assert report["checks"]["position_trajectory_present"] is False
    assert report["checks"]["location_trajectory_present"] is False
    assert report["checks"]["cost_quality_train_points_present"] is False
    assert report["checks"]["cost_quality_eval_points_present"] is False
    assert report["checks"]["training_timing_metrics_present"] is False
    assert report["checks"]["inference_timing_metrics_present"] is False
    assert report["checks"]["route_loss_terms_present"] is False


def test_routing_report_rejects_boolean_route_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {
                "step": 1,
                "loss": 3.0,
                "route_entropy": True,
                "block_load_entropy": True,
                "route_path_diversity": True,
                "active_block_evals_per_token": True,
                "average_route_steps": True,
                "advance_ratio": True,
                "skip_ratio": True,
                "recur_ratio": True,
                "position_norm_mean": True,
                "location_distance_mean": True,
                "p_output_mean": True,
                "cost_loss": True,
                "balance_loss": True,
                "location_loss": True,
                "tokens_per_second": 10.0,
                "train_step_time_seconds": 0.2,
                "train_latency_ms_per_token": 20.0,
                "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
                "topk_block_histogram": {"0": 1, "1": 1, "2": 1},
                "exit_step_distribution": [1, 1],
                "route_path_examples": [{"sample_index": 0, "actions": [0, 1]}],
                "position_norm_trajectory": [1.0],
                "location_distance_trajectory": [0.5],
            }
        ],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [
            {
                "step": 1,
                "validation_loss": 2.0,
                "perplexity": 7.4,
                "inference_time_seconds": 0.2,
                "inference_tokens_per_second": 10.0,
                "inference_latency_ms_per_token": 100.0,
            }
        ],
    )

    report = json.loads(make_routing_report(run_dir).read_text(encoding="utf-8"))

    assert "route_entropy" not in report["summary"]
    assert report["overall_status"] == "warn"
    assert report["checks"]["core_route_metrics_present"] is False
    assert report["checks"]["route_transition_ratios_present"] is False
    assert report["checks"]["position_location_metrics_present"] is False
    assert report["checks"]["output_probability_present"] is False
    assert report["checks"]["route_loss_terms_present"] is False


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
