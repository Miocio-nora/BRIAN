import json
from pathlib import Path

import pytest
import yaml

from brian_sphere_llm.eval.compute_report import estimate_gpu_hours, make_compute_report, summarize_run


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_run(
    root: Path,
    name: str,
    *,
    config: dict,
    model_stats: dict,
    routing_summary: dict | None = None,
    validation_loss: float = 10.0,
    tokens_per_second: int = 100,
    train_latency_ms_per_token: float = 0.5,
    inference_latency_ms_per_token: float = 0.25,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_json(run_dir / "model_stats.json", model_stats)
    (run_dir / "train_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "loss": validation_loss + 1.0,
                "tokens_per_second": tokens_per_second,
                "train_step_time_seconds": 0.1,
                "train_latency_ms_per_token": train_latency_ms_per_token,
                "cuda_memory_allocated_mb": 12.0,
                "cuda_max_memory_allocated_mb": 16.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "validation_loss": validation_loss,
                "perplexity": 123.0,
                "inference_time_seconds": 0.2,
                "inference_tokens_per_second": 80.0,
                "inference_latency_ms_per_token": inference_latency_ms_per_token,
                "cuda_memory_allocated_mb": 10.0,
                "cuda_max_memory_allocated_mb": 14.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "routing_report.json", {"summary": routing_summary or {}, "latest_eval": {}})
    return run_dir


def test_estimate_gpu_hours_matches_formula() -> None:
    assert estimate_gpu_hours(100, 200, tflops_per_gpu=1.0, utilization=0.5, gamma=2.0) == pytest.approx(
        6 * 100 * 200 * 2 / 1e12 / 0.5 / 3600
    )


def test_summarize_run_estimates_active_route_compute(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "routed",
        config={
            "stage": "stage3_scheduled_free_routing",
            "batch_size": 2,
            "data_config_resolved": {"sequence_length": 8},
            "routing": {"mode": "scheduled", "hard_exit": False},
            "model_config_resolved": {"top_k": 2},
        },
        model_stats={
            "model_name": "brian_route_core",
            "parameter_count": 200,
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "top_k": 2,
        },
        routing_summary={
            "average_route_steps": 3.0,
            "active_block_evals_per_token": 0.5,
            "weighted_fusion_ratio": 1.0,
            "parallel_score_margin_mean": 0.25,
        },
    )
    summary = summarize_run(run, tflops_per_gpu=1.0, utilization=1.0)
    assert summary["physical_layer_count"] == 4
    assert summary["route_mode"] == "scheduled"
    assert summary["hard_exit_enabled"] is False
    assert summary["active_layer_evals_per_token"] == pytest.approx(5.0)
    assert summary["active_layer_ratio"] == pytest.approx(1.25)
    assert summary["trained_tokens_estimate"] == 32
    assert summary["estimated_flops_per_token"] == pytest.approx(1500.0)
    assert summary["routing"]["parallel_score_margin_mean"] == 0.25
    assert summary["train_step_time_seconds_latest"] == 0.1
    assert summary["train_latency_ms_per_token_mean"] == 0.5
    assert summary["inference_latency_ms_per_token_latest"] == 0.25
    assert summary["train_cuda_max_memory_allocated_mb_latest"] == 16.0


def test_summarize_run_uses_later_top_k_for_weighted_fusion_compute(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "later_topk",
        config={
            "stage": "stage3_scheduled_free_routing",
            "model_config_resolved": {"top_k": 1, "later_top_k": 2},
        },
        model_stats={
            "model_name": "brian_route_core",
            "parameter_count": 200,
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "top_k": 1,
            "later_top_k": 2,
        },
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 1.0,
            "weighted_fusion_ratio": 0.5,
        },
    )
    summary = summarize_run(run, tflops_per_gpu=1.0, utilization=1.0)

    assert summary["active_layer_evals_per_token"] == pytest.approx(5.0)


def test_summarize_run_marks_stage4_output_action_as_hard_exit_by_default(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "stage4",
        config={
            "stage": "stage4_output_action",
            "batch_size": 2,
            "data_config_resolved": {"sequence_length": 8},
            "routing": {"mode": "scheduled"},
            "model_config_resolved": {"hard_exit": False},
        },
        model_stats={"model_name": "stage4", "parameter_count": 100, "layers": 4},
    )
    summary = summarize_run(run)
    assert summary["hard_exit_enabled"] is True


def test_summarize_run_rejects_boolean_numeric_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "boolean_metrics"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_scheduled_free_routing",
                "batch_size": 2,
                "data_config_resolved": {"sequence_length": 8},
                "model_config_resolved": {"top_k": True},
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "model_stats.json",
        {"model_name": "boolean_metrics", "parameter_count": 100, "pre_blocks": 1, "route_pool_blocks": 2, "post_blocks": 1},
    )
    (run_dir / "train_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "loss": True,
                "tokens_per_second": True,
                "train_step_time_seconds": True,
                "train_latency_ms_per_token": True,
                "cuda_memory_allocated_mb": True,
                "cuda_max_memory_allocated_mb": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "validation_loss": True,
                "perplexity": True,
                "inference_time_seconds": True,
                "inference_tokens_per_second": True,
                "inference_latency_ms_per_token": True,
                "cuda_memory_allocated_mb": True,
                "cuda_max_memory_allocated_mb": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "routing_report.json",
        {
            "summary": {
                "average_route_steps": True,
                "active_block_evals_per_token": True,
                "weighted_fusion_ratio": True,
                "parallel_branch_count_mean": True,
                "parallel_score_margin_mean": True,
            }
        },
    )

    summary = summarize_run(run_dir)

    assert summary["validation_loss"] is None
    assert summary["train_loss"] is None
    assert summary["tokens_per_second_mean"] is None
    assert summary["inference_latency_ms_per_token_latest"] is None
    assert summary["train_cuda_max_memory_allocated_mb_latest"] is None
    assert summary["routing"]["average_route_steps"] is None
    assert summary["routing"]["parallel_branch_count_mean"] is None


def test_make_compute_report_compares_to_baseline(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        config={
            "stage": "stage0_baseline",
            "batch_size": 2,
            "data_config_resolved": {"sequence_length": 8},
        },
        model_stats={"model_name": "baseline", "parameter_count": 100, "layers": 4},
        validation_loss=10.0,
        tokens_per_second=200,
        train_latency_ms_per_token=0.5,
        inference_latency_ms_per_token=0.25,
    )
    routed = _write_run(
        tmp_path,
        "routed",
        config={
            "stage": "stage1_fixed_route",
            "batch_size": 2,
            "data_config_resolved": {"sequence_length": 8},
        },
        model_stats={
            "model_name": "brian_route_core",
            "parameter_count": 104,
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
        },
        routing_summary={"average_route_steps": 3.0, "active_block_evals_per_token": 2 / 3},
        validation_loss=10.2,
        tokens_per_second=150,
        train_latency_ms_per_token=0.75,
        inference_latency_ms_per_token=0.50,
    )
    output = tmp_path / "compute.json"
    report_path = make_compute_report([baseline, routed], baseline_run=baseline, output_path=output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_count"] == 2
    comparison = report["runs"][1]["baseline_comparison"]
    assert comparison["same_parameter_count_view"] is True
    assert comparison["active_layer_eval_ratio"] == pytest.approx(1.0)
    assert comparison["validation_loss_delta"] == pytest.approx(0.2)
    assert comparison["train_latency_ms_per_token_ratio"] == pytest.approx(1.5)
    assert comparison["inference_latency_ms_per_token_ratio"] == pytest.approx(2.0)
