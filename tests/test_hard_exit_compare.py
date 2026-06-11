import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.hard_exit_compare import make_hard_exit_comparison_report


def _write_run(
    root: Path,
    name: str,
    *,
    hard_exit: bool,
    validation_loss: float,
    inference_time_seconds: float,
    inference_latency_ms_per_token: float,
    average_route_steps: float,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage4_output_action" if hard_exit else "stage4_scheduled_free_routing",
                "batch_size": 2,
                "data_config_resolved": {"sequence_length": 8},
                "routing": {"mode": "scheduled", "hard_exit": hard_exit},
                "model_config_resolved": {"top_k": 1, "hard_exit": hard_exit},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model_stats.json").write_text(
        json.dumps(
            {
                "model_name": name,
                "parameter_count": 100,
                "pre_blocks": 1,
                "route_pool_blocks": 2,
                "post_blocks": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train_log.jsonl").write_text(
        json.dumps({"step": 2, "loss": validation_loss + 1.0, "tokens_per_second": 100.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "validation_loss": validation_loss,
                "perplexity": 100.0,
                "inference_time_seconds": inference_time_seconds,
                "inference_tokens_per_second": 128.0,
                "inference_latency_ms_per_token": inference_latency_ms_per_token,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "average_route_steps": average_route_steps,
                    "active_block_evals_per_token": average_route_steps,
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_hard_exit_compare_passes_when_timing_and_route_steps_improve(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "without_hard_exit",
        hard_exit=False,
        validation_loss=10.0,
        inference_time_seconds=1.0,
        inference_latency_ms_per_token=0.50,
        average_route_steps=4.0,
    )
    hard_exit = _write_run(
        tmp_path,
        "with_hard_exit",
        hard_exit=True,
        validation_loss=10.0,
        inference_time_seconds=0.5,
        inference_latency_ms_per_token=0.25,
        average_route_steps=2.0,
    )

    output = make_hard_exit_comparison_report(
        baseline,
        [hard_exit],
        output_path=tmp_path / "hard_exit_compare.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]

    assert report["overall_status"] == "pass"
    assert report["baseline"]["hard_exit_enabled"] is False
    assert row["candidate"]["hard_exit_enabled"] is True
    assert row["checks"]["inference_timing_present"] is True
    assert row["baseline_comparison"]["inference_latency_ms_per_token_ratio"] == 0.5
    assert row["baseline_comparison"]["inference_time_seconds_ratio"] == 0.5
    assert row["baseline_comparison"]["average_route_steps_ratio"] == 0.5


def test_hard_exit_compare_warns_when_candidate_does_not_enable_hard_exit(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "without_hard_exit",
        hard_exit=False,
        validation_loss=10.0,
        inference_time_seconds=1.0,
        inference_latency_ms_per_token=0.50,
        average_route_steps=4.0,
    )
    candidate = _write_run(
        tmp_path,
        "also_without_hard_exit",
        hard_exit=False,
        validation_loss=10.0,
        inference_time_seconds=0.5,
        inference_latency_ms_per_token=0.25,
        average_route_steps=2.0,
    )

    output = make_hard_exit_comparison_report(
        baseline,
        [candidate],
        output_path=tmp_path / "hard_exit_compare.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]

    assert report["overall_status"] == "warn"
    assert row["checks"]["candidate_with_hard_exit"] is False
    assert row["checks"]["latency_ratio_within_threshold"] is True


def test_hard_exit_compare_rejects_boolean_numeric_metrics(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "without_hard_exit",
        hard_exit=False,
        validation_loss=10.0,
        inference_time_seconds=1.0,
        inference_latency_ms_per_token=0.50,
        average_route_steps=4.0,
    )
    candidate = _write_run(
        tmp_path,
        "with_boolean_metrics",
        hard_exit=True,
        validation_loss=10.0,
        inference_time_seconds=0.5,
        inference_latency_ms_per_token=0.25,
        average_route_steps=2.0,
    )
    (candidate / "eval_log.jsonl").write_text(
        json.dumps(
            {
                "step": 2,
                "validation_loss": True,
                "perplexity": True,
                "inference_time_seconds": True,
                "inference_tokens_per_second": True,
                "inference_latency_ms_per_token": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate / "routing_report.json").write_text(
        json.dumps({"summary": {"average_route_steps": True, "active_block_evals_per_token": True}}),
        encoding="utf-8",
    )

    output = make_hard_exit_comparison_report(
        baseline,
        [candidate],
        output_path=tmp_path / "hard_exit_compare.json",
    )
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]

    assert row["candidate"]["hard_exit_enabled"] is True
    assert row["candidate"]["inference_time_seconds_latest"] is None
    assert row["baseline_comparison"]["validation_loss_delta"] is None
    assert row["baseline_comparison"]["inference_latency_ms_per_token_ratio"] is None
    assert row["baseline_comparison"]["average_route_steps_ratio"] is None
    assert row["checks"]["inference_timing_present"] is False
    assert row["checks"]["validation_loss_not_worse"] is False
