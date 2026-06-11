import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.parallel_compare import make_parallel_comparison_report


def _write_run(
    root: Path,
    name: str,
    *,
    stage: str,
    validation_loss: float,
    tokens_per_second: int,
    routing_summary: dict,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": stage,
                "batch_size": 2,
                "data_config_resolved": {"sequence_length": 8},
                "model_config_resolved": {"top_k": 2},
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
                "top_k": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train_log.jsonl").write_text(
        json.dumps({"step": 2, "loss": validation_loss + 1.0, "tokens_per_second": tokens_per_second}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"step": 2, "validation_loss": validation_loss, "perplexity": 100.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(
        json.dumps({"summary": routing_summary, "latest_eval": {}}),
        encoding="utf-8",
    )
    return run_dir


def test_parallel_compare_passes_bounded_parallel_candidate(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "topk",
        stage="stage5_global_kv",
        validation_loss=10.0,
        tokens_per_second=100,
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 0.5,
            "weighted_fusion_ratio": 1.0,
        },
    )
    candidate = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        validation_loss=9.9,
        tokens_per_second=120,
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 0.25,
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
        },
    )
    output = make_parallel_comparison_report(baseline, [candidate], output_path=tmp_path / "parallel_compare.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["comparisons"][0]
    assert report["overall_status"] == "pass"
    assert row["checks"]["parallel_branch_active"] is True
    assert row["checks"]["parallel_score_margin_present"] is True
    assert row["checks"]["quality_not_worse"] is True
    assert row["checks"]["parallel_branch_benefit_proxy"] is True
    assert row["baseline_comparison"]["validation_loss_delta"] < 0.0


def test_parallel_compare_warns_without_parallel_metrics(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "topk",
        stage="stage5_global_kv",
        validation_loss=10.0,
        tokens_per_second=100,
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 0.5,
            "weighted_fusion_ratio": 1.0,
        },
    )
    candidate = _write_run(
        tmp_path,
        "not_parallel",
        stage="stage6_parallel_passing",
        validation_loss=10.5,
        tokens_per_second=100,
        routing_summary={"average_route_steps": 2.0},
    )
    output = make_parallel_comparison_report(
        baseline,
        [candidate],
        output_path=tmp_path / "parallel_compare.json",
        max_validation_loss_delta=0.0,
    )
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]
    assert row["status"] == "warn"
    assert row["checks"]["parallel_branch_active"] is False
    assert row["checks"]["parallel_score_margin_present"] is False
    assert row["checks"]["quality_not_worse"] is False


def test_parallel_compare_rejects_boolean_parallel_metrics(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "topk",
        stage="stage5_global_kv",
        validation_loss=10.0,
        tokens_per_second=100,
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 0.5,
            "weighted_fusion_ratio": 1.0,
        },
    )
    candidate = _write_run(
        tmp_path,
        "parallel_bool",
        stage="stage6_parallel_passing",
        validation_loss=9.9,
        tokens_per_second=120,
        routing_summary={
            "average_route_steps": 2.0,
            "active_block_evals_per_token": 0.25,
            "parallel_branch_count_mean": True,
            "parallel_score_margin_mean": True,
        },
    )

    output = make_parallel_comparison_report(baseline, [candidate], output_path=tmp_path / "parallel_compare.json")
    row = json.loads(output.read_text(encoding="utf-8"))["comparisons"][0]

    assert row["status"] == "warn"
    assert row["parallel"]["parallel_branch_count_mean"] is None
    assert row["parallel"]["parallel_score_margin_mean"] is None
    assert row["checks"]["parallel_branch_active"] is False
    assert row["checks"]["parallel_score_margin_present"] is False
