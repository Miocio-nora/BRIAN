import json
from pathlib import Path

from brian_sphere_llm.eval.lm_eval import make_lm_eval_report


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_lm_eval_report_summarizes_standard_and_downstream_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {
                "step": 2,
                "loss": 3.0,
                "tokens_per_second": 128.0,
                "train_latency_ms_per_token": 0.5,
            }
        ],
    )
    _write_jsonl(
        run_dir / "eval_log.jsonl",
        [
            {
                "step": 2,
                "validation_loss": 2.0,
                "perplexity": 7.4,
                "inference_tokens_per_second": 64.0,
                "inference_latency_ms_per_token": 1.0,
            }
        ],
    )
    _write_json(
        run_dir / "routing_report.json",
        {"summary": {"active_block_evals_per_token": 0.5, "average_route_steps": 2.0}},
    )
    downstream = tmp_path / "reasoning_report.json"
    _write_json(
        downstream,
        {
            "run_dir": str(run_dir),
            "sample_count": 2,
            "overall": {
                "exact_match_accuracy": 0.75,
                "teacher_forced_token_accuracy": 0.9,
            },
        },
    )

    output = make_lm_eval_report(
        run_dir,
        output_path=tmp_path / "lm_eval_report.json",
        metrics=["validation_loss", "perplexity", "tokens_per_second", "active_block_evals_per_token"],
        downstream_report_paths=[downstream],
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["metrics"]["validation_loss"] == 2.0
    assert report["metrics"]["perplexity"] == 7.4
    assert report["metrics"]["tokens_per_second"] == 128.0
    assert report["metrics"]["active_block_evals_per_token"] == 0.5
    assert report["downstream"]["downstream_report_count"] == 1
    assert report["downstream"]["downstream_task_accuracy_mean"] == 0.75
    assert report["downstream"]["teacher_forced_token_accuracy_mean"] == 0.9
    assert report["downstream"]["benchmark_score"] == 0.75
    assert report["checks"]["benchmark_score_present"] is True


def test_lm_eval_report_fails_missing_requested_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "eval_log.jsonl", [{"validation_loss": 2.0}])

    output = make_lm_eval_report(
        run_dir,
        output_path=tmp_path / "lm_eval_report.json",
        metrics=["validation_loss", "active_block_evals_per_token"],
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["requested_metrics_present"] is False
    assert report["metrics"]["active_block_evals_per_token"] is None


def test_lm_eval_report_rejects_boolean_requested_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "train_log.jsonl", [{"tokens_per_second": True}])
    _write_jsonl(run_dir / "eval_log.jsonl", [{"validation_loss": 2.0, "perplexity": 7.4}])

    output = make_lm_eval_report(
        run_dir,
        output_path=tmp_path / "lm_eval_report.json",
        metrics=["validation_loss", "perplexity", "tokens_per_second"],
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["requested_metrics_present"] is False
    assert report["metrics"]["tokens_per_second"] is None
