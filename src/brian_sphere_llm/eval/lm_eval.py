from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.perplexity import perplexity
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.logging import write_json


DEFAULT_METRICS = [
    "validation_loss",
    "perplexity",
    "tokens_per_second",
    "active_block_evals_per_token",
]


def make_lm_eval_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    metrics: list[str] | None = None,
    downstream_report_paths: list[str | Path] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    requested_metrics = metrics or DEFAULT_METRICS
    train_rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    routing_report = _routing_report(run_dir)
    routing_summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    latest_train = train_rows[-1] if train_rows else {}
    latest_eval = eval_rows[-1] if eval_rows else {}
    metric_values = _metric_values(latest_eval, latest_train, routing_summary)
    selected_metrics = {name: metric_values.get(name) for name in requested_metrics}
    downstream_reports = [_downstream_report_row(Path(path)) for path in (downstream_report_paths or [])]
    downstream_summary = _downstream_summary(downstream_reports)
    checks = {
        "eval_log_present": bool(eval_rows),
        "validation_loss_present": _finite(metric_values.get("validation_loss")),
        "perplexity_present": _finite(metric_values.get("perplexity")),
        "requested_metrics_present": all(_finite(selected_metrics.get(name)) for name in requested_metrics),
        "downstream_reports_present": bool(downstream_reports),
        "downstream_task_accuracy_present": _finite(downstream_summary.get("downstream_task_accuracy_mean")),
        "benchmark_score_present": _finite(downstream_summary.get("benchmark_score")),
    }
    report = {
        "run_dir": str(run_dir),
        "requested_metrics": requested_metrics,
        "metrics": selected_metrics,
        "all_metrics": metric_values,
        "downstream_reports": downstream_reports,
        "downstream": downstream_summary,
        "checks": checks,
        "overall_status": _overall_status(checks, downstream_reports),
    }
    output_path = Path(output_path or run_dir / "lm_eval_report.json")
    write_json(report, output_path)
    return output_path


def _metric_values(
    latest_eval: dict[str, Any],
    latest_train: dict[str, Any],
    routing_summary: dict[str, Any],
) -> dict[str, float | None]:
    validation_loss = _num(latest_eval.get("validation_loss"))
    perplexity_value = _num(latest_eval.get("perplexity"))
    if perplexity_value is None and validation_loss is not None:
        perplexity_value = perplexity(validation_loss)
    values = {
        "validation_loss": validation_loss,
        "perplexity": perplexity_value,
        "tokens_per_second": _num(latest_train.get("tokens_per_second")),
        "train_step_time_seconds": _num(latest_train.get("train_step_time_seconds")),
        "train_latency_ms_per_token": _num(latest_train.get("train_latency_ms_per_token")),
        "inference_time_seconds": _num(latest_eval.get("inference_time_seconds")),
        "inference_tokens_per_second": _num(latest_eval.get("inference_tokens_per_second")),
        "inference_latency_ms_per_token": _num(latest_eval.get("inference_latency_ms_per_token")),
        "cuda_memory_allocated_mb": _num(latest_eval.get("cuda_memory_allocated_mb")),
        "cuda_max_memory_allocated_mb": _num(latest_eval.get("cuda_max_memory_allocated_mb")),
    }
    for key, value in routing_summary.items():
        numeric = _num(value)
        if numeric is not None:
            values[key] = numeric
    return values


def _downstream_report_row(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    overall = report.get("overall", {}) if isinstance(report.get("overall"), dict) else {}
    exact = _num(overall.get("exact_match_accuracy"))
    teacher = _num(overall.get("teacher_forced_token_accuracy"))
    explicit_score = _num(report.get("benchmark_score"))
    benchmark_score = explicit_score
    if benchmark_score is None:
        benchmark_score = exact if exact is not None else teacher
    return {
        "path": str(path),
        "run_dir": report.get("run_dir"),
        "sample_count": _num(report.get("sample_count")) or _num(overall.get("sample_count")),
        "exact_match_accuracy": exact,
        "teacher_forced_token_accuracy": teacher,
        "benchmark_score": benchmark_score,
    }


def _downstream_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    exact_values = [_num(row.get("exact_match_accuracy")) for row in rows]
    teacher_values = [_num(row.get("teacher_forced_token_accuracy")) for row in rows]
    score_values = [_num(row.get("benchmark_score")) for row in rows]
    return {
        "downstream_report_count": len(rows),
        "downstream_task_accuracy_mean": _mean([value for value in exact_values if value is not None]),
        "teacher_forced_token_accuracy_mean": _mean([value for value in teacher_values if value is not None]),
        "benchmark_score": _mean([value for value in score_values if value is not None]),
    }


def _overall_status(checks: dict[str, bool], downstream_reports: list[dict[str, Any]]) -> str:
    required = [
        checks["eval_log_present"],
        checks["validation_loss_present"],
        checks["perplexity_present"],
        checks["requested_metrics_present"],
    ]
    if not all(required):
        return "fail" if any(required) else "unknown"
    if downstream_reports and not checks["benchmark_score_present"]:
        return "warn"
    return "pass"


def _routing_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "routing_report.json"
    if not report_path.exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    if not report_path.exists():
        return {}
    return _read_json(report_path)


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _finite(value: Any) -> bool:
    numeric = _num(value)
    return numeric is not None and math.isfinite(numeric)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
