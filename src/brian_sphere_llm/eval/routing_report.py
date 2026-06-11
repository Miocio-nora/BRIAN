from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


ROUTING_KEYS = {
    "route_entropy",
    "block_load_entropy",
    "block_load_entropy_normalized",
    "route_path_count",
    "route_path_diversity",
    "p_output_mean",
    "active_block_evals_per_token",
    "average_route_steps",
    "advance_ratio",
    "skip_ratio",
    "recur_ratio",
    "position_norm_mean",
    "location_distance_mean",
    "route_imitation_accuracy",
    "weighted_fusion_ratio",
    "global_attention_mass",
    "global_sink_attention_mass",
    "global_window_attention_mass",
    "global_read_gate_mean",
    "local_read_fraction_mean",
    "global_to_local_read_ratio",
    "local_to_global_read_ratio",
    "global_cache_slots_mean",
    "parallel_branch_count_mean",
    "parallel_score_margin_mean",
    "parallel_delta_cache_slots_mean",
    "parallel_delta_cache_slots_max",
    "max_route_steps",
    "forced_max_step_exit_count",
    "forced_max_step_exit_fraction",
}


def make_routing_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    aggregates: dict[str, list[float]] = defaultdict(list)
    latest_histogram: dict[str, Any] | None = None
    latest_exit_distribution: list[int] | None = None
    latest_first_exit_histogram: dict[str, Any] | None = None
    latest_route_path_examples: list[Any] | None = None
    latest_position_norm_trajectory: list[Any] | None = None
    latest_location_distance_trajectory: list[Any] | None = None
    for row in rows:
        for key in ROUTING_KEYS:
            value = _num(row.get(key))
            if value is not None and math.isfinite(value):
                aggregates[key].append(value)
        if isinstance(row.get("top1_block_histogram"), dict):
            latest_histogram = row["top1_block_histogram"]
        if isinstance(row.get("exit_step_distribution"), list):
            latest_exit_distribution = row["exit_step_distribution"]
        if isinstance(row.get("first_exit_step_histogram"), dict):
            latest_first_exit_histogram = row["first_exit_step_histogram"]
        if isinstance(row.get("route_path_examples"), list):
            latest_route_path_examples = row["route_path_examples"]
        if isinstance(row.get("position_norm_trajectory"), list):
            latest_position_norm_trajectory = row["position_norm_trajectory"]
        if isinstance(row.get("location_distance_trajectory"), list):
            latest_location_distance_trajectory = row["location_distance_trajectory"]
    summary = {key: sum(values) / max(1, len(values)) for key, values in aggregates.items()}
    cost_quality_curve = _cost_quality_curve(rows, eval_rows)
    checks = _report_checks(
        rows=rows,
        eval_rows=eval_rows,
        summary=summary,
        latest_histogram=latest_histogram,
        latest_exit_distribution=latest_exit_distribution,
        latest_first_exit_histogram=latest_first_exit_histogram,
        latest_route_path_examples=latest_route_path_examples,
        latest_position_norm_trajectory=latest_position_norm_trajectory,
        latest_location_distance_trajectory=latest_location_distance_trajectory,
        cost_quality_curve=cost_quality_curve,
    )
    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "latest_block_histogram": latest_histogram or {},
        "latest_exit_step_distribution": latest_exit_distribution or [],
        "latest_first_exit_step_histogram": latest_first_exit_histogram or {},
        "latest_route_path_examples": latest_route_path_examples or [],
        "latest_position_norm_trajectory": latest_position_norm_trajectory or [],
        "latest_location_distance_trajectory": latest_location_distance_trajectory or [],
        "cost_quality_curve": cost_quality_curve,
        "latest_eval": eval_rows[-1] if eval_rows else {},
        "checks": checks,
        "overall_status": _overall_status(checks),
    }
    output = run_dir / "routing_report.json"
    write_json(report, output)
    return output


def _report_checks(
    *,
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    summary: dict[str, float],
    latest_histogram: dict[str, Any] | None,
    latest_exit_distribution: list[int] | None,
    latest_first_exit_histogram: dict[str, Any] | None,
    latest_route_path_examples: list[Any] | None,
    latest_position_norm_trajectory: list[Any] | None,
    latest_location_distance_trajectory: list[Any] | None,
    cost_quality_curve: dict[str, Any],
) -> dict[str, bool]:
    curve_summary = cost_quality_curve.get("summary", {})
    latest_eval = eval_rows[-1] if eval_rows else {}
    train_points = cost_quality_curve.get("train_points", [])
    eval_points = cost_quality_curve.get("eval_points", [])
    return {
        "train_log_present": bool(rows),
        "eval_log_present": bool(eval_rows),
        "latest_eval_validation_loss_present": _finite(latest_eval.get("validation_loss")),
        "core_route_metrics_present": all(
            _finite(summary.get(key))
            for key in [
                "route_entropy",
                "block_load_entropy",
                "route_path_diversity",
                "active_block_evals_per_token",
                "average_route_steps",
            ]
        ),
        "route_transition_ratios_present": all(
            _finite(summary.get(key)) for key in ["advance_ratio", "skip_ratio", "recur_ratio"]
        ),
        "position_location_metrics_present": all(
            _finite(summary.get(key)) for key in ["position_norm_mean", "location_distance_mean"]
        ),
        "block_histogram_present": bool(latest_histogram),
        "exit_distribution_present": _numeric_sequence_present(latest_exit_distribution)
        or bool(latest_first_exit_histogram),
        "route_path_examples_present": bool(latest_route_path_examples),
        "position_trajectory_present": _numeric_sequence_present(latest_position_norm_trajectory),
        "location_trajectory_present": _numeric_sequence_present(latest_location_distance_trajectory),
        "cost_quality_train_points_present": int(curve_summary.get("train_point_count") or 0) >= 1,
        "cost_quality_eval_points_present": int(curve_summary.get("eval_point_count") or 0) >= 1,
        "training_timing_metrics_present": _points_have_metrics(
            train_points,
            ["tokens_per_second", "train_step_time_seconds", "train_latency_ms_per_token"],
        ),
        "inference_timing_metrics_present": _points_have_metrics(
            eval_points,
            ["inference_time_seconds", "inference_tokens_per_second", "inference_latency_ms_per_token"],
        ),
    }


def _overall_status(checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "pass"
    required_logs = [checks["train_log_present"], checks["eval_log_present"], checks["latest_eval_validation_loss_present"]]
    if not all(required_logs):
        return "fail" if any(required_logs) else "unknown"
    return "warn"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cost_quality_curve(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_points = [_train_curve_point(row) for row in rows]
    train_points = [point for point in train_points if point is not None]
    eval_points = []
    for eval_row in eval_rows:
        train_row = _matching_train_row(rows, eval_row)
        point = _eval_curve_point(eval_row, train_row)
        if point is not None:
            eval_points.append(point)
    return {
        "train_points": train_points,
        "eval_points": eval_points,
        "summary": {
            "train_point_count": len(train_points),
            "eval_point_count": len(eval_points),
            "active_compute_range": _range(
                [_num(point.get("active_block_evals_per_token")) for point in train_points + eval_points]
            ),
            "average_route_steps_range": _range(
                [_num(point.get("average_route_steps")) for point in train_points + eval_points]
            ),
            "train_loss_range": _range([_num(point.get("train_loss")) for point in train_points]),
            "validation_loss_range": _range([_num(point.get("validation_loss")) for point in eval_points]),
            "train_loss_vs_active_compute_correlation": _curve_correlation(
                train_points,
                x_key="active_block_evals_per_token",
                y_key="train_loss",
            ),
            "validation_loss_vs_active_compute_correlation": _curve_correlation(
                eval_points,
                x_key="active_block_evals_per_token",
                y_key="validation_loss",
            ),
        },
    }


def _train_curve_point(row: dict[str, Any]) -> dict[str, Any] | None:
    train_loss = _num(row.get("loss"))
    if train_loss is None:
        return None
    cost = _cost_metrics(row)
    if not cost:
        return None
    return {
        "step": _num(row.get("step")),
        "train_loss": train_loss,
        **cost,
    }


def _eval_curve_point(eval_row: dict[str, Any], train_row: dict[str, Any] | None) -> dict[str, Any] | None:
    validation_loss = _num(eval_row.get("validation_loss"))
    if validation_loss is None or train_row is None:
        return None
    cost = _cost_metrics(train_row)
    if not cost:
        return None
    return {
        "step": _num(eval_row.get("step")) or _num(train_row.get("step")),
        "validation_loss": validation_loss,
        "perplexity": _num(eval_row.get("perplexity")),
        "inference_time_seconds": _num(eval_row.get("inference_time_seconds")),
        "inference_tokens_per_second": _num(eval_row.get("inference_tokens_per_second")),
        "inference_latency_ms_per_token": _num(eval_row.get("inference_latency_ms_per_token")),
        **cost,
    }


def _cost_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "active_block_evals_per_token": _num(row.get("active_block_evals_per_token")),
        "average_route_steps": _num(row.get("average_route_steps")),
        "p_output_mean": _num(row.get("p_output_mean")),
        "tokens_per_second": _num(row.get("tokens_per_second")),
        "train_step_time_seconds": _num(row.get("train_step_time_seconds")),
        "train_latency_ms_per_token": _num(row.get("train_latency_ms_per_token")),
        "cuda_memory_allocated_mb": _num(row.get("cuda_memory_allocated_mb")),
        "cuda_max_memory_allocated_mb": _num(row.get("cuda_max_memory_allocated_mb")),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _matching_train_row(rows: list[dict[str, Any]], eval_row: dict[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    eval_step = _num(eval_row.get("step"))
    if eval_step is None:
        return rows[-1]
    candidates = [row for row in rows if (_num(row.get("step")) is not None and _num(row.get("step")) <= eval_step)]
    if candidates:
        return candidates[-1]
    return rows[0]


def _curve_correlation(points: list[dict[str, Any]], *, x_key: str, y_key: str) -> float | None:
    pairs = []
    for point in points:
        x_value = _num(point.get(x_key))
        y_value = _num(point.get(y_key))
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    if len(pairs) < 2:
        return None
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_dev = [value - x_mean for value in x_values]
    y_dev = [value - y_mean for value in y_values]
    x_denom = math.sqrt(sum(value * value for value in x_dev))
    y_denom = math.sqrt(sum(value * value for value in y_dev))
    if x_denom == 0.0 or y_denom == 0.0:
        return None
    return sum(x * y for x, y in zip(x_dev, y_dev)) / (x_denom * y_denom)


def _range(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return max(finite) - min(finite)


def _finite(value: Any) -> bool:
    numeric = _num(value)
    return numeric is not None and math.isfinite(numeric)


def _numeric_sequence_present(values: list[Any] | None) -> bool:
    return bool(values) and all(_finite(value) for value in values)


def _points_have_metrics(points: Any, keys: list[str]) -> bool:
    if not isinstance(points, list):
        return False
    return any(all(_finite(point.get(key)) for key in keys) for point in points if isinstance(point, dict))


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
