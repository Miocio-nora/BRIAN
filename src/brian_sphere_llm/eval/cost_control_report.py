from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.eval.stage_gate_report import pearson_correlation
from brian_sphere_llm.utils.logging import write_json


def make_cost_control_report(
    run_dirs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    min_active_compute_range: float = 0.05,
) -> Path:
    summaries = [_summarize_run(Path(run_dir)) for run_dir in run_dirs]
    summaries.sort(key=lambda row: (row["cost_weight"] is None, row["cost_weight"] if row["cost_weight"] is not None else 0.0))
    analysis = _analyze_cost_sweep(summaries, min_active_compute_range=min_active_compute_range)
    report = {
        "run_count": len(summaries),
        "runs": summaries,
        "analysis": analysis,
        "min_active_compute_range": min_active_compute_range,
    }
    if output_path is None:
        output_path = Path("reports") / "cost_control_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "routing_report.json").exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    config = _read_yaml_if_exists(run_dir / "config_resolved.yaml")
    routing_config = config.get("routing", {}) if isinstance(config.get("routing"), dict) else {}
    routing_report = _read_json_if_exists(run_dir / "routing_report.json")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    routing_summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    latest_eval = eval_rows[-1] if eval_rows else {}
    return {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "routing_mode": routing_config.get("mode"),
        "hard_exit": routing_config.get("hard_exit"),
        "cost_weight": _num(config.get("loss_weights", {}).get("cost") if isinstance(config.get("loss_weights"), dict) else None),
        "validation_loss": _num(latest_eval.get("validation_loss")),
        "perplexity": _num(latest_eval.get("perplexity")),
        "average_route_steps": _num(routing_summary.get("average_route_steps")),
        "active_block_evals_per_token": _num(routing_summary.get("active_block_evals_per_token")),
        "p_output_mean": _num(routing_summary.get("p_output_mean")),
        "route_entropy": _num(routing_summary.get("route_entropy")),
        "first_exit_step_histogram": _latest_exit_hist(run_dir),
    }


def _analyze_cost_sweep(rows: list[dict[str, Any]], *, min_active_compute_range: float) -> dict[str, Any]:
    valid = [row for row in rows if row.get("cost_weight") is not None]
    cost_values = [float(row["cost_weight"]) for row in valid]
    average_steps = [_num(row.get("average_route_steps")) for row in valid]
    active_evals = [_num(row.get("active_block_evals_per_token")) for row in valid]
    p_output = [_num(row.get("p_output_mean")) for row in valid]
    step_pairs = _finite_pairs(cost_values, average_steps)
    active_pairs = _finite_pairs(cost_values, active_evals)
    output_pairs = _finite_pairs(cost_values, p_output)
    active_range = _range([value for _, value in active_pairs])
    step_range = _range([value for _, value in step_pairs])
    p_output_range = _range([value for _, value in output_pairs])
    active_corr = _corr(active_pairs)
    step_corr = _corr(step_pairs)
    output_corr = _corr(output_pairs)
    checks = {
        "stage4_output_action_runs": bool(rows) and all(row.get("stage") == "stage4_output_action" for row in rows),
        "hard_exit_enabled": bool(rows) and all(row.get("hard_exit") is True for row in rows),
        "has_multiple_cost_weights": len(set(cost_values)) >= 2,
        "active_compute_range_present": active_range is not None and active_range >= min_active_compute_range,
        "active_compute_not_increasing_with_cost": active_corr is not None and active_corr <= 0.0,
        "average_steps_not_increasing_with_cost": step_corr is not None and step_corr <= 0.0,
        "output_probability_not_decreasing_with_cost": output_corr is not None and output_corr >= 0.0,
    }
    return {
        "cost_values": cost_values,
        "distinct_cost_weights": sorted(set(cost_values)),
        "average_route_steps_range": step_range,
        "active_block_evals_range": active_range,
        "p_output_mean_range": p_output_range,
        "cost_vs_average_route_steps_correlation": step_corr,
        "cost_vs_active_block_evals_correlation": active_corr,
        "cost_vs_p_output_correlation": output_corr,
        "average_route_steps_monotonic_nonincreasing": _monotonic_nonincreasing(step_pairs),
        "active_block_evals_monotonic_nonincreasing": _monotonic_nonincreasing(active_pairs),
        "p_output_monotonic_nondecreasing": _monotonic_nondecreasing(output_pairs),
        "checks": checks,
        "status": _status(checks),
    }


def _status(checks: dict[str, bool]) -> str:
    if checks.get("has_multiple_cost_weights") is False:
        return "fail"
    if all(value is True for value in checks.values()):
        return "pass"
    if any(value is True for value in checks.values()):
        return "warn"
    return "fail"


def _finite_pairs(x_values: list[float], y_values: list[float | None]) -> list[tuple[float, float]]:
    pairs = []
    for x_value, y_value in zip(x_values, y_values):
        if y_value is not None and math.isfinite(y_value):
            pairs.append((x_value, float(y_value)))
    return pairs


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    return pearson_correlation([x for x, _ in pairs], [y for _, y in pairs])


def _range(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _monotonic_nonincreasing(pairs: list[tuple[float, float]]) -> bool | None:
    if len(pairs) < 2:
        return None
    ordered = [value for _, value in sorted(pairs)]
    return all(next_value <= value for value, next_value in zip(ordered, ordered[1:]))


def _monotonic_nondecreasing(pairs: list[tuple[float, float]]) -> bool | None:
    if len(pairs) < 2:
        return None
    ordered = [value for _, value in sorted(pairs)]
    return all(next_value >= value for value, next_value in zip(ordered, ordered[1:]))


def _latest_exit_hist(run_dir: Path) -> dict[str, int]:
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    for row in reversed(rows):
        hist = row.get("first_exit_step_histogram")
        if isinstance(hist, dict):
            counts = {}
            for key, value in hist.items():
                count = _num(value)
                if count is not None:
                    counts[str(key)] = int(count)
            return counts
    return {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}
