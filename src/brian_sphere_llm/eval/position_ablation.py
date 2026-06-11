from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.logging import write_json


ROUTING_METRICS = [
    "average_route_steps",
    "active_block_evals_per_token",
    "route_entropy",
    "block_load_entropy",
    "route_path_diversity",
    "position_norm_mean",
    "location_distance_mean",
    "route_imitation_accuracy",
]


def make_position_ablation_report(
    reference_run: str | Path,
    candidate_runs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    min_validation_loss_delta: float = 0.001,
    min_routing_metric_delta: float = 0.001,
) -> Path:
    reference = _summarize_run(Path(reference_run))
    candidates = [_summarize_run(Path(run_dir)) for run_dir in candidate_runs]
    comparisons = [
        _compare(reference, candidate, min_validation_loss_delta, min_routing_metric_delta)
        for candidate in candidates
    ]
    any_measurable = any(item["checks"]["measurable_difference"] for item in comparisons)
    checks = {
        "candidate_present": bool(comparisons),
        "any_measurable_difference": any_measurable,
    }
    report = {
        "overall_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "reference_run": reference,
        "candidate_count": len(candidates),
        "comparisons": comparisons,
        "thresholds": {
            "min_validation_loss_delta": min_validation_loss_delta,
            "min_routing_metric_delta": min_routing_metric_delta,
        },
    }
    if output_path is None:
        output_path = Path("reports") / "position_ablation_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "routing_report.json").exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    config = _read_yaml_if_exists(run_dir / "config_resolved.yaml")
    model_stats = _read_json_if_exists(run_dir / "model_stats.json")
    routing_report = _read_json_if_exists(run_dir / "routing_report.json")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    latest_eval = eval_rows[-1] if eval_rows else routing_report.get("latest_eval", {})
    routing_summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    return {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "model_name": str(model_stats.get("model_name", "")),
        "validation_loss": _num(latest_eval.get("validation_loss")),
        "perplexity": _num(latest_eval.get("perplexity")),
        "routing": {key: _num(routing_summary.get(key)) for key in ROUTING_METRICS},
    }


def _compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    min_validation_loss_delta: float,
    min_routing_metric_delta: float,
) -> dict[str, Any]:
    validation_delta = _delta(candidate.get("validation_loss"), reference.get("validation_loss"))
    routing_deltas = {
        key: _delta(candidate.get("routing", {}).get(key), reference.get("routing", {}).get(key))
        for key in ROUTING_METRICS
    }
    measurable_routing = {
        key: value
        for key, value in routing_deltas.items()
        if value is not None and abs(value) > min_routing_metric_delta
    }
    validation_measurable = validation_delta is not None and abs(validation_delta) > min_validation_loss_delta
    routing_measurable = bool(measurable_routing)
    checks = {
        "validation_loss_delta_measurable": validation_measurable,
        "routing_metric_delta_measurable": routing_measurable,
        "measurable_difference": validation_measurable or routing_measurable,
    }
    return {
        "run_dir": candidate["run_dir"],
        "stage": candidate.get("stage"),
        "model_name": candidate.get("model_name"),
        "status": "pass" if checks["measurable_difference"] else "fail",
        "checks": checks,
        "validation_loss_delta": validation_delta,
        "routing_metric_deltas": routing_deltas,
        "measurable_routing_metric_deltas": measurable_routing,
    }


def _delta(value: Any, baseline: Any) -> float | None:
    left = _num(value)
    right = _num(baseline)
    if left is None or right is None:
        return None
    return left - right


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
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
