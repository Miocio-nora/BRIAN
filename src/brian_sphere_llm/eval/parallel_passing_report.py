from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json


def make_parallel_passing_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    max_beam_size: int = 2,
    min_parallel_branch_count: float = 1.5,
    min_branch_cost: float = 0.0,
    tolerance: float = 1e-6,
) -> Path:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    model_config = _model_config(config)
    routing_report = _routing_report(run_dir)
    train_rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    latest_eval = routing_report.get("latest_eval", {}) if isinstance(routing_report.get("latest_eval"), dict) else {}

    branch_counts = _series(train_rows, eval_rows, summary, latest_eval, "parallel_branch_count_mean")
    score_margins = _series(train_rows, eval_rows, summary, latest_eval, "parallel_score_margin_mean")
    delta_cache_slots = _series(train_rows, eval_rows, summary, latest_eval, "parallel_delta_cache_slots_max")
    beam_size = int(_num(model_config.get("beam_size")) or 0)
    branch_cost = _num(model_config.get("branch_cost"))
    branch_score_decay = _num(model_config.get("branch_score_decay"))
    global_kv_enabled = _bool_value(model_config.get("global_kv", False), "model_config_resolved.global_kv")
    parallel_passing_enabled = _bool_value(
        model_config.get("parallel_passing", False),
        "model_config_resolved.parallel_passing",
    )
    global_window_slots = int(_num(model_config.get("global_window_slots")) or 0)
    parallel_stage = str(config.get("stage", "")).startswith("stage6") or str(config.get("stage", "")).startswith("stage7")
    checks = {
        "stage6_parallel_stage": parallel_stage,
        "parallel_passing_enabled": parallel_passing_enabled,
        "parallel_route_selected": parallel_stage or _routing_mode(config) == "parallel",
        "shared_base_global_memory_enabled": global_kv_enabled,
        "beam_size_present": beam_size >= 1,
        "beam_size_within_limit": beam_size >= 1 and beam_size <= max_beam_size,
        "branch_cost_enabled": branch_cost is not None and branch_cost > min_branch_cost,
        "branch_score_decay_configured": branch_score_decay is not None and 0.0 < branch_score_decay < 1.0,
        "branch_metrics_present": bool(branch_counts),
        "parallel_branch_active": _max(branch_counts) is not None and float(_max(branch_counts)) >= min_parallel_branch_count,
        "branch_count_bounded_by_beam": (
            bool(branch_counts) and beam_size >= 1 and float(_max(branch_counts) or 0.0) <= beam_size + tolerance
        ),
        "score_margin_measured": bool(score_margins),
        "score_margin_nonnegative": bool(score_margins)
        and _min(score_margins) is not None
        and float(_min(score_margins) or 0.0) >= -tolerance,
        "branch_delta_memory_measured": global_kv_enabled and bool(delta_cache_slots),
        "delta_cache_nonnegative": (not global_kv_enabled)
        or (
            bool(delta_cache_slots)
            and _min(delta_cache_slots) is not None
            and float(_min(delta_cache_slots) or 0.0) >= -tolerance
        ),
        "delta_memory_policy_present": (not global_kv_enabled) or bool(delta_cache_slots),
        "delta_cache_bounded_by_window": (not global_kv_enabled)
        or (
            bool(delta_cache_slots)
            and global_window_slots >= 0
            and float(_max(delta_cache_slots) or 0.0) <= global_window_slots + tolerance
        ),
    }
    report = {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "model": {
            "parallel_passing_enabled": parallel_passing_enabled,
            "beam_size": beam_size,
            "branch_cost": branch_cost,
            "branch_score_decay": branch_score_decay,
            "parallel_exit_policy": str(model_config.get("parallel_exit_policy", "branch")),
            "global_kv_enabled": global_kv_enabled,
            "global_sink_slots": int(_num(model_config.get("global_sink_slots")) or 0),
            "global_window_slots": global_window_slots,
            "memory_policy": "shared_base_global_kv_with_branch_delta" if global_kv_enabled else "local_only",
        },
        "routing": {
            "mode": _routing_mode(config),
            "parallel_branch_count": _summary(branch_counts),
            "parallel_score_margin": _summary(score_margins),
            "parallel_delta_cache_slots": _summary(delta_cache_slots),
        },
        "thresholds": {
            "max_beam_size": max_beam_size,
            "min_parallel_branch_count": min_parallel_branch_count,
            "min_branch_cost": min_branch_cost,
            "tolerance": tolerance,
        },
        "checks": checks,
        "overall_status": "pass" if all(checks.values()) else "fail",
    }
    if output_path is None:
        output_path = run_dir / "parallel_passing_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model_config_resolved")
    return model_config if isinstance(model_config, dict) else {}


def _routing_mode(config: dict[str, Any]) -> str:
    routing = config.get("routing", {})
    if not isinstance(routing, dict):
        return ""
    return str(routing.get("mode", ""))


def _routing_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "routing_report.json"
    if not report_path.exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    return _read_json_if_exists(report_path)


def _series(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    latest_eval: dict[str, Any],
    key: str,
) -> list[float]:
    values = [_num(row.get(key)) for row in train_rows + eval_rows]
    values.extend([_num(summary.get(key)), _num(latest_eval.get(key))])
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "latest": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "latest": values[-1],
    }


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


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
