from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json


GLOBAL_KV_KEYS = [
    "global_attention_mass",
    "global_sink_attention_mass",
    "global_window_attention_mass",
    "global_read_gate_mean",
    "global_cache_slots_mean",
]


def make_global_kv_retention_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    min_global_attention_mass: float = 1e-6,
    min_global_read_gate: float = 1e-6,
    mass_tolerance: float = 1e-5,
    capacity_slack: float = 1e-6,
) -> Path:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    model_config = _model_config(config)
    routing_report = _routing_report(run_dir)
    summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    latest_eval = routing_report.get("latest_eval", {}) if isinstance(routing_report.get("latest_eval"), dict) else {}
    metrics, metric_sources = _global_kv_metrics(summary, latest_eval)

    global_kv_enabled = _bool(model_config.get("global_kv", False))
    sink_slots = int(_num(model_config.get("global_sink_slots")) or 0)
    window_slots = int(_num(model_config.get("global_window_slots")) or 0)
    retention_capacity = sink_slots + window_slots if global_kv_enabled else 0
    attention_mass = metrics["global_attention_mass"]
    sink_mass = metrics["global_sink_attention_mass"]
    window_mass = metrics["global_window_attention_mass"]
    read_gate = metrics["global_read_gate_mean"]
    cache_slots = metrics["global_cache_slots_mean"]

    checks = {
        "stage5_global_kv_stage": str(config.get("stage", "")) == "stage5_global_kv",
        "global_kv_enabled": global_kv_enabled,
        "sink_slots_configured": sink_slots > 0,
        "window_slots_configured": window_slots > 0,
        "retention_capacity_present": retention_capacity > 0,
        "global_attention_mass_nonzero": _at_least(attention_mass, min_global_attention_mass),
        "global_read_gate_nonzero": _at_least(read_gate, min_global_read_gate),
        "global_cache_slots_present": _at_least(cache_slots, 1e-12),
        "sink_attention_mass_measured": _bounded_mass(sink_mass),
        "window_attention_mass_measured": _bounded_mass(window_mass),
        "sink_window_mass_conserved": _mass_conserved(attention_mass, sink_mass, window_mass, mass_tolerance),
        "cache_slots_within_retention_capacity": (
            _finite(cache_slots) and retention_capacity > 0 and float(cache_slots) <= retention_capacity + capacity_slack
        ),
    }
    report = {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "model": {
            "global_kv_enabled": global_kv_enabled,
            "global_sink_slots": sink_slots,
            "global_window_slots": window_slots,
            "retention_capacity_slots": retention_capacity,
        },
        "metrics": metrics,
        "metric_sources": metric_sources,
        "thresholds": {
            "min_global_attention_mass": min_global_attention_mass,
            "min_global_read_gate": min_global_read_gate,
            "mass_tolerance": mass_tolerance,
            "capacity_slack": capacity_slack,
        },
        "checks": checks,
        "overall_status": "pass" if all(checks.values()) else "fail",
    }
    if output_path is None:
        output_path = run_dir / "global_kv_retention_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model_config_resolved")
    if isinstance(model_config, dict):
        return model_config
    return config


def _routing_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "routing_report.json"
    if not report_path.exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    return _read_json(report_path)


def _global_kv_metrics(
    summary: dict[str, Any],
    latest_eval: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    metrics: dict[str, float | None] = {}
    sources: dict[str, str | None] = {}
    for key in GLOBAL_KV_KEYS:
        value = _num(summary.get(key))
        if value is not None and math.isfinite(value):
            metrics[key] = value
            sources[key] = "routing_summary"
            continue
        value = _num(latest_eval.get(key))
        if value is not None and math.isfinite(value):
            metrics[key] = value
            sources[key] = "latest_eval"
            continue
        metrics[key] = None
        sources[key] = None
    return metrics, sources


def _at_least(value: float | None, minimum: float) -> bool:
    return _finite(value) and float(value) >= minimum


def _bounded_mass(value: float | None) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def _mass_conserved(
    attention_mass: float | None,
    sink_mass: float | None,
    window_mass: float | None,
    tolerance: float,
) -> bool:
    if not (_finite(attention_mass) and _finite(sink_mass) and _finite(window_mass)):
        return False
    return abs(float(attention_mass) - float(sink_mass) - float(window_mass)) <= tolerance


def _finite(value: float | None) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on", "enabled"}


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data
