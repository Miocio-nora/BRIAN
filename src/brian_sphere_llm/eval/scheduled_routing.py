from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brian_sphere_llm.routing.schedule import scheduled_value
from brian_sphere_llm.utils.logging import write_json


def make_scheduled_routing_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    min_final_router_probability: float = 1.0,
    tolerance: float = 1e-9,
) -> Path:
    run_dir = Path(run_dir)
    config = _read_yaml(run_dir / "config_resolved.yaml")
    routing_config = config.get("routing", {}) if isinstance(config.get("routing"), dict) else {}
    schedule_config = routing_config.get("schedule", [])
    schedule = schedule_config if isinstance(schedule_config, list) else []
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    schedule_points = _schedule_points(schedule)
    logged_rows = [row for row in rows if _has_logged_schedule(row)]
    logged_eval_rows = [row for row in eval_rows if _has_logged_schedule(row)]
    checks = {
        "scheduled_stage": str(config.get("stage", "")).startswith("stage3") or routing_config.get("mode") == "scheduled",
        "schedule_present": bool(schedule),
        "schedule_values_numeric": bool(schedule_points) and len(schedule_points) == len(schedule),
        "router_probability_monotonic_nondecreasing": _monotonic(
            [point["router_probability"] for point in schedule_points],
            direction="nondecreasing",
        ),
        "lambda_route_monotonic_nonincreasing": _monotonic(
            [point["lambda_route"] for point in schedule_points],
            direction="nonincreasing",
        ),
        "router_probability_increases": _increases([point["router_probability"] for point in schedule_points]),
        "lambda_route_decays": _decays([point["lambda_route"] for point in schedule_points]),
        "reaches_free_router": bool(schedule_points)
        and max(point["router_probability"] for point in schedule_points) >= min_final_router_probability,
        "logged_schedule_values_present": bool(logged_rows),
        "logged_router_probability_matches_schedule": _logged_matches(
            logged_rows,
            schedule_points,
            key="scheduled_router_probability",
            schedule_key="router_probability",
            default=0.0,
            tolerance=tolerance,
        ),
        "logged_lambda_route_matches_schedule": _logged_matches(
            logged_rows,
            schedule_points,
            key="scheduled_lambda_route",
            schedule_key="lambda_route",
            default=_default_lambda_route(config),
            tolerance=tolerance,
        ),
        "logged_eval_schedule_values_present": bool(logged_eval_rows),
        "logged_eval_router_probability_matches_schedule": _logged_matches(
            logged_eval_rows,
            schedule_points,
            key="scheduled_router_probability",
            schedule_key="router_probability",
            default=0.0,
            tolerance=tolerance,
        ),
        "logged_eval_lambda_route_matches_schedule": _logged_matches(
            logged_eval_rows,
            schedule_points,
            key="scheduled_lambda_route",
            schedule_key="lambda_route",
            default=_default_lambda_route(config),
            tolerance=tolerance,
        ),
    }
    report = {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "schedule": schedule_points,
        "train_step_count": len(rows),
        "logged_schedule_step_count": len(logged_rows),
        "eval_step_count": len(eval_rows),
        "logged_eval_schedule_step_count": len(logged_eval_rows),
        "logged_schedule_values": _logged_schedule_values(logged_rows),
        "logged_eval_schedule_values": _logged_schedule_values(logged_eval_rows),
        "latest_eval_schedule_values": _latest_eval_schedule_values(eval_rows),
        "thresholds": {
            "min_final_router_probability": min_final_router_probability,
            "tolerance": tolerance,
        },
        "checks": checks,
        "overall_status": "pass" if all(checks.values()) else "fail",
    }
    if output_path is None:
        output_path = run_dir / "scheduled_routing_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _schedule_points(schedule: list[Any]) -> list[dict[str, float | int]]:
    return [point for item in schedule if (point := _schedule_point(item)) is not None]


def _schedule_point(item: Any) -> dict[str, float | int] | None:
    if not isinstance(item, dict):
        return None
    max_step = _num(item.get("max_step"))
    router_probability = _num(item.get("router_probability"))
    lambda_route = _num(item.get("lambda_route"))
    if max_step is None or router_probability is None or lambda_route is None:
        return None
    return {
        "max_step": int(max_step),
        "router_probability": float(router_probability),
        "lambda_route": float(lambda_route),
    }


def _has_logged_schedule(row: dict[str, Any]) -> bool:
    return (
        _num(row.get("scheduled_router_probability")) is not None
        and _num(row.get("scheduled_lambda_route")) is not None
    )


def _logged_matches(
    rows: list[dict[str, Any]],
    schedule: list[dict[str, float | int]],
    *,
    key: str,
    schedule_key: str,
    default: float,
    tolerance: float,
) -> bool:
    if not rows:
        return False
    for row in rows:
        step = int(row.get("step", 0))
        expected = scheduled_value(schedule, step, schedule_key, default)
        actual = _num(row.get(key))
        if actual is None:
            return False
        if abs(actual - expected) > tolerance:
            return False
    return True


def _logged_schedule_values(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    values = []
    for row in rows:
        router_probability = _num(row.get("scheduled_router_probability"))
        lambda_route = _num(row.get("scheduled_lambda_route"))
        if router_probability is None or lambda_route is None:
            continue
        values.append(
            {
                "step": int(row.get("step", 0)),
                "scheduled_router_probability": float(router_probability),
                "scheduled_lambda_route": float(lambda_route),
            }
        )
    return values


def _latest_eval_schedule_values(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    for row in reversed(rows):
        if _has_logged_schedule(row):
            router_probability = _num(row.get("scheduled_router_probability"))
            lambda_route = _num(row.get("scheduled_lambda_route"))
            if router_probability is None or lambda_route is None:
                continue
            return {
                "scheduled_router_probability": float(router_probability),
                "scheduled_lambda_route": float(lambda_route),
            }
    return None


def _default_lambda_route(config: dict[str, Any]) -> float:
    loss_weights = config.get("loss_weights", {})
    return float(_num(loss_weights.get("route")) or 0.0) if isinstance(loss_weights, dict) else 0.0


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _monotonic(values: list[float], *, direction: str) -> bool:
    if not values:
        return False
    if direction == "nondecreasing":
        return all(next_value >= value for value, next_value in zip(values, values[1:]))
    if direction == "nonincreasing":
        return all(next_value <= value for value, next_value in zip(values, values[1:]))
    raise ValueError(f"Unknown monotonic direction: {direction}")


def _increases(values: list[float]) -> bool:
    return len(values) >= 2 and values[-1] > values[0]


def _decays(values: list[float]) -> bool:
    return len(values) >= 2 and values[-1] < values[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data
