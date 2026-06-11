from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.difficulty_report import _checkpoint_step, _load_model_for_run
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.train.trainer import evaluate
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json
from brian_sphere_llm.utils.seed import set_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


NONDETERMINISTIC_NUMERIC_KEYS = {
    "created_at",
    "inference_time_seconds",
    "inference_tokens_per_second",
    "inference_latency_ms_per_token",
    "cuda_memory_allocated_mb",
    "cuda_max_memory_allocated_mb",
}


def make_eval_determinism_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    split: str = "val",
    batch_size: int | None = None,
    checkpoint: str = "checkpoint_best",
    seed: int = 1,
    device_name: str = "auto",
    tolerance: float = 1e-8,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for eval determinism reports.")
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    data_config = config.get("data_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    route_mode = train_mode_for_stage(str(config["stage"]))
    global_step = _checkpoint_step(run_dir, checkpoint)
    effective_batch_size = _effective_batch_size(batch_size, config)
    effective_tolerance = _float_value(tolerance, "tolerance", minimum=0.0)
    effective_seed = _int_value(seed, "seed", minimum=0)
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=effective_batch_size,
        shuffle=False,
    )

    first = _eval_once(
        model,
        loader,
        config=config,
        device=device,
        route_mode=route_mode,
        global_step=global_step,
        seed=effective_seed,
    )
    second = _eval_once(
        model,
        loader,
        config=config,
        device=device,
        route_mode=route_mode,
        global_step=global_step,
        seed=effective_seed,
    )
    comparison = _compare_numeric_metrics(first, second, tolerance=effective_tolerance)
    checks = {
        "checkpoint_loaded": True,
        "two_eval_passes_completed": True,
        "compared_numeric_metrics_present": comparison["compared_metric_count"] > 0,
        "numeric_metrics_within_tolerance": comparison["max_abs_delta"] is not None
        and comparison["max_abs_delta"] <= effective_tolerance
        and not comparison["mismatched_metrics"],
    }
    report = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "checkpoint_step": global_step,
        "split": split,
        "batch_size": effective_batch_size,
        "seed": effective_seed,
        "tolerance": effective_tolerance,
        "first_eval": first,
        "second_eval": second,
        "comparison": comparison,
        "checks": checks,
        "overall_status": "pass" if all(checks.values()) else "fail",
    }
    if output_path is None:
        output_path = run_dir / "eval_determinism_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _eval_once(
    model: Any,
    loader: Any,
    *,
    config: dict[str, Any],
    device: "torch.device",
    route_mode: str,
    global_step: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    return evaluate(model, loader, config=config, device=device, route_mode=route_mode, global_step=global_step)


def _compare_numeric_metrics(first: dict[str, Any], second: dict[str, Any], *, tolerance: float) -> dict[str, Any]:
    keys = sorted((set(first) & set(second)) - NONDETERMINISTIC_NUMERIC_KEYS)
    rows = []
    max_abs_delta: float | None = None
    for key in keys:
        left = _num(first.get(key))
        right = _num(second.get(key))
        if left is None or right is None:
            continue
        delta = abs(left - right)
        max_abs_delta = delta if max_abs_delta is None else max(max_abs_delta, delta)
        rows.append(
            {
                "metric": key,
                "first": left,
                "second": right,
                "abs_delta": delta,
                "within_tolerance": delta <= tolerance,
            }
        )
    mismatches = [row for row in rows if not row["within_tolerance"]]
    return {
        "compared_metric_count": len(rows),
        "max_abs_delta": max_abs_delta,
        "mismatched_metrics": mismatches,
        "metrics": rows,
    }


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _effective_batch_size(batch_size: int | None, config: dict[str, Any]) -> int:
    value = batch_size if batch_size is not None else config.get("batch_size", 1)
    return _int_value(value, "batch_size", minimum=1)


def _int_value(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def _float_value(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def _device(name: str) -> "torch.device":
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
