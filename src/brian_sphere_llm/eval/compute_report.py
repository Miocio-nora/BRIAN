from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

DEFAULT_TFLOPS_PER_GPU = 989.0
DEFAULT_UTILIZATION = 0.35


def estimate_flops(params: int, tokens: int, gamma: float = 1.0) -> float:
    return 6.0 * float(params) * float(tokens) * float(gamma)


def estimate_gpu_hours(
    params: int,
    tokens: int,
    tflops_per_gpu: float = DEFAULT_TFLOPS_PER_GPU,
    utilization: float = DEFAULT_UTILIZATION,
    gamma: float = 1.0,
) -> float:
    flops = estimate_flops(params, tokens, gamma)
    return flops / (float(tflops_per_gpu) * 1e12 * float(utilization)) / 3600.0


def make_compute_report(
    run_dirs: list[str | Path],
    *,
    baseline_run: str | Path | None = None,
    output_path: str | Path | None = None,
    tflops_per_gpu: float = DEFAULT_TFLOPS_PER_GPU,
    utilization: float = DEFAULT_UTILIZATION,
) -> Path:
    baseline_summary = summarize_run(baseline_run, tflops_per_gpu=tflops_per_gpu, utilization=utilization) if baseline_run else None
    summaries = [
        summarize_run(run_dir, baseline=baseline_summary, tflops_per_gpu=tflops_per_gpu, utilization=utilization)
        for run_dir in run_dirs
    ]
    report = {
        "run_count": len(summaries),
        "baseline_run": str(baseline_run) if baseline_run else None,
        "tflops_per_gpu": tflops_per_gpu,
        "utilization": utilization,
        "estimation_note": "FLOPs are rough block-equivalent estimates from logged active routing behavior, not profiler measurements.",
        "runs": summaries,
    }
    if output_path is None:
        output_path = Path("reports") / "compute_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def summarize_run(
    run_dir: str | Path,
    *,
    baseline: dict[str, Any] | None = None,
    tflops_per_gpu: float = DEFAULT_TFLOPS_PER_GPU,
    utilization: float = DEFAULT_UTILIZATION,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not (run_dir / "routing_report.json").exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    config = _read_yaml_if_exists(run_dir / "config_resolved.yaml")
    model_stats = _read_json_if_exists(run_dir / "model_stats.json")
    routing_report = _read_json_if_exists(run_dir / "routing_report.json")
    train_rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    routing_summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    final_train = train_rows[-1] if train_rows else {}
    final_eval = eval_rows[-1] if eval_rows else {}
    routing_config = config.get("routing", {}) if isinstance(config.get("routing"), dict) else {}
    model_config = config.get("model_config_resolved", {})
    model_config = model_config if isinstance(model_config, dict) else {}

    parameter_count = int(_num(model_stats.get("parameter_count")) or 0)
    physical_layers = _physical_layer_count(model_stats, config)
    active_layer_evals = _active_layer_evals(model_stats, config, routing_summary)
    active_layer_ratio = active_layer_evals / physical_layers if physical_layers > 0 else None
    trained_tokens = _trained_tokens(config, final_train, final_eval)
    estimated_flops_per_token = 6.0 * parameter_count * (active_layer_ratio if active_layer_ratio is not None else 1.0)
    estimated_training_flops = estimated_flops_per_token * trained_tokens
    estimated_gpu_hours = (
        estimated_training_flops / (float(tflops_per_gpu) * 1e12 * float(utilization)) / 3600.0
        if tflops_per_gpu > 0 and utilization > 0
        else None
    )
    summary = {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", _stage_from_name(run_dir.name))),
        "model_name": model_stats.get("model_name", ""),
        "route_mode": routing_config.get("mode"),
        "global_kv_enabled": _model_bool_value(model_config, "global_kv", default=False),
        "parallel_passing_enabled": _model_bool_value(model_config, "parallel_passing", default=False),
        "hard_exit_enabled": _hard_exit_enabled(config),
        "distributed_world_size": _world_size(config),
        "micro_batch_size": _batch_size(config),
        "gradient_accumulation_steps": _gradient_accumulation_steps(config),
        "local_effective_batch_size": _batch_size(config) * _gradient_accumulation_steps(config),
        "effective_batch_size": _batch_size(config) * _gradient_accumulation_steps(config) * _world_size(config),
        "parameter_count": parameter_count,
        "physical_layer_count": physical_layers,
        "active_layer_evals_per_token": active_layer_evals,
        "active_layer_ratio": active_layer_ratio,
        "trained_tokens_estimate": trained_tokens,
        "estimated_flops_per_token": estimated_flops_per_token,
        "estimated_training_flops": estimated_training_flops,
        "estimated_gpu_hours": estimated_gpu_hours,
        "validation_loss": _num(final_eval.get("validation_loss")),
        "perplexity": _num(final_eval.get("perplexity")),
        "train_loss": _num(final_train.get("loss")),
        "tokens_per_second_latest": _num(final_train.get("tokens_per_second")),
        "tokens_per_second_mean": _mean([_num(row.get("tokens_per_second")) for row in train_rows]),
        "train_step_time_seconds_latest": _num(final_train.get("train_step_time_seconds")),
        "train_step_time_seconds_mean": _mean([_num(row.get("train_step_time_seconds")) for row in train_rows]),
        "train_latency_ms_per_token_latest": _num(final_train.get("train_latency_ms_per_token")),
        "train_latency_ms_per_token_mean": _mean([_num(row.get("train_latency_ms_per_token")) for row in train_rows]),
        "inference_time_seconds_latest": _num(final_eval.get("inference_time_seconds")),
        "inference_tokens_per_second_latest": _num(final_eval.get("inference_tokens_per_second")),
        "inference_latency_ms_per_token_latest": _num(final_eval.get("inference_latency_ms_per_token")),
        "train_cuda_memory_allocated_mb_latest": _num(final_train.get("cuda_memory_allocated_mb")),
        "train_cuda_max_memory_allocated_mb_latest": _num(final_train.get("cuda_max_memory_allocated_mb")),
        "eval_cuda_memory_allocated_mb_latest": _num(final_eval.get("cuda_memory_allocated_mb")),
        "eval_cuda_max_memory_allocated_mb_latest": _num(final_eval.get("cuda_max_memory_allocated_mb")),
        "routing": {
            "average_route_steps": _num(routing_summary.get("average_route_steps")),
            "active_internal_decision_fraction": _num(routing_summary.get("active_block_evals_per_token")),
            "weighted_fusion_ratio": _num(routing_summary.get("weighted_fusion_ratio")),
            "parallel_branch_count_mean": _num(routing_summary.get("parallel_branch_count_mean")),
            "parallel_score_margin_mean": _num(routing_summary.get("parallel_score_margin_mean")),
        },
    }
    if baseline:
        summary["baseline_comparison"] = compare_to_baseline(summary, baseline)
    return summary


def compare_to_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    active_ratio = _ratio(summary.get("active_layer_evals_per_token"), baseline.get("active_layer_evals_per_token"))
    flops_ratio = _ratio(summary.get("estimated_flops_per_token"), baseline.get("estimated_flops_per_token"))
    param_ratio = _ratio(summary.get("parameter_count"), baseline.get("parameter_count"))
    loss_ratio = _ratio(summary.get("validation_loss"), baseline.get("validation_loss"))
    tokens_per_second_ratio = _ratio(summary.get("tokens_per_second_mean"), baseline.get("tokens_per_second_mean"))
    train_latency_ratio = _ratio(summary.get("train_latency_ms_per_token_mean"), baseline.get("train_latency_ms_per_token_mean"))
    inference_latency_ratio = _ratio(
        summary.get("inference_latency_ms_per_token_latest"),
        baseline.get("inference_latency_ms_per_token_latest"),
    )
    return {
        "baseline_run": baseline["run_dir"],
        "parameter_ratio": param_ratio,
        "same_parameter_count_view": _near(param_ratio, 1.0, tolerance=0.05),
        "active_layer_eval_ratio": active_ratio,
        "same_active_compute_view": _between(active_ratio, 0.8, 1.2),
        "estimated_flops_per_token_ratio": flops_ratio,
        "similar_training_flops_view": _between(flops_ratio, 0.8, 1.2),
        "validation_loss_delta": _delta(summary.get("validation_loss"), baseline.get("validation_loss")),
        "validation_loss_ratio": loss_ratio,
        "tokens_per_second_ratio": tokens_per_second_ratio,
        "train_latency_ms_per_token_ratio": train_latency_ratio,
        "inference_latency_ms_per_token_ratio": inference_latency_ratio,
    }


def _physical_layer_count(model_stats: dict[str, Any], config: dict[str, Any]) -> int:
    if _num(model_stats.get("layers")) is not None:
        return int(model_stats["layers"])
    pre = int(_num(model_stats.get("pre_blocks")) or 0)
    route = int(_num(model_stats.get("route_pool_blocks")) or 0)
    post = int(_num(model_stats.get("post_blocks")) or 0)
    if pre + route + post > 0:
        return pre + route + post
    model_config = config.get("model_config_resolved", {})
    if isinstance(model_config, dict) and _num(model_config.get("layers")) is not None:
        return int(model_config["layers"])
    return 0


def _active_layer_evals(model_stats: dict[str, Any], config: dict[str, Any], routing_summary: dict[str, Any]) -> float:
    if _num(model_stats.get("layers")) is not None:
        return float(model_stats["layers"])
    pre = float(_num(model_stats.get("pre_blocks")) or 0.0)
    post = float(_num(model_stats.get("post_blocks")) or 0.0)
    route_pool = float(_num(model_stats.get("route_pool_blocks")) or 0.0)
    if route_pool <= 0.0:
        return float(_physical_layer_count(model_stats, config))
    average_steps = _num(routing_summary.get("average_route_steps"))
    if average_steps is None:
        average_steps = route_pool
    internal_fraction = _num(routing_summary.get("active_block_evals_per_token"))
    if internal_fraction is None:
        internal_fraction = 1.0
    model_config = config.get("model_config_resolved", {})
    top_k = _num(model_stats.get("top_k"))
    if top_k is None and isinstance(model_config, dict):
        top_k = _num(model_config.get("top_k"))
    later_top_k = _num(model_stats.get("later_top_k"))
    if later_top_k is None and isinstance(model_config, dict):
        later_top_k = _num(model_config.get("later_top_k"))
    top_k = max(1.0, float(top_k or 1.0), float(later_top_k or 1.0))
    weighted_ratio = max(0.0, min(1.0, float(_num(routing_summary.get("weighted_fusion_ratio")) or 0.0)))
    parallel_branch_count = max(1.0, float(_num(routing_summary.get("parallel_branch_count_mean")) or 1.0))
    route_exec_multiplier = (1.0 + weighted_ratio * (top_k - 1.0)) * parallel_branch_count
    active_route_evals = float(average_steps) * float(internal_fraction) * route_exec_multiplier
    return pre + post + active_route_evals


def _trained_tokens(config: dict[str, Any], final_train: dict[str, Any], final_eval: dict[str, Any]) -> int:
    step = int(_num(final_train.get("step")) or _num(final_eval.get("step")) or _num(config.get("max_steps")) or 0)
    logged_tokens_per_step = _num(final_train.get("tokens_per_optimizer_step"))
    if logged_tokens_per_step is not None:
        return int(step * logged_tokens_per_step)
    batch_size = _batch_size(config)
    gradient_accumulation_steps = _gradient_accumulation_steps(config)
    sequence_length = _sequence_length(config)
    return int(step * batch_size * gradient_accumulation_steps * sequence_length * _world_size(config))


def _batch_size(config: dict[str, Any]) -> int:
    value = config.get("batch_size")
    if value is None:
        return 0
    return _int_value(value, "batch_size", minimum=1)


def _gradient_accumulation_steps(config: dict[str, Any]) -> int:
    value = config.get("gradient_accumulation_steps", 1)
    return _int_value(value, "gradient_accumulation_steps", minimum=1)


def _world_size(config: dict[str, Any]) -> int:
    distributed = config.get("distributed", {})
    if not isinstance(distributed, dict):
        return 1
    enabled = distributed.get("enabled", False)
    if enabled is not True:
        return 1
    value = distributed.get("world_size", 1)
    return _int_value(value, "distributed.world_size", minimum=1)


def _hard_exit_enabled(config: dict[str, Any]) -> bool | None:
    routing_config = config.get("routing", {})
    if isinstance(routing_config, dict) and "hard_exit" in routing_config:
        return _bool_value(routing_config["hard_exit"], "routing.hard_exit")
    stage = config.get("stage")
    if isinstance(stage, str) and stage == "stage4_output_action":
        return True
    model_config = config.get("model_config_resolved", {})
    if isinstance(model_config, dict) and "hard_exit" in model_config:
        return _bool_value(model_config["hard_exit"], "model_config_resolved.hard_exit")
    return None


def _sequence_length(config: dict[str, Any]) -> int:
    data_config = config.get("data_config_resolved", {})
    if isinstance(data_config, dict) and data_config.get("sequence_length") is not None:
        return _int_value(data_config["sequence_length"], "data_config_resolved.sequence_length", minimum=1)
    model_config = config.get("model_config_resolved", {})
    if isinstance(model_config, dict) and model_config.get("context_length") is not None:
        return _int_value(model_config["context_length"], "model_config_resolved.context_length", minimum=1)
    if isinstance(model_config, dict) and isinstance(model_config.get("base"), dict):
        base = model_config["base"]
        if base.get("context_length") is not None:
            return _int_value(base["context_length"], "model_config_resolved.base.context_length", minimum=1)
    return 0


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _model_bool_value(model_config: dict[str, Any], key: str, *, default: bool) -> bool:
    return _bool_value(model_config.get(key, default), f"model_config_resolved.{key}")


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


def _stage_from_name(name: str) -> str:
    for part in name.split("_"):
        if part.startswith("stage"):
            return part
    return ""


def _ratio(numerator: Any, denominator: Any) -> float | None:
    x = _num(numerator)
    y = _num(denominator)
    if x is None or y is None or y == 0.0:
        return None
    return x / y


def _delta(value: Any, baseline: Any) -> float | None:
    x = _num(value)
    y = _num(baseline)
    if x is None or y is None:
        return None
    return x - y


def _near(value: float | None, target: float, *, tolerance: float) -> bool | None:
    if value is None:
        return None
    return abs(value - target) <= tolerance


def _between(value: float | None, low: float, high: float) -> bool | None:
    if value is None:
        return None
    return low <= value <= high


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


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
    return load_config(path)
