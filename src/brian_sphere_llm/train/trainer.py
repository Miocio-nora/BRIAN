from __future__ import annotations

from collections.abc import Mapping
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.train.checkpoint import load_checkpoint, save_checkpoint
from brian_sphere_llm.train.stage_runner import build_model_from_config, train_mode_for_stage
from brian_sphere_llm.routing.schedule import scheduled_value
from brian_sphere_llm.utils.config import load_config, save_yaml
from brian_sphere_llm.utils.logging import JsonlLogger, write_json
from brian_sphere_llm.utils.seed import set_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _device(name: str) -> "torch.device":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for training.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _autocast_context(device: "torch.device", precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def run_name(config: dict[str, Any], model_name: str, data_name: str, context_length: int | None = None) -> str:
    if config.get("run_name") and config["run_name"] != "auto":
        return str(config["run_name"])
    date = datetime.utcnow().strftime("%Y%m%d")
    stage = config["stage"]
    seed = config.get("seed", 1)
    context = context_length or config.get("context_length") or config.get("sequence_length")
    context_part = f"ctx{context}" if context else "ctxunknown"
    return f"{date}_{model_name}_{stage}_{data_name}_{context_part}_seed{seed}"


def train_from_config(config_path: str | Path) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for training. Create the conda env from environment.yml.")

    config_path = Path(config_path)
    config = load_config(config_path)
    seed = _int_config(config, "seed", default=1, minimum=0)
    batch_size = _int_config(config, "batch_size", minimum=1)
    gradient_accumulation_steps = _int_config(config, "gradient_accumulation_steps", default=1, minimum=1)
    max_steps = _int_config(config, "max_steps", minimum=1)
    eval_interval = _int_config(config, "eval_interval", default=max_steps, minimum=1)
    save_interval = _int_config(config, "save_interval", default=max_steps, minimum=1)
    learning_rate = _float_config(config, "learning_rate", minimum=0.0)
    weight_decay = _float_config(config, "weight_decay", default=0.0, minimum=0.0)
    set_seed(seed)

    model_config_path = (config_path.parent / config["model_config"]).resolve()
    data_config_path = (config_path.parent / config["data_config"]).resolve()
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    tokenized_dir = Path(data_config["output_dir"])
    model = build_model_from_config(model_config_path)
    _set_activation_checkpointing(model, _bool_config(config, "activation_checkpointing", default=False))
    device = _device(str(config.get("device", "auto")))
    model.to(device)

    run_dir = Path(config.get("output_root", "runs")) / run_name(
        config,
        model_config["model_name"],
        data_config["recipe_name"],
        context_length=_int_config(data_config, "sequence_length", default=0, minimum=0) or 0,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml({**config, "model_config_resolved": model_config, "data_config_resolved": data_config}, run_dir / "config_resolved.yaml")
    model_stats = _model_stats(model)
    write_json(model_stats, run_dir / "model_stats.json")
    write_json(_data_manifest_ref(data_config, tokenized_dir), run_dir / "data_manifest_ref.json")

    train_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split="val",
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    start_step = 0
    best_eval_loss: float | None = None
    latest = run_dir / "checkpoint_latest"
    if _bool_config(config, "resume", default=False) and (latest / "state.pt").exists():
        payload = load_checkpoint(latest, model=model, optimizer=optimizer)
        start_step = int(payload.get("step", 0))
        best_eval_loss = payload.get("best_eval_loss")
        JsonlLogger(run_dir / "resume_events.jsonl").write(
            {
                "checkpoint": str(latest),
                "resumed_from_step": start_step,
                "target_max_steps": max_steps,
                "optimizer_state_loaded": "optimizer" in payload,
                "best_eval_loss": best_eval_loss,
            }
        )

    train_log = JsonlLogger(run_dir / "train_log.jsonl")
    eval_log = JsonlLogger(run_dir / "eval_log.jsonl")
    write_routing_report = _bool_config(config, "write_routing_report_on_checkpoint", default=True)
    stage_mode = train_mode_for_stage(config["stage"])
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize_if_cuda(device)
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        token_count = 0
        losses: list[float] = []
        loss_components: dict[str, list[float]] = {}
        schedule_values: dict[str, float] = {}
        routing_summary: dict[str, Any] = {}
        routing_numeric_values: dict[str, list[float]] = {}
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = batch.to(device)
            with _autocast_context(device, str(config.get("precision", "fp32"))):
                outputs = _forward_for_stage(model, batch, config=config, route_mode=stage_mode, global_step=step)
                loss = outputs["loss"]
                scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            token_count += int(batch.numel())
            losses.append(float(loss.detach().cpu()))
            _accumulate_loss_components(loss_components, outputs.get("loss_components", {}))
            if "schedule_values" in outputs:
                schedule_values.update(outputs["schedule_values"])
            _accumulate_routing_summary(routing_summary, routing_numeric_values, outputs.get("routing_summary", {}))
        if config.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _float_config(config, "grad_clip", minimum=0.0))
        optimizer.step()
        _synchronize_if_cuda(device)
        elapsed = max(1e-9, time.time() - started)
        row = {
            "step": step,
            "loss": _mean(losses),
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "tokens_per_optimizer_step": token_count,
            "tokens_per_second": int(token_count / elapsed),
            "train_step_time_seconds": elapsed,
            "train_latency_ms_per_token": _latency_ms_per_token(elapsed, token_count),
        }
        row.update(_cuda_memory_metrics(device))
        if loss_components:
            row.update({key: _mean(values) for key, values in loss_components.items()})
        if schedule_values:
            row.update(schedule_values)
        if routing_summary or routing_numeric_values:
            row.update(_finalize_routing_summary(routing_summary, routing_numeric_values))
        train_log.write(row)

        if step % eval_interval == 0 or step == max_steps:
            eval_row = evaluate(model, val_loader, config=config, device=device, route_mode=stage_mode, global_step=step)
            eval_row["step"] = step
            eval_log.write(eval_row)
            eval_loss = float(eval_row["validation_loss"])
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(run_dir / "checkpoint_best", model=model, optimizer=optimizer, step=step, best_eval_loss=best_eval_loss)
        if step % save_interval == 0 or step == max_steps:
            save_checkpoint(latest, model=model, optimizer=optimizer, step=step, best_eval_loss=best_eval_loss)
            if write_routing_report:
                make_routing_report(run_dir)
    if not (run_dir / "routing_report.json").exists():
        make_routing_report(run_dir)
    return run_dir


def _forward_for_stage(model: Any, batch: "torch.Tensor", *, config: dict[str, Any], route_mode: str, global_step: int) -> dict:
    if route_mode == "baseline":
        return model(batch, targets=batch)
    routing_cfg = _mapping_config(config, "routing")
    loss_weights = dict(_mapping_config(config, "loss_weights"))
    schedule_values = _schedule_values(config, route_mode=route_mode, global_step=global_step)
    router_probability = schedule_values.get("scheduled_router_probability")
    if "scheduled_lambda_route" in schedule_values:
        loss_weights["route"] = schedule_values["scheduled_lambda_route"]
    outputs = model(
        batch,
        targets=batch,
        route_mode=route_mode,
        pseudo_policy=str(routing_cfg.get("pseudo_policy", "sequential")),
        loss_weights=loss_weights,
        hard_exit=_bool_mapping_value(
            routing_cfg,
            "hard_exit",
            default=str(config.get("stage")) == "stage4_output_action",
            name="routing.hard_exit",
        ),
        router_probability=router_probability,
        global_step=global_step,
    )
    if schedule_values:
        outputs["schedule_values"] = schedule_values
    return outputs


def _set_activation_checkpointing(model: Any, enabled: bool) -> None:
    if hasattr(model, "activation_checkpointing"):
        model.activation_checkpointing = enabled


def _accumulate_loss_components(accumulator: dict[str, list[float]], components: Any) -> None:
    if not isinstance(components, Mapping):
        return
    for key, value in components.items():
        number = _metric_number(value)
        if number is not None:
            accumulator.setdefault(str(key), []).append(number)


def _accumulate_routing_summary(
    last_values: dict[str, Any],
    numeric_values: dict[str, list[float]],
    summary: Any,
) -> None:
    if not isinstance(summary, Mapping):
        return
    for key, value in summary.items():
        number = _metric_number(value)
        if number is None:
            last_values[str(key)] = value
        else:
            numeric_values.setdefault(str(key), []).append(number)


def _finalize_routing_summary(last_values: dict[str, Any], numeric_values: dict[str, list[float]]) -> dict[str, Any]:
    summary = dict(last_values)
    for key, values in numeric_values.items():
        summary[key] = _mean(values)
    return summary


def _metric_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if torch is not None and isinstance(value, torch.Tensor) and value.numel() == 1:
        number = float(value.detach().cpu())
        return number if math.isfinite(number) else None
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _schedule_values(config: dict[str, Any], *, route_mode: str, global_step: int) -> dict[str, float]:
    if route_mode != "scheduled":
        return {}
    routing_cfg = _mapping_config(config, "routing")
    schedule = routing_cfg.get("schedule", [])
    loss_weights = _mapping_config(config, "loss_weights")
    default_lambda_route = loss_weights.get("route", 0.0)
    return {
        "scheduled_router_probability": scheduled_value(schedule, global_step, "router_probability", 0.0),
        "scheduled_lambda_route": scheduled_value(schedule, global_step, "lambda_route", default_lambda_route),
    }


@torch.no_grad() if torch is not None else (lambda fn: fn)
def evaluate(
    model: Any,
    val_loader: Any,
    *,
    config: dict[str, Any],
    device: "torch.device",
    route_mode: str,
    global_step: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    summary_accumulator: dict[str, list[float]] = {}
    token_count = 0
    batch_count = 0
    max_batches = min(8, len(val_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize_if_cuda(device)
    started = time.time()
    for index, batch in enumerate(val_loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        token_count += int(batch.numel())
        batch_count += 1
        outputs = _forward_for_stage(model, batch, config=config, route_mode=route_mode, global_step=global_step)
        losses.append(float(outputs["loss"].detach().cpu()))
        for key, value in outputs.get("routing_summary", {}).items():
            if isinstance(value, (int, float)):
                summary_accumulator.setdefault(key, []).append(float(value))
    _synchronize_if_cuda(device)
    elapsed = max(1e-9, time.time() - started)
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    row: dict[str, Any] = {
        "validation_loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "eval_batch_count": batch_count,
        "eval_token_count": token_count,
        "inference_time_seconds": elapsed,
        "inference_tokens_per_second": token_count / elapsed if token_count else None,
        "inference_latency_ms_per_token": _latency_ms_per_token(elapsed, token_count),
    }
    row.update(_cuda_memory_metrics(device))
    row.update(_schedule_values(config, route_mode=route_mode, global_step=global_step))
    for key, values in summary_accumulator.items():
        row[key] = sum(values) / max(1, len(values))
    return row


def _synchronize_if_cuda(device: "torch.device") -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_ms_per_token(elapsed_seconds: float, token_count: int) -> float | None:
    if token_count <= 0:
        return None
    return elapsed_seconds * 1000.0 / token_count


def _cuda_memory_metrics(device: "torch.device") -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
    }


def _int_config(
    config: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key in config:
        value = config[key]
    elif default is not None:
        value = default
    else:
        raise KeyError(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{key} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return number


def _float_config(
    config: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    if key in config:
        value = config[key]
    elif default is not None:
        value = default
    else:
        raise KeyError(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite numeric value.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return number


def _bool_config(config: dict[str, Any], key: str, *, default: bool) -> bool:
    return _bool_value(config.get(key, default), key)


def _bool_mapping_value(mapping: Mapping[str, Any], key: str, *, default: bool, name: str) -> bool:
    return _bool_value(mapping.get(key, default), name)


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


def _mapping_config(config: dict[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _data_manifest_ref(data_config: dict[str, Any], tokenized_dir: Path) -> dict[str, Any]:
    manifest_value = data_config.get("manifest_path")
    manifest_path = Path(str(manifest_value)) if manifest_value else tokenized_dir / "manifest.jsonl"
    stats_path = tokenized_dir / "stats.json"
    ref: dict[str, Any] = {
        "recipe_name": data_config.get("recipe_name"),
        "path": str(manifest_path),
        "tokenized_dir": str(tokenized_dir),
        "stats_path": str(stats_path),
        "sequence_length": data_config.get("sequence_length"),
    }
    if not stats_path.exists():
        return ref
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    for key in [
        "num_documents",
        "num_tokens_train",
        "num_tokens_val",
        "avg_tokens_per_doc",
        "vocab_size",
        "source_mixture_expected",
        "source_mixture_realized",
        "source_mixture_realized_share",
        "sha256_manifest",
    ]:
        if key in stats:
            ref[key] = stats[key]
    tokenizer = stats.get("tokenizer")
    if isinstance(tokenizer, dict):
        ref["tokenizer"] = {
            key: tokenizer.get(key)
            for key in ["name", "revision", "license", "vocab_size", "special_tokens"]
            if key in tokenizer
        }
    return ref


def _model_stats(model: Any) -> dict[str, Any]:
    if not hasattr(model, "model_stats"):
        raise ValueError("Model must expose model_stats() so parameter counts are recorded.")
    stats = model.model_stats()
    if not isinstance(stats, dict):
        raise ValueError("model_stats() must return a mapping.")
    if "parameter_count" not in stats:
        raise ValueError("model_stats() must include parameter_count.")
    parameter_count = stats["parameter_count"]
    if type(parameter_count) is not int or parameter_count <= 0:
        raise ValueError("model_stats().parameter_count must be a positive integer.")
    return stats
