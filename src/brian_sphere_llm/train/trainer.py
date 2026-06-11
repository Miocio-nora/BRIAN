from __future__ import annotations

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


def run_name(config: dict[str, Any], model_name: str, data_name: str) -> str:
    if config.get("run_name") and config["run_name"] != "auto":
        return str(config["run_name"])
    date = datetime.utcnow().strftime("%Y%m%d")
    stage = config["stage"]
    seed = config.get("seed", 1)
    return f"{date}_{model_name}_{stage}_{data_name}_seed{seed}"


def train_from_config(config_path: str | Path) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for training. Create the conda env from environment.yml.")

    config_path = Path(config_path)
    config = load_config(config_path)
    set_seed(int(config.get("seed", 1)))

    model_config_path = (config_path.parent / config["model_config"]).resolve()
    data_config_path = (config_path.parent / config["data_config"]).resolve()
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    tokenized_dir = Path(data_config["output_dir"])
    model = build_model_from_config(model_config_path)
    device = _device(str(config.get("device", "auto")))
    model.to(device)

    run_dir = Path(config.get("output_root", "runs")) / run_name(config, model_config["model_name"], data_config["recipe_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml({**config, "model_config_resolved": model_config, "data_config_resolved": data_config}, run_dir / "config_resolved.yaml")
    if hasattr(model, "model_stats"):
        write_json(model.model_stats(), run_dir / "model_stats.json")
    write_json({"path": str(data_config.get("manifest_path", ""))}, run_dir / "data_manifest_ref.json")

    train_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split="train",
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split="val",
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    start_step = 0
    best_eval_loss: float | None = None
    latest = run_dir / "checkpoint_latest"
    if config.get("resume") and (latest / "state.pt").exists():
        payload = load_checkpoint(latest, model=model, optimizer=optimizer)
        start_step = int(payload.get("step", 0))
        best_eval_loss = payload.get("best_eval_loss")
        JsonlLogger(run_dir / "resume_events.jsonl").write(
            {
                "checkpoint": str(latest),
                "resumed_from_step": start_step,
                "target_max_steps": int(config["max_steps"]),
                "optimizer_state_loaded": "optimizer" in payload,
                "best_eval_loss": best_eval_loss,
            }
        )

    train_log = JsonlLogger(run_dir / "train_log.jsonl")
    eval_log = JsonlLogger(run_dir / "eval_log.jsonl")
    max_steps = int(config["max_steps"])
    eval_interval = int(config.get("eval_interval", max_steps))
    save_interval = int(config.get("save_interval", max_steps))
    write_routing_report = bool(config.get("write_routing_report_on_checkpoint", True))
    stage_mode = train_mode_for_stage(config["stage"])
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize_if_cuda(device)
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, str(config.get("precision", "fp32"))):
            outputs = _forward_for_stage(model, batch, config=config, route_mode=stage_mode, global_step=step)
            loss = outputs["loss"]
        loss.backward()
        if config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        _synchronize_if_cuda(device)
        elapsed = max(1e-9, time.time() - started)
        token_count = int(batch.numel())
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "tokens_per_second": int(token_count / elapsed),
            "train_step_time_seconds": elapsed,
            "train_latency_ms_per_token": _latency_ms_per_token(elapsed, token_count),
        }
        row.update(_cuda_memory_metrics(device))
        if "loss_components" in outputs:
            row.update({key: float(value.cpu()) for key, value in outputs["loss_components"].items()})
        if "schedule_values" in outputs:
            row.update(outputs["schedule_values"])
        if "routing_summary" in outputs:
            row.update(outputs["routing_summary"])
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
    return run_dir


def _forward_for_stage(model: Any, batch: "torch.Tensor", *, config: dict[str, Any], route_mode: str, global_step: int) -> dict:
    if route_mode == "baseline":
        return model(batch, targets=batch)
    routing_cfg = config.get("routing", {})
    loss_weights = dict(config.get("loss_weights", {}))
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
        hard_exit=bool(routing_cfg.get("hard_exit", config.get("stage") == "stage4_output_action")),
        router_probability=router_probability,
        global_step=global_step,
    )
    if schedule_values:
        outputs["schedule_values"] = schedule_values
    return outputs


def _schedule_values(config: dict[str, Any], *, route_mode: str, global_step: int) -> dict[str, float]:
    if route_mode != "scheduled":
        return {}
    routing_cfg = config.get("routing", {})
    schedule = routing_cfg.get("schedule", []) if isinstance(routing_cfg, dict) else []
    loss_weights = config.get("loss_weights", {})
    default_lambda_route = float(loss_weights.get("route", 0.0)) if isinstance(loss_weights, dict) else 0.0
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
