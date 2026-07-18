from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.data.manifest import sha256_text
from brian_sphere_llm.eval.bdre_cache_visualization import make_bdre_cache_visualization_from_payload
from brian_sphere_llm.eval.post_train_benchmarks import run_checkpoint_benchmarks
from brian_sphere_llm.eval.router_space_visualization import make_router_space_visualization_from_payload
from brian_sphere_llm.eval.route_path_visualization import make_route_path_visualization_from_train_log
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.train.checkpoint import (
    load_checkpoint,
    load_rank_state,
    rank_state_path,
    save_checkpoint,
    save_rank_state,
)
from brian_sphere_llm.train.live_telemetry import (
    LiveTelemetryWriter,
    TerminalDashboardConfig,
    cuda_device_metadata,
    extract_route_trace,
    internal_position_snapshot,
)
from brian_sphere_llm.train.stage_runner import build_model_from_config, train_mode_for_stage
from brian_sphere_llm.routing.schedule import scheduled_value
from brian_sphere_llm.utils.config import load_config, save_yaml
from brian_sphere_llm.utils import distributed as dist_utils
from brian_sphere_llm.utils.logging import JsonlLogger, write_json
from brian_sphere_llm.utils.seed import set_seed

try:
    import torch
    from torch.nn.parallel import DistributedDataParallel
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DistributedDataParallel = None


def _device(name: str) -> "torch.device":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for training.")
    if name == "auto":
        if dist_utils.is_distributed() and torch.cuda.is_available():
            return torch.device("cuda", dist_utils.local_rank())
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if dist_utils.is_distributed() and device.type == "cuda" and device.index is None:
        return torch.device("cuda", dist_utils.local_rank())
    return device


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
    save_best_checkpoint = _bool_config(config, "save_best_checkpoint", default=True)
    learning_rate = _float_config(config, "learning_rate", minimum=0.0)
    lr_schedule = _lr_schedule_config(config)
    warmup_steps = _int_config(config, "warmup_steps", default=0, minimum=0)
    min_learning_rate = _float_config(config, "min_learning_rate", default=0.0, minimum=0.0)
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate.")
    weight_decay = _float_config(config, "weight_decay", default=0.0, minimum=0.0)
    terminal_dashboard_config = TerminalDashboardConfig.from_train_config(config)
    stateful_tbptt = _stateful_tbptt_config(config)
    stage_mode = train_mode_for_stage(config["stage"])
    ddp_find_unused_parameters = _bool_config(
        config,
        "ddp_find_unused_parameters",
        default=stage_mode != "baseline",
    )
    ddp_static_graph = _bool_config(config, "ddp_static_graph", default=False)
    ddp_gradient_as_bucket_view = _bool_config(config, "ddp_gradient_as_bucket_view", default=False)
    ddp_broadcast_buffers = _bool_config(
        config,
        "ddp_broadcast_buffers",
        default=not bool(stateful_tbptt["enabled"]),
    )
    distributed_timeout_seconds = (
        _int_config(config, "distributed_timeout_seconds", minimum=1)
        if "distributed_timeout_seconds" in config
        else None
    )
    set_seed(seed)
    _set_float32_matmul_precision(config)

    model_config_path = (config_path.parent / config["model_config"]).resolve()
    data_config_path = (config_path.parent / config["data_config"]).resolve()
    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)
    tokenized_dir = Path(data_config["output_dir"])
    model = build_model_from_config(model_config_path)
    _set_activation_checkpointing(model, _bool_config(config, "activation_checkpointing", default=False))
    _validate_stateful_chunk_semantics(model, stateful_tbptt)
    device = _device(str(config.get("device", "auto")))
    distributed = dist_utils.init_distributed(device, timeout_seconds=distributed_timeout_seconds)
    try:
        distributed_batch_contract = _distributed_batch_contract(
            config,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=dist_utils.world_size() if distributed else 1,
        )
    except Exception:
        dist_utils.destroy_distributed()
        raise
    is_main_process = dist_utils.is_main_process()
    if stateful_tbptt["enabled"]:
        if not hasattr(model, "forward_stream_chunk") or not hasattr(
            model,
            "prepare_stream_route_targets",
        ):
            raise ValueError("stateful_tbptt requires a BDRE model with streaming chunk support.")
    model.to(device)

    run_dir = Path(config.get("output_root", "runs")) / run_name(
        config,
        model_config["model_name"],
        data_config["recipe_name"],
        context_length=_int_config(data_config, "sequence_length", default=0, minimum=0) or 0,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "model_config_resolved": model_config,
        "data_config_resolved": data_config,
        "distributed": {
            "enabled": distributed,
            "world_size": dist_utils.world_size(),
            "rank": dist_utils.rank(),
            "local_rank": dist_utils.local_rank(),
            "find_unused_parameters": ddp_find_unused_parameters,
            "static_graph": ddp_static_graph,
            "gradient_as_bucket_view": ddp_gradient_as_bucket_view,
            "broadcast_buffers": ddp_broadcast_buffers,
            "timeout_seconds": distributed_timeout_seconds,
            "stateful_manual_gradient_sync": bool(distributed and stateful_tbptt["enabled"]),
            "stateful_gradient_sync_bucket_mb": int(stateful_tbptt["gradient_sync_bucket_mb"]),
            "batch_contract": distributed_batch_contract,
        },
    }
    if is_main_process:
        save_yaml(resolved_config, run_dir / "config_resolved.yaml")
    model_stats = _model_stats(model)
    if is_main_process:
        write_json(model_stats, run_dir / "model_stats.json")
        write_json(_data_manifest_ref(data_config, tokenized_dir), run_dir / "data_manifest_ref.json")
    wandb_run = _init_wandb(
        config,
        resolved_config=resolved_config,
        model_stats=model_stats,
        run_dir=run_dir,
        is_main_process=is_main_process,
    )
    dist_utils.barrier()

    train_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        distributed=distributed,
        rank=dist_utils.rank(),
        world_size=dist_utils.world_size(),
        seed=seed,
    )
    val_loader = build_dataloader(
        tokenized_dir=tokenized_dir,
        split=str(config.get("eval_split", "val")),
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
    data_epoch = 0
    microbatch_in_epoch = 0
    latest = run_dir / "checkpoint_latest"
    if _bool_config(config, "resume", default=False) and (latest / "state.pt").exists():
        payload = load_checkpoint(latest, model=model, optimizer=optimizer, restore_rng_state=True)
        start_step = int(payload.get("step", 0))
        best_eval_loss = payload.get("best_eval_loss")
        data_epoch = _payload_int(payload, "data_epoch", default=0, minimum=0)
        microbatch_in_epoch = _payload_int(payload, "microbatch_in_epoch", default=0, minimum=0)
        rank_payload, rank_state_loaded = _load_rank_training_state(
            latest,
            rank=dist_utils.rank(),
            distributed=distributed,
        )
        if rank_payload:
            data_epoch = _payload_int(rank_payload, "data_epoch", default=data_epoch, minimum=0)
            microbatch_in_epoch = _payload_int(
                rank_payload,
                "microbatch_in_epoch",
                default=microbatch_in_epoch,
                minimum=0,
            )
        loaded_rank_state_path = rank_state_path(latest, rank=dist_utils.rank()) if rank_state_loaded else None
        if is_main_process:
            JsonlLogger(run_dir / "resume_events.jsonl").write(
                {
                    "checkpoint": str(latest),
                    "resumed_from_step": start_step,
                    "target_max_steps": max_steps,
                    "optimizer_state_loaded": "optimizer" in payload,
                    "rng_state_loaded": "rng_state" in payload,
                    "rank_state_loaded": rank_state_loaded,
                    "rank_state_path": str(loaded_rank_state_path) if loaded_rank_state_path is not None else None,
                    "data_epoch": data_epoch,
                    "microbatch_in_epoch": microbatch_in_epoch,
                    "best_eval_loss": best_eval_loss,
                }
            )
    model = _wrap_distributed_model(
        model,
        device,
        distributed=distributed,
        find_unused_parameters=ddp_find_unused_parameters,
        static_graph=ddp_static_graph,
        gradient_as_bucket_view=ddp_gradient_as_bucket_view,
        broadcast_buffers=ddp_broadcast_buffers,
    )

    train_log = JsonlLogger(run_dir / "train_log.jsonl") if is_main_process else None
    eval_log = JsonlLogger(run_dir / "eval_log.jsonl") if is_main_process else None
    terminal_dashboard_writer: LiveTelemetryWriter | None = None
    if is_main_process and terminal_dashboard_config.enabled:
        device_name, device_memory_mb = cuda_device_metadata(device)
        terminal_dashboard_writer = LiveTelemetryWriter(
            run_dir,
            terminal_dashboard_config,
            run_name=run_dir.name,
            max_steps=max_steps,
            start_step=start_step,
            num_blocks=int(model_config.get("route_pool_blocks", 0)),
            model_name=str(model_config["model_name"]),
            world_size=dist_utils.world_size(),
            device_name=device_name,
            device_memory_mb=device_memory_mb,
        )
    write_routing_report = _bool_config(config, "write_routing_report_on_checkpoint", default=True)
    route_path_visualization = _route_path_visualization_config(config, default_interval=save_interval)
    router_space_visualization = _router_space_visualization_config(config, default_interval=save_interval)
    bdre_cache_visualization = _bdre_cache_visualization_config(config, default_interval=save_interval)
    checkpoint_retention = _checkpoint_retention_config(config, default_interval=save_interval)
    checkpoint_benchmarks = _checkpoint_benchmark_config(config, default_interval=save_interval)
    benchmark_log = JsonlLogger(run_dir / "benchmark_log.jsonl") if is_main_process and checkpoint_benchmarks["enabled"] else None
    stateful_manual_gradient_sync = bool(distributed and stateful_tbptt["enabled"])
    ddp_no_sync_microbatches = (
        gradient_accumulation_steps
        if stateful_manual_gradient_sync
        else _ddp_no_sync_microbatch_count(
            model,
            distributed=distributed,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    )
    iterator, data_epoch, microbatch_in_epoch = _restore_dataloader_position(
        train_loader,
        data_epoch=data_epoch,
        microbatch_in_epoch=microbatch_in_epoch,
    )
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize_if_cuda(device)
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        current_learning_rate = _learning_rate_for_step(
            step,
            base_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            schedule=lr_schedule,
        )
        _set_optimizer_lr(optimizer, current_learning_rate)
        token_count = 0
        losses: list[float] = []
        loss_components: dict[str, list[float]] = {}
        schedule_values: dict[str, float] = {}
        routing_summary: dict[str, Any] = {}
        routing_numeric_values: dict[str, list[float]] = {}
        router_space_payload: dict[str, Any] | None = None
        bdre_cache_payload: dict[str, Any] | None = None
        stateful_gradient_sync_stats = _empty_stateful_gradient_sync_stats()
        summarize_routing = _routing_summary_due(config, global_step=step) or _visualization_due(
            route_path_visualization,
            step=step,
            max_steps=max_steps,
        )
        collect_router_space = is_main_process and _visualization_due(
            router_space_visualization,
            step=step,
            max_steps=max_steps,
        )
        collect_bdre_visualization = is_main_process and _visualization_due(
            bdre_cache_visualization,
            step=step,
            max_steps=max_steps,
        )
        for micro_step in range(gradient_accumulation_steps):
            batch, iterator, data_epoch, microbatch_in_epoch = _next_train_batch(
                train_loader,
                iterator,
                data_epoch=data_epoch,
                microbatch_in_epoch=microbatch_in_epoch,
            )
            batch = batch.to(device)
            final_accumulation_microbatch = micro_step == gradient_accumulation_steps - 1
            should_sync_gradients = final_accumulation_microbatch and not stateful_manual_gradient_sync
            with _gradient_sync_context(model, distributed=distributed, should_sync=should_sync_gradients):
                if stateful_tbptt["enabled"]:
                    outputs = _backward_stateful_tbptt_microbatch(
                        model,
                        batch,
                        config=config,
                        route_mode=stage_mode,
                        global_step=step,
                        chunk_size=int(stateful_tbptt["chunk_size"]),
                        detach_interval_chunks=int(stateful_tbptt["detach_interval_chunks"]),
                        gradient_scale=1.0 / gradient_accumulation_steps,
                        device=device,
                        collect_router_space=collect_router_space and final_accumulation_microbatch,
                        collect_bdre_visualization=collect_bdre_visualization and final_accumulation_microbatch,
                        summarize_routing=summarize_routing,
                    )
                    loss = outputs["loss"]
                else:
                    with _autocast_context(device, str(config.get("precision", "fp32"))):
                        outputs = _forward_for_stage(
                            model,
                            batch,
                            config=config,
                            route_mode=stage_mode,
                            global_step=step,
                            collect_router_space=collect_router_space and final_accumulation_microbatch,
                            collect_bdre_visualization=collect_bdre_visualization and final_accumulation_microbatch,
                            summarize_routing=summarize_routing,
                        )
                        loss = outputs["loss"]
                        scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()
            token_count += int(batch.numel())
            losses.append(float(loss.detach().cpu()))
            _accumulate_loss_components(loss_components, outputs.get("loss_components", {}))
            if "schedule_values" in outputs:
                schedule_values.update(outputs["schedule_values"])
            _accumulate_routing_summary(routing_summary, routing_numeric_values, outputs.get("routing_summary", {}))
            if collect_router_space and final_accumulation_microbatch and "router_space" in outputs:
                router_space_payload = outputs["router_space"]
            if collect_bdre_visualization and final_accumulation_microbatch and "bdre_visualization" in outputs:
                bdre_cache_payload = outputs["bdre_visualization"]
        if stateful_manual_gradient_sync:
            stateful_gradient_sync_stats = _sync_stateful_ddp_gradients(
                model,
                bucket_cap_mb=int(stateful_tbptt["gradient_sync_bucket_mb"]),
            )
        if config.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _float_config(config, "grad_clip", minimum=0.0))
        optimizer.step()
        _synchronize_if_cuda(device)
        elapsed = max(1e-9, time.time() - started)
        distributed_world_size = dist_utils.world_size() if distributed else 1
        local_token_count = token_count
        global_token_count = _global_train_token_count(local_token_count, distributed=distributed)
        row = {
            "step": step,
            "loss": _distributed_mean_scalar(_mean(losses), device=device, distributed=distributed),
            "learning_rate": current_learning_rate,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "local_effective_batch_size": batch_size * gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps * distributed_world_size,
            "local_tokens_per_optimizer_step": local_token_count,
            "tokens_per_optimizer_step": global_token_count,
            "distributed_world_size": distributed_world_size,
            "ddp_find_unused_parameters": ddp_find_unused_parameters,
            "ddp_no_sync_microbatches": ddp_no_sync_microbatches,
            "stateful_tbptt_enabled": bool(stateful_tbptt["enabled"]),
            "stateful_tbptt_chunk_size": (
                int(stateful_tbptt["chunk_size"]) if stateful_tbptt["enabled"] else 0
            ),
            "stateful_tbptt_chunks": int(outputs.get("stateful_tbptt_chunks", 0)),
            "stateful_tbptt_detach_interval_chunks": (
                int(stateful_tbptt["detach_interval_chunks"])
                if stateful_tbptt["enabled"]
                else 0
            ),
            "stateful_tbptt_backward_groups": int(
                outputs.get("stateful_tbptt_backward_groups", 0)
            ),
            "stateful_tbptt_gradient_horizon_tokens": int(
                outputs.get("stateful_tbptt_gradient_horizon_tokens", 0)
            ),
            "stateful_ddp_manual_gradient_sync": stateful_manual_gradient_sync,
            **stateful_gradient_sync_stats,
            "local_tokens_per_second": int(local_token_count / elapsed),
            "tokens_per_second": int(global_token_count / elapsed),
            "train_step_time_seconds": elapsed,
            "local_train_latency_ms_per_token": _latency_ms_per_token(elapsed, local_token_count),
            "train_latency_ms_per_token": _latency_ms_per_token(elapsed, global_token_count),
        }
        row.update(_cuda_memory_metrics(device))
        if loss_components:
            row.update(
                _distributed_mean_metrics(
                    {key: _mean(values) for key, values in loss_components.items()},
                    device=device,
                    distributed=distributed,
                )
            )
        if schedule_values:
            row.update(schedule_values)
        if routing_summary or routing_numeric_values:
            row.update(
                _distributed_mean_metrics(
                    _finalize_routing_summary(routing_summary, routing_numeric_values),
                    device=device,
                    distributed=distributed,
                )
            )
        if train_log is not None:
            train_log.write(row)
        _wandb_log(wandb_run, "train", row)
        if terminal_dashboard_writer is not None and terminal_dashboard_config.train_due(step, max_steps):
            unwrapped_model = dist_utils.unwrap_model(model)
            num_internal_blocks = int(getattr(unwrapped_model.config, "route_pool_blocks", 0))
            route_trace = extract_route_trace(
                outputs,
                num_internal_blocks=num_internal_blocks,
                sample_index=terminal_dashboard_config.sample_index,
            )
            positions = (
                internal_position_snapshot(unwrapped_model, num_internal_blocks=num_internal_blocks)
                if terminal_dashboard_config.positions_due(step, max_steps)
                else None
            )
            terminal_dashboard_writer.write_train(
                row,
                step=step,
                max_steps=max_steps,
                route=route_trace,
                token_index=max(0, int(batch.shape[1]) - 1),
                positions=positions,
            )
        if is_main_process:
            _maybe_log_route_path_visualization(
                wandb_run,
                run_dir=run_dir,
                model=dist_utils.unwrap_model(model),
                step=step,
                max_steps=max_steps,
                config=route_path_visualization,
            )
            _maybe_log_router_space_visualization(
                wandb_run,
                run_dir=run_dir,
                model=dist_utils.unwrap_model(model),
                payload=router_space_payload,
                step=step,
                max_steps=max_steps,
                config=router_space_visualization,
            )
            _maybe_log_bdre_cache_visualization(
                wandb_run,
                run_dir=run_dir,
                model=dist_utils.unwrap_model(model),
                payload=bdre_cache_payload,
                step=step,
                max_steps=max_steps,
                config=bdre_cache_visualization,
            )

        if step % eval_interval == 0 or step == max_steps:
            eval_row = evaluate(model, val_loader, config=config, device=device, route_mode=stage_mode, global_step=step)
            eval_row["step"] = step
            if eval_log is not None:
                eval_log.write(eval_row)
            _wandb_log(wandb_run, "eval", eval_row)
            if terminal_dashboard_writer is not None:
                terminal_dashboard_writer.write_eval(eval_row, step=step)
            eval_loss = float(eval_row["validation_loss"])
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if save_best_checkpoint and is_main_process:
                    save_checkpoint(
                        run_dir / "checkpoint_best",
                        model=dist_utils.unwrap_model(model),
                        optimizer=optimizer,
                        step=step,
                        best_eval_loss=best_eval_loss,
                        extra=_checkpoint_training_state(data_epoch, microbatch_in_epoch),
                    )
                if save_best_checkpoint:
                    _save_rank_training_state(
                        run_dir / "checkpoint_best",
                        step=step,
                        best_eval_loss=best_eval_loss,
                        data_epoch=data_epoch,
                        microbatch_in_epoch=microbatch_in_epoch,
                    )
        if step % save_interval == 0 or step == max_steps:
            if is_main_process:
                save_checkpoint(
                    latest,
                    model=dist_utils.unwrap_model(model),
                    optimizer=optimizer,
                    step=step,
                    best_eval_loss=best_eval_loss,
                    extra=_checkpoint_training_state(data_epoch, microbatch_in_epoch),
                )
            _save_rank_training_state(
                latest,
                step=step,
                best_eval_loss=best_eval_loss,
                data_epoch=data_epoch,
                microbatch_in_epoch=microbatch_in_epoch,
            )
            if is_main_process and write_routing_report:
                make_routing_report(run_dir)
        retained_checkpoint_name: str | None = None
        if _retained_checkpoint_due(
            checkpoint_retention,
            checkpoint_benchmarks,
            step=step,
            max_steps=max_steps,
        ):
            retained_checkpoint_name = f"{checkpoint_retention['prefix']}_{step:08d}"
            if is_main_process:
                save_checkpoint(
                    run_dir / retained_checkpoint_name,
                    model=dist_utils.unwrap_model(model),
                    optimizer=optimizer,
                    step=step,
                    best_eval_loss=best_eval_loss,
                    extra=_checkpoint_training_state(data_epoch, microbatch_in_epoch),
                    include_optimizer=bool(checkpoint_retention["include_optimizer"]),
                    include_rng_state=bool(checkpoint_retention["include_rng_state"]),
                )
                _prune_retained_checkpoints(
                    run_dir,
                    prefix=str(checkpoint_retention["prefix"]),
                    keep_last=int(checkpoint_retention["keep_last"]),
                )
        if _checkpoint_benchmark_due(checkpoint_benchmarks, step=step, max_steps=max_steps):
            checkpoint_name = retained_checkpoint_name or f"{checkpoint_retention['prefix']}_{step:08d}"
            if retained_checkpoint_name is None and is_main_process:
                save_checkpoint(
                    run_dir / checkpoint_name,
                    model=dist_utils.unwrap_model(model),
                    optimizer=optimizer,
                    step=step,
                    best_eval_loss=best_eval_loss,
                    extra=_checkpoint_training_state(data_epoch, microbatch_in_epoch),
                    include_optimizer=False,
                    include_rng_state=False,
                )
            if is_main_process:
                benchmark_results = run_checkpoint_benchmarks(
                    run_dir,
                    config,
                    checkpoint=checkpoint_name,
                    step=step,
                    project_root=Path(__file__).resolve().parents[3],
                )
                benchmark_row = _benchmark_results_row(benchmark_results, step=step)
                if benchmark_log is not None:
                    benchmark_log.write(benchmark_row)
                _wandb_log(wandb_run, "benchmark", benchmark_row)
            if distributed:
                dist_utils.barrier()
    if is_main_process and not (run_dir / "routing_report.json").exists():
        make_routing_report(run_dir)
    if terminal_dashboard_writer is not None:
        terminal_dashboard_writer.write_status("complete", step=max_steps)
    _finish_wandb(wandb_run, final_step=max_steps, best_eval_loss=best_eval_loss)
    dist_utils.barrier()
    dist_utils.destroy_distributed()
    return run_dir


def _forward_for_stage(
    model: Any,
    batch: "torch.Tensor",
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    collect_router_space: bool = False,
    collect_bdre_visualization: bool = False,
    summarize_routing: bool = True,
) -> dict:
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
        routing_constraints=_mapping_config(dict(routing_cfg), "constraints"),
        routing_options=routing_cfg,
        hard_exit=_bool_mapping_value(
            routing_cfg,
            "hard_exit",
            default=str(config.get("stage")) == "stage4_output_action",
            name="routing.hard_exit",
        ),
        log_path_counts=_bool_mapping_value(
            routing_cfg,
            "log_path_counts",
            default=False,
            name="routing.log_path_counts",
        ),
        router_probability=router_probability,
        global_step=global_step,
        collect_router_space=collect_router_space,
        collect_bdre_visualization=collect_bdre_visualization,
        summarize_routing=summarize_routing,
    )
    if schedule_values:
        outputs["schedule_values"] = schedule_values
    return outputs


def _backward_stateful_tbptt_microbatch(
    model: Any,
    batch: "torch.Tensor",
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    chunk_size: int,
    detach_interval_chunks: int = 1,
    gradient_scale: float,
    device: "torch.device",
    collect_router_space: bool = False,
    collect_bdre_visualization: bool = False,
    summarize_routing: bool = True,
) -> dict[str, Any]:
    if route_mode in {"baseline", "parallel"}:
        raise ValueError("stateful_tbptt supports routed, token-serial BDRE stages only.")
    if batch.dim() != 2 or batch.size(1) < 2:
        raise ValueError("stateful_tbptt requires batches with at least two tokens per sequence.")
    if chunk_size < 1:
        raise ValueError("stateful_tbptt chunk_size must be positive.")
    if detach_interval_chunks < 1:
        raise ValueError("stateful_tbptt detach_interval_chunks must be positive.")

    routing_cfg = _mapping_config(config, "routing")
    loss_weights = dict(_mapping_config(config, "loss_weights"))
    schedule_values = _schedule_values(config, route_mode=route_mode, global_step=global_step)
    router_probability = schedule_values.get("scheduled_router_probability")
    if "scheduled_lambda_route" in schedule_values:
        loss_weights["route"] = schedule_values["scheduled_lambda_route"]
    pseudo_policy = str(routing_cfg.get("pseudo_policy", "sequential"))
    route_model = dist_utils.unwrap_model(model)
    route_targets = route_model.prepare_stream_route_targets(
        batch,
        route_mode=route_mode,
        pseudo_policy=pseudo_policy,
    )
    constraints = _mapping_config(dict(routing_cfg), "constraints")
    hard_exit = _bool_mapping_value(
        routing_cfg,
        "hard_exit",
        default=str(config.get("stage")) == "stage4_output_action",
        name="routing.hard_exit",
    )
    log_path_counts = _bool_mapping_value(
        routing_cfg,
        "log_path_counts",
        default=False,
        name="routing.log_path_counts",
    )
    sequence_length = int(batch.size(1))
    loss_token_count = int(batch.size(0) * (sequence_length - 1))
    state = None
    total_loss = None
    total_components: dict[str, torch.Tensor] = {}
    final_outputs: dict[str, Any] | None = None
    pending_loss: torch.Tensor | None = None
    chunks_in_pending_graph = 0
    backward_groups = 0

    for start in range(0, sequence_length, chunk_size):
        end = min(sequence_length, start + chunk_size)
        final_chunk = end == sequence_length
        input_chunk = batch[:, start:end]
        next_targets = torch.full_like(input_chunk, -100)
        valid_tokens = max(0, min(end, sequence_length - 1) - start)
        if valid_tokens:
            next_targets[:, :valid_tokens] = batch[:, start + 1 : start + 1 + valid_tokens]

        with _autocast_context(device, str(config.get("precision", "fp32"))):
            outputs = model(
                input_chunk,
                stream_chunk_options={
                    "state": state,
                    "next_token_targets": next_targets,
                    "loss_token_count": loss_token_count,
                    "include_auxiliary_losses": final_chunk,
                    "route_targets": route_targets,
                    "route_mode": route_mode,
                    "pseudo_policy": pseudo_policy,
                    "loss_weights": loss_weights,
                    "routing_constraints": constraints,
                    "routing_options": routing_cfg,
                    "hard_exit": hard_exit,
                    "log_path_counts": log_path_counts,
                    "router_probability": router_probability,
                    "global_step": global_step,
                    "collect_router_space": collect_router_space and final_chunk,
                    "collect_bdre_visualization": collect_bdre_visualization and final_chunk,
                    "summarize_routing": summarize_routing and final_chunk,
                },
            )
            chunk_loss = outputs["loss"]
        pending_loss = chunk_loss if pending_loss is None else pending_loss + chunk_loss
        chunks_in_pending_graph += 1

        detached_loss = chunk_loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        for name, value in outputs.get("loss_components", {}).items():
            detached = value.detach()
            total_components[name] = detached if name not in total_components else total_components[name] + detached
        state = outputs["incremental_state"]
        detach_boundary = chunks_in_pending_graph == detach_interval_chunks or final_chunk
        if detach_boundary:
            (pending_loss * gradient_scale).backward()
            state = state.detached()
            pending_loss = None
            chunks_in_pending_graph = 0
            backward_groups += 1
        if final_chunk:
            final_outputs = outputs

    assert final_outputs is not None and total_loss is not None and pending_loss is None
    final_outputs.pop("incremental_state", None)
    final_outputs["loss"] = total_loss
    final_outputs["loss_components"] = total_components
    final_outputs["stateful_tbptt_chunks"] = math.ceil(sequence_length / chunk_size)
    final_outputs["stateful_tbptt_backward_groups"] = backward_groups
    final_outputs["stateful_tbptt_gradient_horizon_tokens"] = min(
        sequence_length,
        chunk_size * detach_interval_chunks,
    )
    if schedule_values:
        final_outputs["schedule_values"] = schedule_values
    return final_outputs


def _routing_summary_due(config: Mapping[str, Any], *, global_step: int) -> bool:
    routing_cfg = _mapping_config(config, "routing")
    raw_interval = routing_cfg.get("summary_interval", 1)
    try:
        interval = int(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("routing.summary_interval must be an integer.") from exc
    if interval < 0:
        raise ValueError("routing.summary_interval must be >= 0.")
    if interval == 0:
        return False
    return global_step % interval == 0


def _set_activation_checkpointing(model: Any, enabled: bool) -> None:
    if hasattr(model, "activation_checkpointing"):
        model.activation_checkpointing = enabled


def _validate_stateful_chunk_semantics(
    model: Any,
    stateful_tbptt: Mapping[str, int | bool],
) -> None:
    if not bool(stateful_tbptt["enabled"]):
        return
    bdre_config = getattr(model, "bdre_config", None)
    if bdre_config is None or getattr(bdre_config, "depth_visibility_policy", None) != "full_bank":
        return
    train_chunk_size = int(stateful_tbptt["chunk_size"])
    model_chunk_size = int(bdre_config.chunk_size)
    if train_chunk_size != model_chunk_size:
        raise ValueError(
            "CPBC-FB requires stateful_tbptt.chunk_size to equal execution.chunk_size "
            f"so train and full-forward evaluation share completion boundaries; got "
            f"{train_chunk_size} and {model_chunk_size}."
        )


def _set_float32_matmul_precision(config: Mapping[str, Any]) -> None:
    value = config.get("float32_matmul_precision")
    if value is None:
        return
    precision = str(value)
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("float32_matmul_precision must be 'highest', 'high', or 'medium'.")
    torch.set_float32_matmul_precision(precision)


def _wrap_distributed_model(
    model: Any,
    device: "torch.device",
    *,
    distributed: bool,
    find_unused_parameters: bool,
    static_graph: bool,
    gradient_as_bucket_view: bool,
    broadcast_buffers: bool,
) -> Any:
    if not distributed:
        return model
    if DistributedDataParallel is None:
        raise ModuleNotFoundError("DistributedDataParallel is required for distributed training.")
    if device.type == "cuda":
        device_index = device.index if device.index is not None else dist_utils.local_rank()
        return DistributedDataParallel(
            model,
            device_ids=[device_index],
            output_device=device_index,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
            gradient_as_bucket_view=gradient_as_bucket_view,
            broadcast_buffers=broadcast_buffers,
        )
    return DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=broadcast_buffers,
    )


def _init_wandb(
    config: dict[str, Any],
    *,
    resolved_config: dict[str, Any],
    model_stats: dict[str, Any],
    run_dir: Path,
    is_main_process: bool,
) -> Any | None:
    wandb_cfg = _wandb_config(config)
    if not is_main_process or not _bool_value(wandb_cfg.get("enabled", False), "wandb.enabled"):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install `wandb` or set `wandb.enabled: false` to disable W&B logging.") from exc

    name = str(wandb_cfg.get("name", "auto"))
    if name == "auto":
        name = run_dir.name
    init_kwargs: dict[str, Any] = {
        "project": str(wandb_cfg.get("project", "brian-sphere-llm")),
        "name": name,
        "dir": str(run_dir),
        "config": {
            "train": resolved_config,
            "model_stats": model_stats,
            "run_dir": str(run_dir),
        },
    }
    for key in ("entity", "group", "job_type", "mode", "id", "resume"):
        if key in wandb_cfg and wandb_cfg[key] is not None:
            init_kwargs[key] = wandb_cfg[key]
    if "tags" in wandb_cfg:
        init_kwargs["tags"] = _wandb_tags(wandb_cfg["tags"])
    return wandb.init(**init_kwargs)


def _wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(_mapping_config(config, "wandb"))


def _wandb_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError("wandb.tags must be a list.")
    return [str(item) for item in value]


def _wandb_log(wandb_run: Any | None, prefix: str, row: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    step = row.get("step")
    payload = {
        f"{prefix}/{key}": value
        for key, value in row.items()
        if key != "step" and _wandb_loggable(value)
    }
    if not payload:
        return
    if isinstance(step, int) and not isinstance(step, bool):
        wandb_run.log(payload, step=step)
        return
    wandb_run.log(payload)


def _route_path_visualization_config(config: dict[str, Any], *, default_interval: int) -> dict[str, Any]:
    cfg = dict(_mapping_config(config, "route_path_visualization"))
    enabled = _bool_value(cfg.get("enabled", False), "route_path_visualization.enabled")
    interval = _int_config(
        {"interval": cfg.get("interval", default_interval)},
        "interval",
        minimum=1,
    )
    top_paths = _int_config({"top_paths": cfg.get("top_paths", 64)}, "top_paths", minimum=1)
    timeline_max_frames = _int_config(
        {"timeline_max_frames": cfg.get("timeline_max_frames", 100)},
        "timeline_max_frames",
        minimum=1,
    )
    upload_to_wandb = _bool_value(
        cfg.get("upload_to_wandb", True),
        "route_path_visualization.upload_to_wandb",
    )
    output_dir = str(cfg.get("output_dir", "route_path_visualizations"))
    if not output_dir:
        raise ValueError("route_path_visualization.output_dir must be a non-empty string.")
    wandb_key = str(cfg.get("wandb_key", "route_paths"))
    if not wandb_key:
        raise ValueError("route_path_visualization.wandb_key must be a non-empty string.")
    return {
        "enabled": enabled,
        "interval": interval,
        "top_paths": top_paths,
        "timeline_max_frames": timeline_max_frames,
        "upload_to_wandb": upload_to_wandb,
        "output_dir": output_dir,
        "wandb_key": wandb_key,
    }


def _router_space_visualization_config(config: dict[str, Any], *, default_interval: int) -> dict[str, Any]:
    cfg = dict(_mapping_config(config, "router_space_visualization"))
    enabled = _bool_value(cfg.get("enabled", False), "router_space_visualization.enabled")
    interval = _int_config(
        {"interval": cfg.get("interval", default_interval)},
        "interval",
        minimum=1,
    )
    max_points = _int_config({"max_points": cfg.get("max_points", 2048)}, "max_points", minimum=1)
    upload_to_wandb = _bool_value(
        cfg.get("upload_to_wandb", True),
        "router_space_visualization.upload_to_wandb",
    )
    output_dir = str(cfg.get("output_dir", "router_space_visualizations"))
    if not output_dir:
        raise ValueError("router_space_visualization.output_dir must be a non-empty string.")
    wandb_key = str(cfg.get("wandb_key", "router_space"))
    if not wandb_key:
        raise ValueError("router_space_visualization.wandb_key must be a non-empty string.")
    return {
        "enabled": enabled,
        "interval": interval,
        "max_points": max_points,
        "upload_to_wandb": upload_to_wandb,
        "output_dir": output_dir,
        "wandb_key": wandb_key,
    }


def _bdre_cache_visualization_config(config: dict[str, Any], *, default_interval: int) -> dict[str, Any]:
    cfg = dict(_mapping_config(config, "bdre_cache_visualization"))
    enabled = _bool_value(cfg.get("enabled", False), "bdre_cache_visualization.enabled")
    interval = _int_config(
        {"interval": cfg.get("interval", default_interval)},
        "interval",
        minimum=1,
    )
    sample_index = _int_config({"sample_index": cfg.get("sample_index", 0)}, "sample_index", minimum=0)
    upload_to_wandb = _bool_value(
        cfg.get("upload_to_wandb", True),
        "bdre_cache_visualization.upload_to_wandb",
    )
    output_dir = str(cfg.get("output_dir", "bdre_cache_visualizations"))
    if not output_dir:
        raise ValueError("bdre_cache_visualization.output_dir must be a non-empty string.")
    wandb_key = str(cfg.get("wandb_key", "bdre_cache"))
    if not wandb_key:
        raise ValueError("bdre_cache_visualization.wandb_key must be a non-empty string.")
    return {
        "enabled": enabled,
        "interval": interval,
        "sample_index": sample_index,
        "upload_to_wandb": upload_to_wandb,
        "output_dir": output_dir,
        "wandb_key": wandb_key,
    }


def _visualization_due(config: Mapping[str, Any], *, step: int, max_steps: int) -> bool:
    if not config.get("enabled"):
        return False
    interval = int(config["interval"])
    return step % interval == 0 or step == max_steps


def _checkpoint_retention_config(config: dict[str, Any], *, default_interval: int) -> dict[str, Any]:
    cfg = dict(_mapping_config(config, "checkpoint_retention"))
    enabled = _bool_value(cfg.get("enabled", False), "checkpoint_retention.enabled")
    interval = _int_config(
        {"interval": cfg.get("interval", default_interval)},
        "interval",
        minimum=1,
    )
    keep_last = _int_config({"keep_last": cfg.get("keep_last", 0)}, "keep_last", minimum=0)
    prefix = str(cfg.get("prefix", "checkpoint_step"))
    if not prefix:
        raise ValueError("checkpoint_retention.prefix must be a non-empty string.")
    include_optimizer = _bool_value(
        cfg.get("include_optimizer", False),
        "checkpoint_retention.include_optimizer",
    )
    include_rng_state = _bool_value(
        cfg.get("include_rng_state", False),
        "checkpoint_retention.include_rng_state",
    )
    return {
        "enabled": enabled,
        "interval": interval,
        "keep_last": keep_last,
        "prefix": prefix,
        "include_optimizer": include_optimizer,
        "include_rng_state": include_rng_state,
    }


def _checkpoint_benchmark_config(config: dict[str, Any], *, default_interval: int) -> dict[str, Any]:
    cfg = dict(_mapping_config(config, "checkpoint_benchmarks"))
    enabled = _bool_value(cfg.get("enabled", False), "checkpoint_benchmarks.enabled")
    interval = _int_config(
        {"interval": cfg.get("interval", default_interval)},
        "interval",
        minimum=1,
    )
    return {
        "enabled": enabled,
        "interval": interval,
    }


def _retained_checkpoint_due(
    retention: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
    *,
    step: int,
    max_steps: int,
) -> bool:
    if _checkpoint_benchmark_due(benchmarks, step=step, max_steps=max_steps):
        return True
    if not retention.get("enabled"):
        return False
    interval = int(retention["interval"])
    return step % interval == 0 or step == max_steps


def _checkpoint_benchmark_due(config: Mapping[str, Any], *, step: int, max_steps: int) -> bool:
    if not config.get("enabled"):
        return False
    interval = int(config["interval"])
    return step % interval == 0 or step == max_steps


def _prune_retained_checkpoints(run_dir: Path, *, prefix: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        (path for path in run_dir.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: _checkpoint_step_from_name(path.name, prefix=prefix),
    )
    for checkpoint_dir in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _checkpoint_step_from_name(name: str, *, prefix: str) -> int:
    prefix_text = f"{prefix}_"
    if not name.startswith(prefix_text):
        return -1
    try:
        return int(name[len(prefix_text) :])
    except ValueError:
        return -1


def _benchmark_results_row(results: list[dict[str, Any]], *, step: int) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step}
    for result in results:
        name = str(result.get("name", "benchmark"))
        row[f"{name}_returncode"] = result.get("returncode")
        row[f"{name}_elapsed_seconds"] = result.get("elapsed_seconds")
        if result.get("output_path") is not None:
            row[f"{name}_output_path"] = str(result["output_path"])
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                row[f"{name}_{key}"] = value
    return row


def _maybe_log_route_path_visualization(
    wandb_run: Any | None,
    *,
    run_dir: Path,
    model: Any,
    step: int,
    max_steps: int,
    config: Mapping[str, Any],
) -> Path | None:
    if not config.get("enabled"):
        return None
    interval = int(config["interval"])
    if step % interval != 0 and step != max_steps:
        return None
    output_dir = run_dir / str(config["output_dir"])
    output_path = output_dir / f"route_paths_step_{step:08d}.html"
    try:
        html_path = make_route_path_visualization_from_train_log(
            run_dir,
            model,
            output_path=output_path,
            step=step,
            top_paths=int(config["top_paths"]),
            timeline_max_frames=int(config["timeline_max_frames"]),
        )
    except Exception as exc:  # pragma: no cover - defensive; visualization must not kill training.
        error = {
            "step": step,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        JsonlLogger(run_dir / "route_path_visualization_errors.jsonl").write(error)
        _wandb_log_visualization_error(wandb_run, error)
        return None
    if not _route_path_visualization_has_paths(html_path):
        error = {
            "step": step,
            "error_type": "EmptyRoutePathVisualization",
            "error": "route path visualization had no path data; skipped W&B upload.",
            "html_path": str(html_path),
            "json_path": str(html_path.with_suffix(".json")),
        }
        JsonlLogger(run_dir / "route_path_visualization_errors.jsonl").write(error)
        _wandb_log_visualization_error(wandb_run, error)
        return html_path
    if config.get("upload_to_wandb"):
        _wandb_log_html(
            wandb_run,
            key=str(config["wandb_key"]),
            html_path=html_path,
            step=step,
        )
    return html_path


def _route_path_visualization_has_paths(html_path: Path) -> bool:
    sidecar_path = html_path.with_suffix(".json")
    try:
        report = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        return False
    return checks.get("paths_present") is True


def _maybe_log_router_space_visualization(
    wandb_run: Any | None,
    *,
    run_dir: Path,
    model: Any,
    payload: dict[str, Any] | None,
    step: int,
    max_steps: int,
    config: Mapping[str, Any],
) -> Path | None:
    if not _visualization_due(config, step=step, max_steps=max_steps):
        return None
    if payload is None:
        return None
    output_dir = run_dir / str(config["output_dir"])
    output_path = output_dir / f"router_space_step_{step:08d}.html"
    try:
        html_path = make_router_space_visualization_from_payload(
            payload,
            model,
            output_path=output_path,
            step=step,
            max_points=int(config["max_points"]),
            metadata={"run_dir": str(run_dir), "source": "train_step"},
        )
    except Exception as exc:  # pragma: no cover - defensive; visualization must not kill training.
        error = {
            "step": step,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        JsonlLogger(run_dir / "router_space_visualization_errors.jsonl").write(error)
        _wandb_log_named_visualization_error(wandb_run, "router_space", error)
        return None
    if config.get("upload_to_wandb"):
        _wandb_log_html(
            wandb_run,
            key=str(config["wandb_key"]),
            html_path=html_path,
            step=step,
        )
    return html_path


def _maybe_log_bdre_cache_visualization(
    wandb_run: Any | None,
    *,
    run_dir: Path,
    model: Any,
    payload: dict[str, Any] | None,
    step: int,
    max_steps: int,
    config: Mapping[str, Any],
) -> Path | None:
    if not _visualization_due(config, step=step, max_steps=max_steps) or payload is None:
        return None
    output_dir = run_dir / str(config["output_dir"])
    output_path = output_dir / f"bdre_cache_step_{step:08d}.html"
    try:
        html_path = make_bdre_cache_visualization_from_payload(
            payload,
            model,
            output_path=output_path,
            step=step,
            sample_index=int(config["sample_index"]),
            metadata={"run_dir": str(run_dir), "source": "train_step"},
        )
    except Exception as exc:  # pragma: no cover - visualization must not kill training.
        error = {
            "step": step,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        JsonlLogger(run_dir / "bdre_cache_visualization_errors.jsonl").write(error)
        _wandb_log_named_visualization_error(wandb_run, "bdre_cache", error)
        return None
    if config.get("upload_to_wandb"):
        _wandb_log_html(
            wandb_run,
            key=str(config["wandb_key"]),
            html_path=html_path,
            step=step,
        )
    return html_path


def _wandb_log_html(wandb_run: Any | None, *, key: str, html_path: Path, step: int) -> None:
    if wandb_run is None:
        return
    import wandb

    html_text = html_path.read_text(encoding="utf-8")
    sidecar_path = html_path.with_suffix(".json")
    payload: dict[str, Any] = {
        f"visualization/{key}": wandb.Html(html_text),
        f"visualization/{key}_html_path": str(html_path),
        f"visualization/{key}_json_path": str(sidecar_path),
    }
    wandb_run.log(payload, step=step)
    if hasattr(wandb_run, "save"):
        wandb_run.save(str(html_path))
        if sidecar_path.exists():
            wandb_run.save(str(sidecar_path))


def _wandb_log_visualization_error(wandb_run: Any | None, error: dict[str, Any]) -> None:
    _wandb_log_named_visualization_error(wandb_run, "route_paths", error)


def _wandb_log_named_visualization_error(wandb_run: Any | None, key: str, error: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    step = error.get("step")
    payload = {
        f"visualization/{key}_error_type": error.get("error_type"),
        f"visualization/{key}_error": error.get("error"),
    }
    if isinstance(step, int) and not isinstance(step, bool):
        wandb_run.log(payload, step=step)
    else:
        wandb_run.log(payload)


def _wandb_loggable(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        return True
    return value is None


def _finish_wandb(wandb_run: Any | None, *, final_step: int, best_eval_loss: float | None) -> None:
    if wandb_run is None:
        return
    wandb_run.summary["final_step"] = final_step
    if best_eval_loss is not None:
        wandb_run.summary["best_eval_loss"] = best_eval_loss
    wandb_run.finish()


def _gradient_sync_context(model: Any, *, distributed: bool, should_sync: bool):
    if distributed and not should_sync and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _empty_stateful_gradient_sync_stats() -> dict[str, int]:
    return {
        "stateful_ddp_gradient_sync_buckets": 0,
        "stateful_ddp_global_used_parameters": 0,
        "stateful_ddp_local_missing_parameters": 0,
        "stateful_ddp_gradient_sync_bytes": 0,
    }


def _sync_stateful_ddp_gradients(model: Any, *, bucket_cap_mb: int) -> dict[str, int]:
    if bucket_cap_mb < 1:
        raise ValueError("stateful DDP gradient bucket size must be positive.")
    if not dist_utils.is_initialized():
        raise RuntimeError("stateful DDP gradient synchronization requires an initialized process group.")

    parameters = [
        parameter
        for parameter in dist_utils.unwrap_model(model).parameters()
        if parameter.requires_grad
    ]
    stats = _empty_stateful_gradient_sync_stats()
    if not parameters:
        return stats

    mask_device = parameters[0].device
    local_used = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.int32,
        device=mask_device,
    )
    globally_used = local_used.clone()
    torch.distributed.all_reduce(globally_used, op=torch.distributed.ReduceOp.MAX)
    globally_used_flags = globally_used.bool().tolist()
    world_size = torch.distributed.get_world_size()
    bucket_cap_bytes = int(bucket_cap_mb) * 1024 * 1024
    bucket: list[torch.Tensor] = []
    bucket_parameters: list[torch.nn.Parameter] = []
    bucket_bytes = 0
    bucket_key: tuple[torch.device, torch.dtype] | None = None

    def flush_bucket() -> None:
        nonlocal bucket, bucket_parameters, bucket_bytes, bucket_key
        if not bucket:
            return
        flat = torch.cat(bucket)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for parameter in bucket_parameters:
            count = parameter.numel()
            synchronized = flat.narrow(0, offset, count).view_as(parameter)
            if parameter.grad is None:
                parameter.grad = synchronized.clone()
            else:
                parameter.grad.detach().copy_(synchronized)
            offset += count
        stats["stateful_ddp_gradient_sync_buckets"] += 1
        stats["stateful_ddp_gradient_sync_bytes"] += int(flat.numel() * flat.element_size())
        bucket = []
        bucket_parameters = []
        bucket_bytes = 0
        bucket_key = None

    for parameter, global_used in zip(parameters, globally_used_flags):
        if not global_used:
            continue
        stats["stateful_ddp_global_used_parameters"] += 1
        if parameter.grad is None:
            stats["stateful_ddp_local_missing_parameters"] += 1
            gradient = torch.zeros_like(parameter, memory_format=torch.contiguous_format)
        else:
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                raise ValueError("stateful DDP does not support sparse parameter gradients.")
            if gradient.dtype != parameter.dtype or gradient.device != parameter.device:
                raise ValueError("stateful DDP gradients must match their parameter dtype and device.")
        entry = gradient.reshape(-1)
        entry_bytes = int(entry.numel() * entry.element_size())
        entry_key = (entry.device, entry.dtype)
        if bucket and (entry_key != bucket_key or bucket_bytes + entry_bytes > bucket_cap_bytes):
            flush_bucket()
        bucket.append(entry)
        bucket_parameters.append(parameter)
        bucket_bytes += entry_bytes
        bucket_key = entry_key
    flush_bucket()
    return stats


def _ddp_no_sync_microbatch_count(model: Any, *, distributed: bool, gradient_accumulation_steps: int) -> int:
    if distributed and gradient_accumulation_steps > 1 and hasattr(model, "no_sync"):
        return gradient_accumulation_steps - 1
    return 0


def _global_train_token_count(local_token_count: int, *, distributed: bool) -> int:
    if distributed:
        return local_token_count * dist_utils.world_size()
    return local_token_count


def _distributed_batch_contract(
    config: dict[str, Any],
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> dict[str, int | None]:
    if min(batch_size, gradient_accumulation_steps, world_size) < 1:
        raise ValueError("batch size, gradient accumulation, and world size must be positive.")
    expected_world_size = (
        _int_config(config, "expected_world_size", minimum=1)
        if "expected_world_size" in config
        else None
    )
    expected_global_batch_size = (
        _int_config(config, "expected_global_batch_size", minimum=1)
        if "expected_global_batch_size" in config
        else None
    )
    local_effective_batch_size = batch_size * gradient_accumulation_steps
    effective_batch_size = local_effective_batch_size * world_size
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(
            f"Expected world size {expected_world_size}, but the launch provided {world_size}."
        )
    if expected_global_batch_size is not None and effective_batch_size != expected_global_batch_size:
        raise ValueError(
            "Expected global batch size "
            f"{expected_global_batch_size}, but local batch {batch_size} x accumulation "
            f"{gradient_accumulation_steps} x world size {world_size} = {effective_batch_size}."
        )
    return {
        "local_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "local_effective_batch_size": local_effective_batch_size,
        "effective_batch_size": effective_batch_size,
        "expected_world_size": expected_world_size,
        "expected_global_batch_size": expected_global_batch_size,
    }


def _restore_dataloader_position(loader: Any, *, data_epoch: int, microbatch_in_epoch: int) -> tuple[Any, int, int]:
    loader_length = _loader_length(loader)
    if microbatch_in_epoch >= loader_length:
        data_epoch += microbatch_in_epoch // loader_length
        microbatch_in_epoch = microbatch_in_epoch % loader_length
    _set_sampler_epoch(loader, data_epoch)
    iterator = iter(loader)
    skipped = 0
    while skipped < microbatch_in_epoch:
        try:
            next(iterator)
        except StopIteration:
            data_epoch += 1
            microbatch_in_epoch = 0
            _set_sampler_epoch(loader, data_epoch)
            iterator = iter(loader)
            break
        skipped += 1
    return iterator, data_epoch, microbatch_in_epoch


def _next_train_batch(
    loader: Any,
    iterator: Any,
    *,
    data_epoch: int,
    microbatch_in_epoch: int,
) -> tuple[Any, Any, int, int]:
    try:
        batch = next(iterator)
    except StopIteration:
        data_epoch += 1
        microbatch_in_epoch = 0
        _set_sampler_epoch(loader, data_epoch)
        iterator = iter(loader)
        batch = next(iterator)
    microbatch_in_epoch += 1
    if microbatch_in_epoch >= _loader_length(loader):
        data_epoch += 1
        microbatch_in_epoch = 0
        _set_sampler_epoch(loader, data_epoch)
        iterator = iter(loader)
    return batch, iterator, data_epoch, microbatch_in_epoch


def _checkpoint_training_state(data_epoch: int, microbatch_in_epoch: int) -> dict[str, int]:
    return {
        "data_epoch": data_epoch,
        "microbatch_in_epoch": microbatch_in_epoch,
    }


def _save_rank_training_state(
    checkpoint_dir: Path,
    *,
    step: int,
    best_eval_loss: float | None,
    data_epoch: int,
    microbatch_in_epoch: int,
) -> Path:
    return save_rank_state(
        checkpoint_dir,
        rank=dist_utils.rank(),
        step=step,
        data_epoch=data_epoch,
        microbatch_in_epoch=microbatch_in_epoch,
        best_eval_loss=best_eval_loss,
    )


def _load_rank_training_state(checkpoint_dir: Path, *, rank: int, distributed: bool) -> tuple[dict[str, Any], bool]:
    state_path = rank_state_path(checkpoint_dir, rank=rank)
    if not distributed or not state_path.exists():
        return {}, False
    return load_rank_state(checkpoint_dir, rank=rank, restore_rng_state=True), True


def _loader_length(loader: Any) -> int:
    length = len(loader)
    if length <= 0:
        raise ValueError("Training dataloader must contain at least one batch.")
    return int(length)


def _payload_int(payload: Mapping[str, Any], key: str, *, default: int, minimum: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"checkpoint {key} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"checkpoint {key} must be an integer.")
    if number < minimum:
        raise ValueError(f"checkpoint {key} must be >= {minimum}.")
    return number


def _distributed_mean_metrics(metrics: dict[str, Any], *, device: "torch.device", distributed: bool) -> dict[str, Any]:
    if not distributed:
        return metrics
    reduced: dict[str, Any] = {}
    for key, value in metrics.items():
        number = _metric_number(value)
        if number is None:
            reduced[key] = value
        else:
            reduced[key] = dist_utils.mean_scalar(number, device=device)
    return reduced


def _distributed_mean_scalar(value: float, *, device: "torch.device", distributed: bool) -> float:
    if not distributed:
        return value
    return dist_utils.mean_scalar(value, device=device)


def _set_sampler_epoch(loader: Any, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _learning_rate_for_step(
    step: int,
    *,
    base_learning_rate: float,
    min_learning_rate: float,
    max_steps: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    if schedule == "constant":
        return base_learning_rate
    if schedule == "linear_warmup_cosine_decay":
        if warmup_steps > 0 and step <= warmup_steps:
            return base_learning_rate * (step / warmup_steps)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine
    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def _lr_schedule_config(config: dict[str, Any]) -> str:
    schedule = str(config.get("lr_schedule", "constant"))
    if schedule in {"constant", "linear_warmup_cosine_decay"}:
        return schedule
    raise ValueError("lr_schedule must be one of: constant, linear_warmup_cosine_decay.")


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
        if key == "route_path_counts" and isinstance(value, list):
            last_values[key] = _merge_route_path_counts(last_values.get(key), value)
            continue
        if key == "route_transition_counts" and isinstance(value, list):
            last_values[key] = _merge_route_transition_counts(last_values.get(key), value)
            continue
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


def _merge_route_path_counts(existing: Any, new: list[Any]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, ...], int] = {}
    for source in (existing, new):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, Mapping):
                continue
            actions_raw = item.get("actions")
            if not isinstance(actions_raw, list):
                continue
            try:
                actions = tuple(int(value) for value in actions_raw)
                count = int(item.get("count", 0))
            except (TypeError, ValueError):
                continue
            if actions and count > 0:
                counts[actions] = counts.get(actions, 0) + count
    return [
        {"actions": list(actions), "count": int(count)}
        for actions, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _merge_route_transition_counts(existing: Any, new: list[Any]) -> list[dict[str, int]]:
    counts: dict[tuple[int, int], int] = {}
    for source in (existing, new):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, Mapping):
                continue
            try:
                transition = (int(item["source"]), int(item["target"]))
                count = int(item.get("count", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if count > 0:
                counts[transition] = counts.get(transition, 0) + count
    return [
        {"source": int(source), "target": int(target), "count": int(count)}
        for (source, target), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


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
    max_batches = min(_int_config(config, "eval_max_batches", default=8, minimum=1), len(val_loader))
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


def _stateful_tbptt_config(config: dict[str, Any]) -> dict[str, int | bool]:
    raw = dict(_mapping_config(config, "stateful_tbptt"))
    enabled = _bool_config(raw, "enabled", default=False)
    chunk_size = _int_config(raw, "chunk_size", default=8, minimum=1)
    detach_interval_chunks = _int_config(
        raw,
        "detach_interval_chunks",
        default=1,
        minimum=1,
    )
    gradient_sync_bucket_mb = _int_config(raw, "gradient_sync_bucket_mb", default=64, minimum=1)
    return {
        "enabled": enabled,
        "chunk_size": chunk_size,
        "detach_interval_chunks": detach_interval_chunks,
        "gradient_sync_bucket_mb": gradient_sync_bucket_mb,
    }


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
        "path_exists": manifest_path.exists(),
        "tokenized_dir_exists": tokenized_dir.exists(),
        "stats_path_exists": stats_path.exists(),
        "tokenized_artifacts_present": _tokenized_artifacts_present(tokenized_dir),
    }
    if not stats_path.exists():
        ref["sha256_manifest_verified"] = False
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
        "manifest_row_count",
        "manifest_source_text_hashes_verified",
        "manifest_token_hashes_verified",
        "manifest_source_text_hash_failure_count",
        "manifest_token_hash_failure_count",
        "tokenizer_artifact_count",
        "tokenizer_artifacts_present",
        "tokenizer_artifact_hashes",
        "tokenizer_artifact_hashes_present",
    ]:
        if key in stats:
            ref[key] = stats[key]
    ref["sha256_manifest_verified"] = _manifest_hash_verified(manifest_path, stats)
    ref["stats_recipe_name_matches_config"] = stats.get("recipe_name") == data_config.get("recipe_name")
    ref["stats_sequence_length_matches_config"] = stats.get("sequence_length") == data_config.get("sequence_length")
    tokenizer = stats.get("tokenizer")
    if isinstance(tokenizer, dict):
        ref["tokenizer"] = {
            key: tokenizer.get(key)
            for key in ["name", "revision", "license", "vocab_size", "special_tokens"]
            if key in tokenizer
        }
    return ref


def _tokenized_artifacts_present(tokenized_dir: Path) -> bool:
    return all((tokenized_dir / name).exists() for name in ["train.bin", "train.idx", "val.bin", "val.idx"])


def _manifest_hash_verified(manifest_path: Path, stats: Mapping[str, Any]) -> bool:
    expected = stats.get("sha256_manifest")
    if not isinstance(expected, str) or not expected or not manifest_path.exists():
        return False
    actual = sha256_text(manifest_path.read_text(encoding="utf-8"))
    return actual == expected


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
