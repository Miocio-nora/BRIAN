#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from brian_sphere_llm.train.stage_runner import build_model_from_config, train_mode_for_stage
from brian_sphere_llm.train.trainer import _backward_stateful_tbptt_microbatch
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate one BDRE stateful-TBPTT backward on one GPU.")
    parser.add_argument("--config", required=True, help="Stateful-TBPTT train YAML config.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--detach-interval-chunks", type=int, default=None)
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--flex-kernel-variant", default=None)
    parser.add_argument("--flex-reader-group-size", type=int, default=None)
    parser.add_argument("--decoded-key-rope-mode", default=None)
    parser.add_argument("--writer-projection-mode", default=None)
    parser.add_argument("--route-pointwise-mode", default=None)
    parser.add_argument("--prefix-compile-mode", default=None)
    parser.add_argument("--reader-kernel", choices=("flex", "triton_fused"), default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark_bdre_tbptt.py requires one visible CUDA GPU.")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    tbptt = config.get("stateful_tbptt", {})
    if not isinstance(tbptt, dict) or tbptt.get("enabled") is not True:
        raise SystemExit("The selected config must enable stateful_tbptt.")

    model_config_path = (config_path.parent / str(config["model_config"])).resolve()
    data_config_path = (config_path.parent / str(config["data_config"])).resolve()
    data_config = load_config(data_config_path)
    batch_size = int(args.batch_size or config["batch_size"])
    sequence_length = int(args.sequence_length or data_config["sequence_length"])
    chunk_size = int(args.chunk_size or tbptt["chunk_size"])
    detach_interval_chunks = int(
        args.detach_interval_chunks or tbptt.get("detach_interval_chunks", 1)
    )
    if (
        min(batch_size, sequence_length, chunk_size, detach_interval_chunks) < 1
        or sequence_length < 2
    ):
        raise SystemExit(
            "Batch size, chunk size, and detach interval must be positive; "
            "sequence length must be at least two."
        )
    if args.warmup_steps < 0 or args.repeats < 1:
        raise SystemExit("warmup steps must be non-negative and repeats must be positive.")

    set_seed(args.seed)
    torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
    device = torch.device("cuda")
    model = build_model_from_config(model_config_path).to(device).train()
    if any(
        value is not None
        for value in (
            args.flex_kernel_variant,
            args.flex_reader_group_size,
            args.decoded_key_rope_mode,
            args.writer_projection_mode,
            args.route_pointwise_mode,
            args.prefix_compile_mode,
            args.reader_kernel,
        )
    ):
        model.bdre_config = replace(
            model.bdre_config,
            flex_kernel_variant=(
                args.flex_kernel_variant
                if args.flex_kernel_variant is not None
                else model.bdre_config.flex_kernel_variant
            ),
            flex_reader_group_size=(
                args.flex_reader_group_size
                if args.flex_reader_group_size is not None
                else model.bdre_config.flex_reader_group_size
            ),
            decoded_key_rope_mode=(
                args.decoded_key_rope_mode
                if args.decoded_key_rope_mode is not None
                else model.bdre_config.decoded_key_rope_mode
            ),
            writer_projection_mode=(
                args.writer_projection_mode
                if args.writer_projection_mode is not None
                else model.bdre_config.writer_projection_mode
            ),
            route_pointwise_mode=(
                args.route_pointwise_mode
                if args.route_pointwise_mode is not None
                else model.bdre_config.route_pointwise_mode
            ),
            prefix_compile_mode=(
                args.prefix_compile_mode
                if args.prefix_compile_mode is not None
                else model.bdre_config.prefix_compile_mode
            ),
            reader_kernel_mode=(
                args.reader_kernel
                if args.reader_kernel is not None
                else model.bdre_config.reader_kernel_mode
            ),
        )
        model.bdre_config.validate()
    vocab_size = int(model.config.base.vocab_size)
    batch = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    status = "passed"
    error: str | None = None
    measurements: list[dict[str, Any]] = []

    def run_backward(*, measured: bool) -> dict[str, Any] | None:
        nonlocal status, error
        model.zero_grad(set_to_none=True)
        # Reset routing/noise RNG so every repeat measures the same mathematical work.
        set_seed(args.seed + 1)
        if measured:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            result = _backward_stateful_tbptt_microbatch(
                model,
                batch,
                config=config,
                route_mode=train_mode_for_stage(str(config["stage"])),
                global_step=args.global_step,
                chunk_size=chunk_size,
                detach_interval_chunks=detach_interval_chunks,
                gradient_scale=1.0,
                device=device,
                summarize_routing=False,
            )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as exc:
            status = "oom"
            error = str(exc)
            torch.cuda.synchronize(device)
            return None
        elapsed = time.perf_counter() - started
        if measured:
            measurements.append(
                {
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": batch.numel() / elapsed,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
                    "loss": float(result["loss"].cpu()),
                    "loss_components": {
                        name: float(value.cpu())
                        for name, value in result.get("loss_components", {}).items()
                    },
                }
            )
        return result

    for _ in range(args.warmup_steps):
        if run_backward(measured=False) is None:
            break
    if status == "passed":
        for _ in range(args.repeats):
            if run_backward(measured=True) is None:
                break

    elapsed_samples = [float(item["elapsed_seconds"]) for item in measurements]
    throughput_samples = [float(item["tokens_per_second"]) for item in measurements]
    median_elapsed = statistics.median(elapsed_samples) if elapsed_samples else 0.0
    mean_elapsed = statistics.fmean(elapsed_samples) if elapsed_samples else 0.0
    median_throughput = statistics.median(throughput_samples) if throughput_samples else 0.0
    mean_throughput = statistics.fmean(throughput_samples) if throughput_samples else 0.0
    report = {
        "status": status,
        "config": str(config_path),
        "model_config": str(model_config_path),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "chunk_size": chunk_size,
        "detach_interval_chunks": detach_interval_chunks,
        "gradient_horizon_tokens": min(
            sequence_length,
            chunk_size * detach_interval_chunks,
        ),
        "global_step": args.global_step,
        "flex_kernel_variant": model.bdre_config.flex_kernel_variant,
        "flex_reader_group_size": model.bdre_config.flex_reader_group_size,
        "decoded_key_rope_mode": model.bdre_config.decoded_key_rope_mode,
        "writer_projection_mode": model.bdre_config.writer_projection_mode,
        "route_pointwise_mode": model.bdre_config.route_pointwise_mode,
        "prefix_compile_mode": model.bdre_config.prefix_compile_mode,
        "reader_kernel": model.bdre_config.reader_kernel_mode,
        "cache_layout": model.bdre_config.cache_layout,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "elapsed_seconds": median_elapsed,
        "elapsed_seconds_mean": mean_elapsed,
        "tokens_per_second": median_throughput,
        "tokens_per_second_mean": mean_throughput,
        "peak_allocated_mb": max(
            (float(item["peak_allocated_mb"]) for item in measurements), default=0.0
        ),
        "peak_reserved_mb": max(
            (float(item["peak_reserved_mb"]) for item in measurements), default=0.0
        ),
        "final_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "loss": measurements[-1]["loss"] if measurements else None,
        "measurements": measurements,
        "error": error,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
