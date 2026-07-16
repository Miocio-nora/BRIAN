#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="Profile one BDRE stateful-TBPTT microbatch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--detach-interval-chunks", type=int, default=None)
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--flex-reader-group-size", type=int, default=None)
    parser.add_argument("--decoded-key-rope-mode", default=None)
    parser.add_argument("--writer-projection-mode", default=None)
    parser.add_argument("--route-pointwise-mode", default=None)
    parser.add_argument("--prefix-compile-mode", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=80)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--trace", default=None, help="Optional Chrome trace output path.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("profile_bdre_tbptt.py requires one visible CUDA GPU.")
    if args.warmup_steps < 0 or args.row_limit < 1:
        raise SystemExit("warmup-steps must be non-negative and row-limit must be positive.")

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

    set_seed(args.seed)
    torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
    device = torch.device("cuda")
    model = build_model_from_config(model_config_path).to(device).train()
    if any(
        value is not None
        for value in (
            args.flex_reader_group_size,
            args.decoded_key_rope_mode,
            args.writer_projection_mode,
            args.route_pointwise_mode,
            args.prefix_compile_mode,
        )
    ):
        model.bdre_config = replace(
            model.bdre_config,
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
        )
        model.bdre_config.validate()
    batch = torch.randint(
        0,
        int(model.config.base.vocab_size),
        (batch_size, sequence_length),
        device=device,
    )

    def run() -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        set_seed(args.seed + 1)
        return _backward_stateful_tbptt_microbatch(
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

    for _ in range(args.warmup_steps):
        run()
        torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as profiler:
        result = run()
        torch.cuda.synchronize(device)

    print("\n=== Top self CUDA time ===")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.row_limit,
        )
    )
    print("\n=== Top self CPU time ===")
    print(
        profiler.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=args.row_limit,
        )
    )

    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "key": event.key,
                "count": int(event.count),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "self_cuda_time_us": float(getattr(event, "self_device_time_total", 0.0)),
                "cuda_time_us": float(getattr(event, "device_time_total", 0.0)),
            }
        )
    rows.sort(key=lambda item: item["self_cuda_time_us"], reverse=True)
    payload = {
        "config": str(config_path),
        "model_config": str(model_config_path),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "chunk_size": chunk_size,
        "detach_interval_chunks": detach_interval_chunks,
        "flex_reader_group_size": model.bdre_config.flex_reader_group_size,
        "decoded_key_rope_mode": model.bdre_config.decoded_key_rope_mode,
        "writer_projection_mode": model.bdre_config.writer_projection_mode,
        "route_pointwise_mode": model.bdre_config.route_pointwise_mode,
        "prefix_compile_mode": model.bdre_config.prefix_compile_mode,
        "loss": float(result["loss"].detach().cpu()),
        "top_self_cuda": rows[: args.row_limit],
        "top_self_cpu": sorted(
            rows,
            key=lambda item: item["self_cpu_time_us"],
            reverse=True,
        )[: args.row_limit],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
