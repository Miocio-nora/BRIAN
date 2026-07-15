#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
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
    if min(batch_size, sequence_length, chunk_size) < 1 or sequence_length < 2:
        raise SystemExit("batch size and chunk size must be positive; sequence length must be at least two.")

    set_seed(args.seed)
    torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
    device = torch.device("cuda")
    model = build_model_from_config(model_config_path).to(device).train()
    vocab_size = int(model.config.base.vocab_size)
    batch = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    status = "passed"
    error: str | None = None
    output: dict[str, Any] | None = None
    try:
        output = _backward_stateful_tbptt_microbatch(
            model,
            batch,
            config=config,
            route_mode=train_mode_for_stage(str(config["stage"])),
            global_step=args.global_step,
            chunk_size=chunk_size,
            gradient_scale=1.0,
            device=device,
            summarize_routing=False,
        )
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {
        "status": status,
        "config": str(config_path),
        "model_config": str(model_config_path),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "chunk_size": chunk_size,
        "global_step": args.global_step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": batch.numel() / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
        "final_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "loss": float(output["loss"].cpu()) if output is not None else None,
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
