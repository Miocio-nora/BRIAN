#!/usr/bin/env python
from __future__ import annotations

import argparse


def estimate_gpu_hours(params: int, tokens: int, tflops_per_gpu: float, utilization: float, gamma: float) -> float:
    flops = 6 * params * tokens * gamma
    return flops / (tflops_per_gpu * 1e12 * utilization) / 3600


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate rough training GPU-hours.")
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--tflops-per-gpu", type=float, default=989.0)
    parser.add_argument("--utilization", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    print(estimate_gpu_hours(args.params, args.tokens, args.tflops_per_gpu, args.utilization, args.gamma))


if __name__ == "__main__":
    main()
