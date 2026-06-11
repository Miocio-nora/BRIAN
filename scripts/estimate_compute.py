#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.compute_report import estimate_gpu_hours, make_compute_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate rough training GPU-hours.")
    parser.add_argument("--params", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--tflops-per-gpu", type=float, default=989.0)
    parser.add_argument("--utilization", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories for a compute-quality report.")
    parser.add_argument("--baseline-run", default=None, help="Baseline run directory for comparison.")
    parser.add_argument("--output", default=None, help="Optional compute report output path.")
    args = parser.parse_args()
    if args.runs:
        report = make_compute_report(
            args.runs,
            baseline_run=args.baseline_run,
            output_path=args.output,
            tflops_per_gpu=args.tflops_per_gpu,
            utilization=args.utilization,
        )
        print(report)
        return
    if args.params is None or args.tokens is None:
        raise SystemExit("single estimate mode requires --params and --tokens")
    print(estimate_gpu_hours(args.params, args.tokens, args.tflops_per_gpu, args.utilization, args.gamma))


if __name__ == "__main__":
    main()
