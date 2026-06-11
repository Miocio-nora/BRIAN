#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.difficulty_report import make_difficulty_report
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report
from brian_sphere_llm.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BRIAN-Sphere run.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--run", default=None, help="Run directory.")
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories for stage gate reports.")
    parser.add_argument("--baseline-run", default=None, help="Baseline run directory for difficulty-step eval.")
    parser.add_argument("--routed-run", default=None, help="Routed run directory for difficulty-step eval.")
    parser.add_argument("--output", default=None, help="Optional output path override.")
    parser.add_argument("--split", default=None, help="Dataset split override.")
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum eval batches override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    args = parser.parse_args()
    config = load_config(args.config)
    eval_name = config.get("eval_name", "routing_eval")
    if eval_name == "stage_gate_eval":
        runs = args.runs or ([args.run] if args.run else [])
        if not runs:
            raise SystemExit("stage_gate_eval requires --runs or --run")
        report = make_stage_gate_report(
            runs,
            output_path=args.output or config.get("output_path"),
            thresholds=config.get("thresholds", {}),
        )
    elif eval_name == "difficulty_step_eval":
        baseline_run = args.baseline_run or config.get("baseline_run")
        routed_run = args.routed_run or args.run or config.get("routed_run")
        if not baseline_run or not routed_run:
            raise SystemExit("difficulty_step_eval requires --baseline-run and --routed-run/--run")
        report = make_difficulty_report(
            baseline_run,
            routed_run,
            output_path=args.output or config.get("output_path"),
            sample_output_path=config.get("sample_output_path"),
            split=args.split or str(config.get("split", "val")),
            batch_size=args.batch_size or config.get("batch_size"),
            max_batches=args.max_batches or int(config.get("max_batches", 8)),
            device_name=str(config.get("device", "auto")),
            baseline_checkpoint=str(config.get("baseline_checkpoint", "checkpoint_best")),
            routed_checkpoint=str(config.get("routed_checkpoint", "checkpoint_best")),
        )
    else:
        if not args.run:
            raise SystemExit("routing eval requires --run")
        report = make_routing_report(args.run)
    print(report)


if __name__ == "__main__":
    main()
