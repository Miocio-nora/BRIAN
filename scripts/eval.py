#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.compute_report import make_compute_report
from brian_sphere_llm.eval.cost_control_report import make_cost_control_report
from brian_sphere_llm.eval.difficulty_report import make_difficulty_report
from brian_sphere_llm.eval.long_context import make_long_context_report
from brian_sphere_llm.eval.long_context_compare import make_long_context_comparison_report
from brian_sphere_llm.eval.reasoning import make_reasoning_report
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
    parser.add_argument("--tflops-per-gpu", type=float, default=None, help="Reference TFLOPs/GPU for compute reports.")
    parser.add_argument("--utilization", type=float, default=None, help="Reference utilization for compute reports.")
    parser.add_argument("--min-active-compute-range", type=float, default=None, help="Minimum active compute range for cost reports.")
    parser.add_argument("--cost-control-report", default=None, help="Cost-control report path for stage gate eval.")
    parser.add_argument("--long-context-compare-report", default=None, help="Long-context comparison report path for stage gate eval.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint name for model-based evals.")
    parser.add_argument("--sample-count", type=int, default=None, help="Sample count for synthetic evals.")
    parser.add_argument("--baseline-report", default=None, help="Baseline report path for comparison evals.")
    parser.add_argument("--reports", nargs="*", default=None, help="Candidate report paths for comparison evals.")
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
            cost_control_report_path=args.cost_control_report or config.get("cost_control_report_path"),
            long_context_compare_report_path=args.long_context_compare_report
            or config.get("long_context_compare_report_path"),
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
    elif eval_name == "compute_report":
        runs = args.runs or ([args.run] if args.run else config.get("runs", []))
        if not runs:
            raise SystemExit("compute_report requires --runs or --run")
        report = make_compute_report(
            runs,
            baseline_run=args.baseline_run or config.get("baseline_run"),
            output_path=args.output or config.get("output_path"),
            tflops_per_gpu=float(args.tflops_per_gpu or config.get("tflops_per_gpu", 989.0)),
            utilization=float(args.utilization or config.get("utilization", 0.35)),
        )
    elif eval_name == "cost_control_report":
        runs = args.runs or ([args.run] if args.run else config.get("runs", []))
        if not runs:
            raise SystemExit("cost_control_report requires --runs or --run")
        report = make_cost_control_report(
            runs,
            output_path=args.output or config.get("output_path"),
            min_active_compute_range=float(args.min_active_compute_range or config.get("min_active_compute_range", 0.05)),
        )
    elif eval_name == "reasoning_eval":
        if not args.run:
            raise SystemExit("reasoning_eval requires --run")
        report = make_reasoning_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            sample_count=int(args.sample_count or config.get("sample_count", 24)),
            seed=int(config.get("seed", 1)),
            checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_best")),
            device_name=str(config.get("device", "auto")),
            task_families=list(config.get("task_families", ["copy", "reverse", "arithmetic", "rewrite"])),
            difficulties=list(config.get("difficulties", ["easy", "medium", "hard"])),
        )
    elif eval_name == "long_context_eval":
        if not args.run:
            raise SystemExit("long_context_eval requires --run")
        report = make_long_context_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            sample_count=int(args.sample_count or config.get("sample_count", 12)),
            seed=int(config.get("seed", 1)),
            checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_best")),
            device_name=str(config.get("device", "auto")),
            task_families=list(config.get("task_families", ["needle_retrieval", "two_hop_tracing"])),
            difficulties=list(config.get("difficulties", ["near", "middle", "far"])),
        )
    elif eval_name == "long_context_compare":
        baseline_report = args.baseline_report or config.get("baseline_report")
        candidate_reports = args.reports or config.get("candidate_reports", [])
        if not baseline_report or not candidate_reports:
            raise SystemExit("long_context_compare requires --baseline-report and --reports")
        report = make_long_context_comparison_report(
            baseline_report,
            candidate_reports,
            output_path=args.output or config.get("output_path"),
            min_global_attention_mass=float(config.get("min_global_attention_mass", 1e-6)),
            min_global_read_gate=float(config.get("min_global_read_gate", 1e-6)),
            quality_tolerance=float(config.get("quality_tolerance", 0.0)),
        )
    else:
        if not args.run:
            raise SystemExit("routing eval requires --run")
        report = make_routing_report(args.run)
    print(report)


if __name__ == "__main__":
    main()
