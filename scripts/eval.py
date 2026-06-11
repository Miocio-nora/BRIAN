#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.compute_report import make_compute_report
from brian_sphere_llm.eval.cost_control_report import make_cost_control_report
from brian_sphere_llm.eval.determinism_report import make_eval_determinism_report
from brian_sphere_llm.eval.difficulty_report import make_baseline_difficulty_report, make_difficulty_report
from brian_sphere_llm.eval.fixed_route_stability import make_fixed_route_stability_report
from brian_sphere_llm.eval.global_kv_ablation import make_global_kv_ablation_report
from brian_sphere_llm.eval.global_kv_retention import make_global_kv_retention_report
from brian_sphere_llm.eval.go_no_go_report import make_go_no_go_report
from brian_sphere_llm.eval.long_context import make_long_context_report
from brian_sphere_llm.eval.long_context_compare import make_long_context_comparison_report
from brian_sphere_llm.eval.out_by_difficulty import make_out_by_difficulty_report
from brian_sphere_llm.eval.parallel_compare import make_parallel_comparison_report
from brian_sphere_llm.eval.position_ablation import make_position_ablation_report
from brian_sphere_llm.eval.pseudo_route_curriculum import make_pseudo_route_curriculum_report
from brian_sphere_llm.eval.reasoning import make_reasoning_report
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.eval.scheduled_routing import make_scheduled_routing_report
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
    parser.add_argument("--stage-gate-report", default=None, help="Stage-gate report path for decision reports.")
    parser.add_argument("--compute-report", default=None, help="Compute report path for decision reports.")
    parser.add_argument("--long-context-compare-report", default=None, help="Long-context comparison report path for stage gate eval.")
    parser.add_argument("--global-kv-retention-report", default=None, help="Global KV retention report path for stage gate eval.")
    parser.add_argument("--global-kv-ablation-report", default=None, help="Global KV ablation report path for decision reports.")
    parser.add_argument("--parallel-compare-report", default=None, help="Parallel comparison report path for stage gate eval.")
    parser.add_argument("--position-ablation-report", default=None, help="Position ablation report path for go/no-go eval.")
    parser.add_argument("--out-by-difficulty-report", default=None, help="OUT-by-difficulty report path for go/no-go eval.")
    parser.add_argument("--reasoning-report", default=None, help="Reasoning report path for OUT-by-difficulty eval.")
    parser.add_argument("--samples-path", default=None, help="Sample JSONL path for OUT-by-difficulty eval.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint name for model-based evals.")
    parser.add_argument("--sample-count", type=int, default=None, help="Sample count for synthetic evals.")
    parser.add_argument("--baseline-report", default=None, help="Baseline report path for comparison evals.")
    parser.add_argument("--reports", nargs="*", default=None, help="Candidate report paths for comparison evals.")
    parser.add_argument("--experiment-manifest", default=None, help="Experiment manifest path for ablation reports.")
    parser.add_argument("--tolerance", type=float, default=None, help="Numeric tolerance for determinism evals.")
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
            out_by_difficulty_report_path=args.out_by_difficulty_report or config.get("out_by_difficulty_report_path"),
            global_kv_retention_report_path=args.global_kv_retention_report
            or config.get("global_kv_retention_report_path"),
            long_context_compare_report_path=args.long_context_compare_report
            or config.get("long_context_compare_report_path"),
            parallel_compare_report_path=args.parallel_compare_report or config.get("parallel_compare_report_path"),
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
    elif eval_name == "baseline_difficulty_report":
        if not args.run:
            raise SystemExit("baseline_difficulty_report requires --run")
        report = make_baseline_difficulty_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            sample_output_path=config.get("sample_output_path"),
            split=args.split or str(config.get("split", "val")),
            batch_size=args.batch_size or config.get("batch_size"),
            max_batches=args.max_batches or int(config.get("max_batches", 8)),
            device_name=str(config.get("device", "auto")),
            checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_best")),
            difficulty_bins=list(config.get("difficulty_bins") or ["easy", "medium", "hard"]),
        )
    elif eval_name == "fixed_route_stability_report":
        if not args.run:
            raise SystemExit("fixed_route_stability_report requires --run")
        report = make_fixed_route_stability_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            split=args.split or str(config.get("split", "val")),
            batch_size=args.batch_size or config.get("batch_size"),
            max_batches=args.max_batches or int(config.get("max_batches", 8)),
            checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_best")),
            device_name=str(config.get("device", "auto")),
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
    elif eval_name == "eval_determinism_report":
        if not args.run:
            raise SystemExit("eval_determinism_report requires --run")
        report = make_eval_determinism_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            split=args.split or str(config.get("split", "val")),
            batch_size=args.batch_size or config.get("batch_size"),
            checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_best")),
            seed=int(config.get("seed", 1)),
            device_name=str(config.get("device", "auto")),
            tolerance=float(args.tolerance if args.tolerance is not None else config.get("tolerance", 1e-8)),
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
    elif eval_name == "global_kv_retention_report":
        if not args.run:
            raise SystemExit("global_kv_retention_report requires --run")
        report = make_global_kv_retention_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            min_global_attention_mass=float(config.get("min_global_attention_mass", 1e-6)),
            min_global_read_gate=float(config.get("min_global_read_gate", 1e-6)),
            mass_tolerance=float(config.get("mass_tolerance", 1e-5)),
            capacity_slack=float(config.get("capacity_slack", 1e-6)),
        )
    elif eval_name == "global_kv_ablation_report":
        manifest = args.experiment_manifest or config.get("experiment_manifest")
        runs = args.runs or ([args.run] if args.run else config.get("runs", []))
        if not manifest or not runs:
            raise SystemExit("global_kv_ablation_report requires --experiment-manifest and --runs")
        report = make_global_kv_ablation_report(
            manifest,
            runs,
            output_path=args.output or config.get("output_path"),
            long_context_report_paths=args.reports or config.get("long_context_report_paths", []),
        )
    elif eval_name == "parallel_compare":
        baseline_run = args.baseline_run or config.get("baseline_run")
        runs = args.runs or ([args.run] if args.run else config.get("runs", []))
        if not baseline_run or not runs:
            raise SystemExit("parallel_compare requires --baseline-run and --runs/--run")
        report = make_parallel_comparison_report(
            baseline_run,
            runs,
            output_path=args.output or config.get("output_path"),
            max_validation_loss_delta=float(config.get("max_validation_loss_delta", 0.0)),
            max_active_layer_eval_ratio=float(config.get("max_active_layer_eval_ratio", 2.0)),
            max_estimated_flops_ratio=float(config.get("max_estimated_flops_ratio", 2.0)),
            min_throughput_ratio=float(config.get("min_throughput_ratio", 0.0)),
            min_parallel_branch_count=float(config.get("min_parallel_branch_count", 1.5)),
        )
    elif eval_name == "position_ablation_report":
        reference_run = args.baseline_run or args.run or config.get("reference_run")
        runs = args.runs or config.get("candidate_runs", []) or config.get("runs", [])
        if not reference_run or not runs:
            raise SystemExit("position_ablation_report requires --run/--baseline-run and --runs")
        report = make_position_ablation_report(
            reference_run,
            runs,
            output_path=args.output or config.get("output_path"),
            min_validation_loss_delta=float(config.get("min_validation_loss_delta", 0.001)),
            min_routing_metric_delta=float(config.get("min_routing_metric_delta", 0.001)),
        )
    elif eval_name == "pseudo_route_curriculum_report":
        baseline_report = args.baseline_report or config.get("baseline_difficulty_report_path")
        if not args.run or not baseline_report:
            raise SystemExit("pseudo_route_curriculum_report requires --run and --baseline-report")
        report = make_pseudo_route_curriculum_report(
            args.run,
            baseline_difficulty_report_path=baseline_report,
            output_path=args.output or config.get("output_path"),
        )
    elif eval_name == "scheduled_routing_report":
        if not args.run:
            raise SystemExit("scheduled_routing_report requires --run")
        report = make_scheduled_routing_report(
            args.run,
            output_path=args.output or config.get("output_path"),
            min_final_router_probability=float(config.get("min_final_router_probability", 1.0)),
            tolerance=float(config.get("tolerance", 1e-9)),
        )
    elif eval_name == "out_by_difficulty_report":
        reasoning_report = args.reasoning_report or config.get("reasoning_report_path")
        samples_path = args.samples_path or config.get("samples_path")
        if not reasoning_report and not samples_path:
            raise SystemExit("out_by_difficulty_report requires --reasoning-report or --samples-path")
        report = make_out_by_difficulty_report(
            reasoning_report_path=reasoning_report,
            samples_path=samples_path,
            output_path=args.output or config.get("output_path"),
            difficulty_order=list(config.get("difficulty_order", ["easy", "medium", "hard"])),
            min_step_delta=float(config.get("min_step_delta", 0.0)),
            min_output_probability_delta=float(config.get("min_output_probability_delta", 0.0)),
        )
    elif eval_name == "go_no_go_report":
        stage_gate_report = args.stage_gate_report or config.get("stage_gate_report_path")
        if not stage_gate_report:
            raise SystemExit("go_no_go_report requires --stage-gate-report")
        report = make_go_no_go_report(
            stage_gate_report_path=stage_gate_report,
            output_path=args.output or config.get("output_path"),
            phase=str(config.get("phase", "all")),
            compute_report_path=args.compute_report or config.get("compute_report_path"),
            position_ablation_report_path=args.position_ablation_report or config.get("position_ablation_report_path"),
            out_by_difficulty_report_path=args.out_by_difficulty_report or config.get("out_by_difficulty_report_path"),
            reasoning_baseline_report_path=args.baseline_report or config.get("reasoning_baseline_report_path"),
            reasoning_candidate_report_paths=args.reports or config.get("reasoning_candidate_reports", []),
            long_context_compare_report_path=args.long_context_compare_report
            or config.get("long_context_compare_report_path"),
            global_kv_ablation_report_path=args.global_kv_ablation_report
            or config.get("global_kv_ablation_report_path"),
            parallel_compare_report_path=args.parallel_compare_report or config.get("parallel_compare_report_path"),
            min_difficulty_step_correlation=float(config.get("min_difficulty_step_correlation", 0.0)),
            min_reasoning_delta=float(config.get("min_reasoning_delta", 0.0)),
        )
    else:
        if not args.run:
            raise SystemExit("routing eval requires --run")
        report = make_routing_report(args.run)
    print(report)


if __name__ == "__main__":
    main()
