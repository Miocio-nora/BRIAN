#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report
from brian_sphere_llm.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage_gate_report.json across stage run directories.")
    parser.add_argument("--config", default="configs/eval/stage_gate_eval.yaml")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories in any order.")
    parser.add_argument("--output", default=None, help="Optional output path override.")
    parser.add_argument("--cost-control-report", default=None, help="Optional cost-control report path.")
    parser.add_argument("--out-by-difficulty-report", default=None, help="Optional OUT-by-difficulty report path.")
    parser.add_argument("--global-kv-retention-report", default=None, help="Optional Global KV retention report path.")
    parser.add_argument("--long-context-compare-report", default=None, help="Optional long-context comparison report path.")
    parser.add_argument("--parallel-passing-report", default=None, help="Optional parallel-passing safety report path.")
    parser.add_argument("--parallel-compare-report", default=None, help="Optional parallel comparison report path.")
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = args.output or config.get("output_path")
    report = make_stage_gate_report(
        args.runs,
        output_path=output_path,
        thresholds=config.get("thresholds", {}),
        cost_control_report_path=args.cost_control_report or config.get("cost_control_report_path"),
        out_by_difficulty_report_path=args.out_by_difficulty_report or config.get("out_by_difficulty_report_path"),
        global_kv_retention_report_path=args.global_kv_retention_report
        or config.get("global_kv_retention_report_path"),
        long_context_compare_report_path=args.long_context_compare_report
        or config.get("long_context_compare_report_path"),
        parallel_passing_report_path=args.parallel_passing_report
        or config.get("parallel_passing_report_path"),
        parallel_compare_report_path=args.parallel_compare_report or config.get("parallel_compare_report_path"),
    )
    print(report)


if __name__ == "__main__":
    main()
