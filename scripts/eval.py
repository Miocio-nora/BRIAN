#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report
from brian_sphere_llm.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BRIAN-Sphere run.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--run", default=None, help="Run directory.")
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories for stage gate reports.")
    parser.add_argument("--output", default=None, help="Optional output path override.")
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
    else:
        if not args.run:
            raise SystemExit("routing eval requires --run")
        report = make_routing_report(args.run)
    print(report)


if __name__ == "__main__":
    main()
