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
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = args.output or config.get("output_path")
    report = make_stage_gate_report(args.runs, output_path=output_path, thresholds=config.get("thresholds", {}))
    print(report)


if __name__ == "__main__":
    main()
