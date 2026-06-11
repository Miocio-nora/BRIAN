#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BRIAN-Sphere experiment manifest.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML manifest.")
    parser.add_argument("--output-dir", default=None, help="Directory for experiment summary reports.")
    parser.add_argument("--include-baseline", action="store_true", help="Run the manifest baseline before ablations.")
    parser.add_argument("--baseline-run", default=None, help="Existing baseline run directory for compute comparisons.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved experiment plan without training.")
    args = parser.parse_args()
    report = run_experiment(
        args.config,
        output_dir=args.output_dir,
        include_baseline=args.include_baseline,
        baseline_run=args.baseline_run,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(report)


if __name__ == "__main__":
    main()
