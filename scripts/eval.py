#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.routing_report import make_routing_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BRIAN-Sphere run.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--run", required=True, help="Run directory.")
    args = parser.parse_args()
    report = make_routing_report(args.run)
    print(report)


if __name__ == "__main__":
    main()
