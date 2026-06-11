#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.routing_report import make_routing_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build routing_report.json for a run directory.")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(make_routing_report(args.run))


if __name__ == "__main__":
    main()
