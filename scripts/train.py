#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.train.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BRIAN-Sphere stage.")
    parser.add_argument("--config", required=True, help="Path to a train YAML config.")
    args = parser.parse_args()
    run_dir = train_from_config(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
