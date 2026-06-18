#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.eval.post_train_benchmarks import run_post_train_benchmarks
from brian_sphere_llm.train.trainer import train_from_config
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils import distributed as dist_utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BRIAN-Sphere stage.")
    parser.add_argument("--config", required=True, help="Path to a train YAML config.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = train_from_config(args.config)
    if dist_utils.is_main_process():
        run_post_train_benchmarks(run_dir, config, project_root=Path(__file__).resolve().parents[1])
        print(run_dir)


if __name__ == "__main__":
    main()
