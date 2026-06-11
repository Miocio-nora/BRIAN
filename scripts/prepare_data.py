#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.data.prepare import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized BRIAN-Sphere data.")
    parser.add_argument("--config", required=True, help="Path to a data YAML config.")
    args = parser.parse_args()
    output_dir = prepare_data(args.config)
    print(output_dir)


if __name__ == "__main__":
    main()
