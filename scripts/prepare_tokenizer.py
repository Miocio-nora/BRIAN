#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.data.tokenize import load_tokenizer, tokenizer_metadata
from brian_sphere_llm.utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or inspect a tokenizer.")
    parser.add_argument("--name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--output", default="data/tokenized/tokenizer_metadata.json")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.name, revision=args.revision)
    write_json(asdict(tokenizer_metadata(tokenizer, name=args.name, revision=args.revision, license=args.license)), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
