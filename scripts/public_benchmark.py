#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from brian_sphere_llm.eval.difficulty_report import _checkpoint_step, _device, _forward_routed_for_eval, _load_model_for_run
from brian_sphere_llm.eval.reasoning import _load_tokenizer_from_run_config
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json, write_jsonl

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


TASKS = ("piqa", "hellaswag", "arc_easy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small public multiple-choice benchmark.")
    parser.add_argument("--config", default=None, help="Optional benchmark YAML config.")
    parser.add_argument("--run", default=None, help="Run directory.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--samples-output", default=None, help="Output JSONL sample path.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, choices=TASKS)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--length-normalized", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for public benchmark eval.")
    config = load_config(args.config) if args.config else {}
    run_dir = args.run or config.get("run")
    output = args.output or config.get("output_path")
    if not run_dir or not output:
        raise SystemExit("public benchmark requires --run/--output or config run/output_path.")

    report_path = run_public_benchmark(
        run_dir,
        output_path=output,
        samples_output_path=args.samples_output or config.get("samples_output_path"),
        checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_latest")),
        tasks=list(args.tasks or config.get("tasks", TASKS)),
        sample_count=int(args.sample_count if args.sample_count is not None else config.get("sample_count", 50)),
        seed=int(args.seed if args.seed is not None else config.get("seed", 1)),
        device_name=str(args.device or config.get("device", "auto")),
        length_normalized=bool(
            args.length_normalized if args.length_normalized is not None else config.get("length_normalized", True)
        ),
    )
    print(report_path)


def run_public_benchmark(
    run_dir: str | Path,
    *,
    output_path: str | Path,
    samples_output_path: str | Path | None = None,
    checkpoint: str = "checkpoint_latest",
    tasks: list[str] | None = None,
    sample_count: int = 50,
    seed: int = 1,
    device_name: str = "auto",
    length_normalized: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    tokenizer = _load_tokenizer_from_run_config(config)
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    route_mode = train_mode_for_stage(str(config["stage"]))
    global_step = _checkpoint_step(run_dir, checkpoint)
    context_length = _context_length(config)

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for task in tasks or list(TASKS):
            examples = _load_examples(task, sample_count=sample_count, rng=rng)
            for local_index, example in enumerate(examples):
                scores = [
                    _choice_score(
                        model,
                        tokenizer,
                        example["prompt"],
                        choice,
                        config=config,
                        route_mode=route_mode,
                        global_step=global_step,
                        context_length=context_length,
                        device=device,
                        length_normalized=length_normalized,
                    )
                    for choice in example["choices"]
                ]
                predicted = max(range(len(scores)), key=lambda index: scores[index])
                rows.append(
                    {
                        "task": task,
                        "sample_id": local_index,
                        "prompt": example["prompt"],
                        "choices": example["choices"],
                        "label": example["label"],
                        "prediction": predicted,
                        "scores": scores,
                        "correct": predicted == example["label"],
                    }
                )

    output = Path(output_path)
    samples_output = Path(samples_output_path) if samples_output_path else output.with_name(output.stem + "_samples.jsonl")
    write_jsonl(rows, samples_output)
    report = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "tasks": list(tasks or TASKS),
        "sample_count_per_task": sample_count,
        "seed": seed,
        "length_normalized": length_normalized,
        "overall": _summarize(rows),
        "by_task": {task: _summarize([row for row in rows if row["task"] == task]) for task in tasks or TASKS},
        "samples_path": str(samples_output),
    }
    write_json(report, output)
    return output


def _choice_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    choice: str,
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    context_length: int,
    device: "torch.device",
    length_normalized: bool,
) -> float:
    bos = getattr(tokenizer, "bos_token_id", None)
    prompt_ids = ([int(bos)] if bos is not None else []) + tokenizer.encode(prompt, add_special_tokens=False)
    choice_ids = tokenizer.encode(choice, add_special_tokens=False)
    if not choice_ids:
        return -math.inf
    full_ids = prompt_ids + choice_ids
    overflow = max(0, len(full_ids) - context_length)
    full_ids = full_ids[overflow:]
    start = len(prompt_ids) - overflow
    if start <= 0:
        return -math.inf
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    outputs = _forward_routed_for_eval(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
    logits = outputs["logits"][0]
    target = input_ids[0, start:]
    pred_logits = logits[start - 1 : input_ids.size(1) - 1]
    token_scores = F.log_softmax(pred_logits.float(), dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
    score = float(token_scores.sum().detach().cpu())
    if length_normalized:
        score /= max(1, int(target.numel()))
    return score


def _load_examples(task: str, *, sample_count: int, rng: random.Random) -> list[dict[str, Any]]:
    if task == "piqa":
        dataset = load_dataset("piqa", split="validation")
        rows = [_piqa(row) for row in dataset]
    elif task == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        rows = [_hellaswag(row) for row in dataset]
    elif task == "arc_easy":
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
        rows = [_arc_easy(row) for row in dataset]
    else:
        raise ValueError(f"Unsupported task: {task}")
    if sample_count >= len(rows):
        return rows
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    return [rows[index] for index in indexes[:sample_count]]


def _piqa(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": f"Question: {row['goal']}\nAnswer:",
        "choices": [" " + str(row["sol1"]), " " + str(row["sol2"])],
        "label": int(row["label"]),
    }


def _hellaswag(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": f"Context: {row['ctx']}\nEnding:",
        "choices": [" " + str(item) for item in row["endings"]],
        "label": int(row["label"]),
    }


def _arc_easy(row: dict[str, Any]) -> dict[str, Any]:
    labels = [str(item) for item in row["choices"]["label"]]
    texts = [str(item) for item in row["choices"]["text"]]
    answer = str(row["answerKey"])
    if answer in labels:
        label = labels.index(answer)
    else:
        label = labels.index(str(int(answer) + 1)) if answer.isdigit() and str(int(answer) + 1) in labels else 0
    return {
        "prompt": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + text for text in texts],
        "label": label,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "accuracy": None}
    return {
        "sample_count": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows),
    }


def _context_length(config: dict[str, Any]) -> int:
    model_config = config.get("model_config_resolved", {})
    base = model_config.get("base") if isinstance(model_config.get("base"), dict) else {}
    return int(model_config.get("context_length") or base.get("context_length") or 2048)


if __name__ == "__main__":
    main()
