#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from brian_sphere_llm.eval.difficulty_report import (
    _checkpoint_step,
    _device,
    _forward_routed_for_eval,
    _load_model_for_run,
)
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


LEGACY_TASKS = ("piqa", "hellaswag", "arc_easy")
EXPANDED_TASKS = (
    *LEGACY_TASKS,
    "arc_challenge",
    "openbookqa",
    "winogrande",
    "boolq",
    "commonsense_qa",
    "mmlu_elementary_mathematics",
    "mmlu_high_school_mathematics",
    "mmlu_college_mathematics",
    "mmlu_abstract_algebra",
    "mmlu_formal_logic",
)
# Keep this exported default stable: checkpoint diagnostics import TASKS and
# rely on it meaning the historical three-task public contract.
TASKS = LEGACY_TASKS
SUPPORTED_TASKS = EXPANDED_TASKS

_TASK_GROUPS = {
    "legacy_core": LEGACY_TASKS,
    "expanded_reasoning": (
        "arc_challenge",
        "openbookqa",
        "winogrande",
        "boolq",
        "commonsense_qa",
    ),
    "mmlu_math_logic": (
        "mmlu_elementary_mathematics",
        "mmlu_high_school_mathematics",
        "mmlu_college_mathematics",
        "mmlu_abstract_algebra",
        "mmlu_formal_logic",
    ),
}

_DATASET_REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
}


@dataclass(frozen=True)
class ChoiceRequest:
    row_index: int
    choice_index: int
    input_ids: tuple[int, ...]
    target_start: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small public multiple-choice benchmark.")
    parser.add_argument("--config", default=None, help="Optional benchmark YAML config.")
    parser.add_argument("--run", default=None, help="Run directory.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--samples-output", default=None, help="Output JSONL sample path.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, choices=SUPPORTED_TASKS)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--full-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate every labeled row in each selected task.",
    )
    parser.add_argument("--length-normalized", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for public benchmark eval.")
    config = load_config(args.config) if args.config else {}
    run_dir = args.run or config.get("run")
    output = args.output or config.get("output_path")
    if not run_dir or not output:
        raise SystemExit("public benchmark requires --run/--output or config run/output_path.")

    full_dataset = bool(
        args.full_dataset if args.full_dataset is not None else config.get("full_dataset", False)
    )
    report_path = run_public_benchmark(
        run_dir,
        output_path=output,
        samples_output_path=args.samples_output or config.get("samples_output_path"),
        checkpoint=str(args.checkpoint or config.get("checkpoint", "checkpoint_latest")),
        tasks=list(args.tasks or config.get("tasks", LEGACY_TASKS)),
        sample_count=(
            None
            if full_dataset
            else int(args.sample_count if args.sample_count is not None else config.get("sample_count", 50))
        ),
        seed=int(args.seed if args.seed is not None else config.get("seed", 1)),
        device_name=str(args.device or config.get("device", "auto")),
        batch_size=int(args.batch_size if args.batch_size is not None else config.get("batch_size", 1)),
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
    sample_count: int | None = 50,
    seed: int = 1,
    device_name: str = "auto",
    batch_size: int = 1,
    length_normalized: bool = True,
) -> Path:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if sample_count is not None and (
        isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer or None for the full split.")
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
    selected_tasks = list(tasks or LEGACY_TASKS)
    unknown_tasks = sorted(set(selected_tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {', '.join(unknown_tasks)}")
    rows: list[dict[str, Any]] = []
    requests: list[ChoiceRequest] = []
    task_contracts: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        examples, contract = _load_examples_with_contract(task, sample_count=sample_count, rng=rng)
        task_contracts[task] = contract
        for local_index, example in enumerate(examples):
            row_index = len(rows)
            scores = [-math.inf] * len(example["choices"])
            rows.append(
                {
                    "task": task,
                    "sample_id": local_index,
                    "prompt": example["prompt"],
                    "choices": example["choices"],
                    "label": example["label"],
                    "prediction": None,
                    "scores": scores,
                    "correct": False,
                    "source_index": example.get("source_index"),
                    "source_id": example.get("source_id"),
                }
            )
            for choice_index, choice in enumerate(example["choices"]):
                prepared = _prepare_choice(
                    tokenizer,
                    example["prompt"],
                    choice,
                    context_length=context_length,
                )
                if prepared is not None:
                    input_ids, target_start = prepared
                    requests.append(
                        ChoiceRequest(
                            row_index=row_index,
                            choice_index=choice_index,
                            input_ids=tuple(input_ids),
                            target_start=target_start,
                        )
                    )

    scoring_started = time.perf_counter()
    with torch.inference_mode():
        if batch_size == 1:
            request_scores = [
                _prepared_choice_score(
                    model,
                    request,
                    config=config,
                    route_mode=route_mode,
                    global_step=global_step,
                    device=device,
                    length_normalized=length_normalized,
                )
                for request in requests
            ]
        else:
            request_scores = _batched_choice_scores(
                model,
                requests,
                batch_size=batch_size,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                device=device,
                length_normalized=length_normalized,
            )
    scoring_seconds = time.perf_counter() - scoring_started
    for request, score in zip(requests, request_scores, strict=True):
        rows[request.row_index]["scores"][request.choice_index] = score
    for row in rows:
        scores = row["scores"]
        predicted = max(range(len(scores)), key=lambda index: scores[index])
        row["prediction"] = predicted
        row["correct"] = predicted == row["label"]

    output = Path(output_path)
    samples_output = (
        Path(samples_output_path)
        if samples_output_path
        else output.with_name(output.stem + "_samples.jsonl")
    )
    write_jsonl(rows, samples_output)
    by_task = {task: _summarize([row for row in rows if row["task"] == task]) for task in selected_tasks}
    task_accuracies = [summary["accuracy"] for summary in by_task.values() if summary["accuracy"] is not None]
    report = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "tasks": selected_tasks,
        "sample_count_per_task": sample_count,
        "full_dataset": sample_count is None,
        "seed": seed,
        "length_normalized": length_normalized,
        "inference": {
            "batch_size": batch_size,
            "batching": "exact_length" if batch_size > 1 else "reference",
            "score_equivalence": "numeric" if batch_size > 1 else "reference",
            "scoring_seconds": scoring_seconds,
        },
        "overall": _summarize(rows),
        "macro_task_accuracy": sum(task_accuracies) / len(task_accuracies) if task_accuracies else None,
        "by_task": by_task,
        "by_group": {
            group: _summarize([row for row in rows if row["task"] in group_tasks])
            for group, group_tasks in _TASK_GROUPS.items()
            if any(task in selected_tasks for task in group_tasks)
        },
        "task_contracts": task_contracts,
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
    prepared = _prepare_choice(tokenizer, prompt, choice, context_length=context_length)
    if prepared is None:
        return -math.inf
    input_ids, target_start = prepared
    return _prepared_choice_score(
        model,
        ChoiceRequest(0, 0, tuple(input_ids), target_start),
        config=config,
        route_mode=route_mode,
        global_step=global_step,
        device=device,
        length_normalized=length_normalized,
    )


def _prepare_choice(
    tokenizer: Any,
    prompt: str,
    choice: str,
    *,
    context_length: int,
) -> tuple[list[int], int] | None:
    bos = getattr(tokenizer, "bos_token_id", None)
    prompt_ids = ([int(bos)] if bos is not None else []) + tokenizer.encode(prompt, add_special_tokens=False)
    choice_ids = tokenizer.encode(choice, add_special_tokens=False)
    if not choice_ids:
        return None
    full_ids = prompt_ids + choice_ids
    overflow = max(0, len(full_ids) - context_length)
    full_ids = full_ids[overflow:]
    start = len(prompt_ids) - overflow
    if start <= 0:
        return None
    return full_ids, start


def _prepared_choice_score(
    model: Any,
    request: ChoiceRequest,
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    device: "torch.device",
    length_normalized: bool,
) -> float:
    input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=device)
    outputs = _forward_routed_for_eval(
        model,
        input_ids,
        config=config,
        route_mode=route_mode,
        global_step=global_step,
        summarize_routing=True,
    )
    logits = outputs["logits"][0]
    target = input_ids[0, request.target_start :]
    pred_logits = logits[request.target_start - 1 : input_ids.size(1) - 1]
    token_scores = F.log_softmax(pred_logits.float(), dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
    score = float(token_scores.sum().detach().cpu())
    if length_normalized:
        score /= max(1, int(target.numel()))
    return score


def _batched_choice_scores(
    model: Any,
    requests: list[ChoiceRequest],
    *,
    batch_size: int,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    device: "torch.device",
    length_normalized: bool,
) -> list[float]:
    scores = [float("nan")] * len(requests)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        groups[len(request.input_ids)].append(index)

    for sequence_length, indexes in groups.items():
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            batch_requests = [requests[index] for index in batch_indexes]
            if len(batch_requests) == 1:
                scores[batch_indexes[0]] = _prepared_choice_score(
                    model,
                    batch_requests[0],
                    config=config,
                    route_mode=route_mode,
                    global_step=global_step,
                    device=device,
                    length_normalized=length_normalized,
                )
                continue
            input_ids = torch.tensor(
                [request.input_ids for request in batch_requests],
                dtype=torch.long,
                device=device,
            )
            target_starts = torch.tensor(
                [request.target_start for request in batch_requests],
                dtype=torch.long,
                device=device,
            )
            outputs = _forward_routed_for_eval(
                model,
                input_ids,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                summarize_routing=False,
            )
            positions = torch.arange(sequence_length - 1, device=device).unsqueeze(0)
            target_mask = positions >= (target_starts - 1).unsqueeze(1)
            selected_logits = outputs["logits"][:, :-1][target_mask].float()
            selected_targets = input_ids[:, 1:][target_mask]
            token_scores = F.log_softmax(selected_logits, dim=-1).gather(
                1,
                selected_targets.unsqueeze(1),
            ).squeeze(1)
            sample_indexes = torch.arange(len(batch_requests), device=device).unsqueeze(1).expand_as(target_mask)
            selected_samples = sample_indexes[target_mask]
            batch_scores = torch.zeros(len(batch_requests), dtype=torch.float32, device=device)
            batch_scores.scatter_add_(0, selected_samples, token_scores)
            if length_normalized:
                batch_scores = batch_scores / target_mask.sum(dim=1).clamp_min(1)
            for request_index, score in zip(batch_indexes, batch_scores.detach().cpu().tolist(), strict=True):
                scores[request_index] = float(score)
    if any(math.isnan(score) for score in scores):
        raise RuntimeError("Batched public benchmark did not score every choice.")
    return scores


def _load_examples(task: str, *, sample_count: int | None, rng: random.Random) -> list[dict[str, Any]]:
    examples, _ = _load_examples_with_contract(task, sample_count=sample_count, rng=rng)
    return examples


def _load_examples_with_contract(
    task: str,
    *,
    sample_count: int | None,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task == "piqa":
        dataset = load_dataset("piqa", split="validation")
        rows = [_piqa(row) for row in dataset]
        dataset_name, dataset_config, split = "piqa", None, "validation"
    elif task == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        rows = [_hellaswag(row) for row in dataset]
        dataset_name, dataset_config, split = "hellaswag", None, "validation"
    elif task == "arc_easy":
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
        rows = [_arc_easy(row) for row in dataset]
        dataset_name, dataset_config, split = "ai2_arc", "ARC-Easy", "validation"
    elif task == "arc_challenge":
        dataset_name, dataset_config, split = "allenai/ai2_arc", "ARC-Challenge", "test"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_arc_easy(row) for row in dataset]
    elif task == "openbookqa":
        dataset_name, dataset_config, split = "allenai/openbookqa", "main", "test"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_openbookqa(row) for row in dataset]
    elif task == "winogrande":
        dataset_name, dataset_config, split = "allenai/winogrande", "winogrande_debiased", "validation"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_winogrande(row) for row in dataset]
    elif task == "boolq":
        dataset_name, dataset_config, split = "google/boolq", None, "validation"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_boolq(row) for row in dataset]
    elif task == "commonsense_qa":
        dataset_name, dataset_config, split = "tau/commonsense_qa", None, "validation"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_commonsense_qa(row) for row in dataset]
    elif task.startswith("mmlu_"):
        dataset_name, dataset_config, split = "cais/mmlu", task.removeprefix("mmlu_"), "test"
        dataset = _load_pinned_dataset(dataset_name, dataset_config, split)
        rows = [_mmlu(row) for row in dataset]
    else:
        raise ValueError(f"Unsupported task: {task}")
    for source_index, row in enumerate(rows):
        row["source_index"] = source_index
        row.setdefault("source_id", str(source_index))
    if sample_count is None or sample_count >= len(rows):
        selected = rows
    else:
        indexes = list(range(len(rows)))
        rng.shuffle(indexes)
        selected = [rows[index] for index in indexes[:sample_count]]
    contract = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "revision": _DATASET_REVISIONS.get(dataset_name),
        "split": split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "available_rows": len(rows),
        "selected_rows": len(selected),
        "selection_sha256": _selection_hash(selected),
    }
    return selected, contract


def _load_pinned_dataset(dataset_name: str, dataset_config: str | None, split: str) -> Any:
    kwargs = {"split": split, "revision": _DATASET_REVISIONS[dataset_name]}
    if dataset_config is None:
        return load_dataset(dataset_name, **kwargs)
    return load_dataset(dataset_name, dataset_config, **kwargs)


def _piqa(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": f"Question: {row['goal']}\nAnswer:",
        "choices": [" " + str(row["sol1"]), " " + str(row["sol2"])],
        "label": int(row["label"]),
        "source_id": row.get("id"),
    }


def _hellaswag(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": f"Context: {row['ctx']}\nEnding:",
        "choices": [" " + str(item) for item in row["endings"]],
        "label": int(row["label"]),
        "source_id": row.get("source_id", row.get("ind")),
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
        "source_id": row.get("id"),
    }


def _openbookqa(row: dict[str, Any]) -> dict[str, Any]:
    labels = [str(item) for item in row["choices"]["label"]]
    answer = str(row["answerKey"])
    return {
        "prompt": f"Question: {row['question_stem']}\nAnswer:",
        "choices": [" " + str(item) for item in row["choices"]["text"]],
        "label": labels.index(answer),
        "source_id": row.get("id"),
    }


def _winogrande(row: dict[str, Any]) -> dict[str, Any]:
    prefix, suffix = str(row["sentence"]).split("_", maxsplit=1)
    return {
        "prompt": prefix,
        "choices": [str(row["option1"]) + suffix, str(row["option2"]) + suffix],
        "label": int(row["answer"]) - 1,
    }


def _boolq(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("answer", row.get("label"))
    return {
        "prompt": f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer:",
        "choices": [" no", " yes"],
        "label": int(bool(label)),
        "source_id": row.get("idx"),
    }


def _commonsense_qa(row: dict[str, Any]) -> dict[str, Any]:
    labels = [str(item) for item in row["choices"]["label"]]
    answer = str(row["answerKey"])
    return {
        "prompt": f"Question: {row['question']}\nAnswer:",
        "choices": [" " + str(item) for item in row["choices"]["text"]],
        "label": labels.index(answer),
        "source_id": row.get("id"),
    }


def _mmlu(row: dict[str, Any]) -> dict[str, Any]:
    choices = [str(item) for item in row["choices"]]
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    return {
        "prompt": f"Question: {row['question']}\n{rendered}\nAnswer:",
        "choices": [" " + label for label in labels],
        "label": int(row["answer"]),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "accuracy": None}
    correct = sum(1 for row in rows if row["correct"])
    low, high = _wilson_interval(correct, len(rows))
    return {
        "sample_count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_ci95_low": low,
        "accuracy_ci95_high": high,
    }


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _selection_hash(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "source_index": row.get("source_index"),
            "source_id": row.get("source_id"),
            "prompt": row["prompt"],
            "choices": row["choices"],
            "label": row["label"],
        }
        for row in rows
    ]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context_length(config: dict[str, Any]) -> int:
    model_config = config.get("model_config_resolved", {})
    base = model_config.get("base") if isinstance(model_config.get("base"), dict) else {}
    return int(model_config.get("context_length") or base.get("context_length") or 2048)


if __name__ == "__main__":
    main()
