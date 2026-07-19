from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from brian_sphere_llm.eval.difficulty_report import _checkpoint_step, _device, _load_model_for_run
from brian_sphere_llm.eval.reasoning import (
    _context_length,
    _load_tokenizer_from_run_config,
    _prompt_ids,
    batched_greedy_generate,
)
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json, write_jsonl

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


TASKS = ("gsm8k", "math500")

_DATASETS = {
    "gsm8k": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
    },
    "math500": {
        "name": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    },
}

_DEFAULT_MAX_NEW_TOKENS = {"gsm8k": 128, "math500": 256}


@dataclass(frozen=True)
class MathSample:
    task: str
    source_index: int
    source_id: str
    prompt: str
    answer: str
    subject: str | None = None
    level: str | None = None


def make_classic_math_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    samples_output_path: str | Path | None = None,
    checkpoint: str = "checkpoint_latest",
    tasks: list[str] | None = None,
    sample_counts: dict[str, int | None] | None = None,
    max_new_tokens: dict[str, int] | None = None,
    seed: int = 1,
    device_name: str = "auto",
    batch_size: int = 16,
    prompt_style: str = "zero_shot_cot",
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for classic math evaluation.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if prompt_style not in {"zero_shot_cot", "zero_shot_direct"}:
        raise ValueError("prompt_style must be 'zero_shot_cot' or 'zero_shot_direct'.")

    selected_tasks = list(tasks or TASKS)
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unsupported classic math tasks: {', '.join(unknown)}")
    limits = dict(sample_counts or {})
    generation_limits = {**_DEFAULT_MAX_NEW_TOKENS, **(max_new_tokens or {})}
    for task in selected_tasks:
        _validate_optional_count(limits.get(task), f"sample_counts.{task}")
        _validate_positive_int(generation_limits[task], f"max_new_tokens.{task}")

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    tokenizer = _load_tokenizer_from_run_config(config)
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    route_mode = train_mode_for_stage(str(config["stage"]))
    global_step = _checkpoint_step(run_dir, checkpoint)
    context_length = _context_length(config)

    samples: list[MathSample] = []
    contracts: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        task_samples, contract = load_math_samples(task, sample_count=limits.get(task), seed=seed)
        samples.extend(task_samples)
        contracts[task] = contract

    prompt_token_ids: list[list[int]] = []
    token_budgets: list[int] = []
    for sample in samples:
        budget = generation_limits[sample.task]
        prompt = _format_prompt(sample, prompt_style=prompt_style)
        prompt_ids = _prompt_ids(tokenizer, prompt)
        prompt_token_ids.append(prompt_ids[-max(1, context_length - budget) :])
        token_budgets.append(budget)

    started = time.perf_counter()
    with torch.inference_mode():
        generated = batched_greedy_generate(
            model,
            prompt_token_ids,
            new_token_counts=token_budgets,
            batch_size=batch_size,
            config=config,
            route_mode=route_mode,
            global_step=global_step,
            context_length=context_length,
            device=device,
        )
    generation_seconds = time.perf_counter() - started

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    rows: list[dict[str, Any]] = []
    for sample, generated_ids in zip(samples, generated, strict=True):
        truncated_ids = _truncate_at_token(generated_ids, eos_token_id)
        prediction = _decode(tokenizer, truncated_ids)
        parsed, correct, parsed_repr = grade_math_answer(sample.answer, prediction)
        rows.append(
            {
                "task": sample.task,
                "source_index": sample.source_index,
                "source_id": sample.source_id,
                "subject": sample.subject,
                "level": sample.level,
                "prompt": sample.prompt,
                "answer": sample.answer,
                "prediction": prediction,
                "parsed": parsed,
                "parsed_prediction": parsed_repr,
                "correct": correct,
                "generated_token_count": len(truncated_ids),
            }
        )

    output = Path(output_path or run_dir / "classic_math_report.json")
    samples_output = Path(samples_output_path or output.with_name(output.stem + "_samples.jsonl"))
    write_jsonl(rows, samples_output)
    by_task = {task: _summarize([row for row in rows if row["task"] == task]) for task in selected_tasks}
    task_accuracies = [row["accuracy"] for row in by_task.values() if row["accuracy"] is not None]
    report = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "tasks": selected_tasks,
        "seed": seed,
        "protocol": {
            "prompt_style": prompt_style,
            "decoding": "greedy_temperature_0",
            "generation_batch_size": batch_size,
            "max_new_tokens": {task: generation_limits[task] for task in selected_tasks},
            "stop_policy": "truncate_at_eos_after_fixed_budget_generation",
            "grader": "math-verify-0.9.0",
            "leaderboard_comparable": False,
        },
        "inference": {
            "route_mode": route_mode,
            "context_length": context_length,
            "generation_seconds": generation_seconds,
            "samples_per_second": len(rows) / generation_seconds if generation_seconds > 0 else None,
        },
        "overall": _summarize(rows),
        "macro_task_accuracy": sum(task_accuracies) / len(task_accuracies) if task_accuracies else None,
        "by_task": by_task,
        "math500_by_subject": _group_summary(rows, "subject", task="math500"),
        "math500_by_level": _group_summary(rows, "level", task="math500"),
        "dataset_contracts": contracts,
        "samples_path": str(samples_output),
    }
    write_json(report, output)
    return output


def load_math_samples(
    task: str,
    *,
    sample_count: int | None,
    seed: int,
) -> tuple[list[MathSample], dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unsupported classic math task: {task}")
    _validate_optional_count(sample_count, "sample_count")
    spec = _DATASETS[task]
    kwargs = {"split": spec["split"], "revision": spec["revision"]}
    dataset = (
        load_dataset(spec["name"], spec["config"], **kwargs)
        if spec["config"] is not None
        else load_dataset(spec["name"], **kwargs)
    )
    rows = [_math_sample(task, index, dict(row)) for index, row in enumerate(dataset)]
    if sample_count is None or sample_count >= len(rows):
        selected = rows
    else:
        indexes = list(range(len(rows)))
        random.Random(_task_seed(seed, task)).shuffle(indexes)
        selected = [rows[index] for index in indexes[:sample_count]]
    contract = {
        "dataset": spec["name"],
        "dataset_config": spec["config"],
        "revision": spec["revision"],
        "split": spec["split"],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "available_rows": len(rows),
        "selected_rows": len(selected),
        "selection_sha256": _selection_hash(selected),
    }
    return selected, contract


def grade_math_answer(gold: str, prediction: str) -> tuple[bool, bool, str | None]:
    try:
        from math_verify import parse, verify
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "classic math evaluation requires `pip install 'math-verify[antlr4_13_2]==0.9.0'`."
        ) from exc
    gold_parsed = parse(f"${gold}$")
    prediction_parsed = parse(prediction)
    parsed = bool(prediction_parsed)
    correct = bool(parsed and gold_parsed and verify(gold_parsed, prediction_parsed))
    return parsed, correct, repr(prediction_parsed) if prediction_parsed else None


def _math_sample(task: str, source_index: int, row: dict[str, Any]) -> MathSample:
    if task == "gsm8k":
        answer = str(row["answer"]).rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
        return MathSample(
            task=task,
            source_index=source_index,
            source_id=str(source_index),
            prompt=str(row["question"]),
            answer=answer,
        )
    if task == "math500":
        return MathSample(
            task=task,
            source_index=source_index,
            source_id=str(row["unique_id"]),
            prompt=str(row["problem"]),
            answer=str(row["answer"]),
            subject=str(row["subject"]),
            level=str(row["level"]),
        )
    raise ValueError(f"Unsupported classic math task: {task}")


def _format_prompt(sample: MathSample, *, prompt_style: str) -> str:
    prefix = "Question" if sample.task == "gsm8k" else "Problem"
    if prompt_style == "zero_shot_direct":
        return f"{prefix}: {sample.prompt}\nGive only the final answer.\nFinal answer:"
    return (
        f"{prefix}: {sample.prompt}\n"
        "Solve the problem step by step. Finish with `Final answer is ...`.\n"
        "Solution:"
    )


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(token_ids, skip_special_tokens=True))
    byte_values = bytes(token for token in token_ids if 0 <= token < 256)
    return byte_values.decode("utf-8", errors="replace")


def _truncate_at_token(token_ids: list[int], stop_token_id: int | None) -> list[int]:
    if stop_token_id is None:
        return list(token_ids)
    try:
        index = token_ids.index(int(stop_token_id))
    except ValueError:
        return list(token_ids)
    return list(token_ids[:index])


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "correct": 0, "accuracy": None, "parse_rate": None}
    correct = sum(bool(row["correct"]) for row in rows)
    parsed = sum(bool(row["parsed"]) for row in rows)
    return {
        "sample_count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "parsed": parsed,
        "parse_rate": parsed / len(rows),
    }


def _group_summary(rows: list[dict[str, Any]], key: str, *, task: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows if row["task"] == task and row.get(key) is not None})
    return {
        value: _summarize([row for row in rows if row["task"] == task and str(row.get(key)) == value])
        for value in values
    }


def _selection_hash(samples: list[MathSample]) -> str:
    payload = [
        {
            "task": sample.task,
            "source_index": sample.source_index,
            "source_id": sample.source_id,
            "prompt": sample.prompt,
            "answer": sample.answer,
        }
        for sample in samples
    ]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_optional_count(value: int | None, name: str) -> None:
    if value is None:
        return
    _validate_positive_int(value, name)


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
