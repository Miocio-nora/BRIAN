from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.tokenize import load_tokenizer
from brian_sphere_llm.eval.difficulty_report import _checkpoint_step, _forward_routed_for_eval, _load_model_for_run
from brian_sphere_llm.eval.reasoning import _decode, _device, _mean, normalize_answer
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class LongContextSample:
    task_family: str
    difficulty: str
    prompt: str
    answer: str
    key: str


def make_long_context_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    sample_output_path: str | Path | None = None,
    sample_count: int = 12,
    seed: int = 1,
    checkpoint: str = "checkpoint_best",
    device_name: str = "auto",
    task_families: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for long-context eval.")
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    tokenizer = _load_tokenizer_from_run_config(config)
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    route_mode = train_mode_for_stage(str(config["stage"]))
    global_step = _checkpoint_step(run_dir, checkpoint)
    context_length = _context_length(config)
    samples = generate_long_context_samples(
        sample_count,
        seed=seed,
        context_length=context_length,
        task_families=task_families,
        difficulties=difficulties,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            rows.append(
                evaluate_long_context_sample(
                    model,
                    tokenizer,
                    sample,
                    config=config,
                    route_mode=route_mode,
                    global_step=global_step,
                    context_length=context_length,
                    sample_id=index,
                    device=device,
                )
            )

    if output_path is None:
        output_path = run_dir / "long_context_report.json"
    output_path = Path(output_path)
    if sample_output_path is None:
        sample_output_path = output_path.with_name(output_path.stem + "_samples.jsonl")
    sample_output_path = Path(sample_output_path)
    _write_jsonl(rows, sample_output_path)
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "sample_count": len(rows),
        "seed": seed,
        "context_length": context_length,
        "samples_path": str(sample_output_path),
        "overall": summarize_long_context_rows(rows),
        "by_task_family": _group_summary(rows, "task_family"),
        "by_difficulty": _group_summary(rows, "difficulty"),
        "routing": _routing_summary(rows),
        "global_kv": _global_kv_summary(rows),
    }
    write_json(report, output_path)
    return output_path


def generate_long_context_samples(
    count: int,
    *,
    seed: int = 1,
    context_length: int = 128,
    task_families: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> list[LongContextSample]:
    rng = random.Random(seed)
    families = task_families or ["needle_retrieval", "two_hop_tracing"]
    difficulty_values = difficulties or ["near", "middle", "far"]
    samples = []
    for index in range(count):
        family = families[index % len(families)]
        difficulty = difficulty_values[index % len(difficulty_values)]
        samples.append(_make_sample(rng, family, difficulty, context_length=context_length))
    return samples


def evaluate_long_context_sample(
    model: Any,
    tokenizer: Any,
    sample: LongContextSample,
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    context_length: int,
    sample_id: int,
    device: "torch.device",
) -> dict[str, Any]:
    prompt_ids = _prompt_ids(tokenizer, sample.prompt)
    answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)
    full_ids = prompt_ids + answer_ids
    truncated = len(full_ids) > context_length
    model_ids = full_ids[-context_length:]
    input_ids = torch.tensor([model_ids], dtype=torch.long, device=device)
    outputs = _forward_routed_for_eval(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
    prompt_len = max(0, min(len(prompt_ids), len(model_ids) - len(answer_ids)))
    start = max(0, prompt_len - 1)
    end = start + len(answer_ids)
    teacher_predictions = outputs["logits"][0, start:end].argmax(dim=-1).detach().cpu().tolist()
    teacher_accuracy = _token_accuracy(teacher_predictions, answer_ids)
    generated_ids = greedy_generate(
        model,
        prompt_ids[-context_length:],
        new_tokens=len(answer_ids),
        config=config,
        route_mode=route_mode,
        global_step=global_step,
        context_length=context_length,
        device=device,
    )
    generated_text = _decode(tokenizer, generated_ids)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "task_family": sample.task_family,
        "difficulty": sample.difficulty,
        "key": sample.key,
        "prompt": sample.prompt,
        "expected_answer": sample.answer,
        "generated_answer": generated_text,
        "exact_match": normalize_answer(generated_text) == normalize_answer(sample.answer),
        "teacher_forced_token_accuracy": teacher_accuracy,
        "truncated": truncated,
        "prompt_token_count": len(prompt_ids),
    }
    for key, value in outputs.get("routing_summary", {}).items():
        if isinstance(value, (int, float)):
            row[f"routing_{key}"] = float(value)
    return row


def greedy_generate(
    model: Any,
    prompt_ids: list[int],
    *,
    new_tokens: int,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    context_length: int,
    device: "torch.device",
) -> list[int]:
    current = list(prompt_ids)
    generated: list[int] = []
    for _ in range(new_tokens):
        input_ids = torch.tensor([current[-context_length:]], dtype=torch.long, device=device)
        outputs = _forward_routed_for_eval(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
        next_id = int(outputs["logits"][0, -1].argmax().detach().cpu())
        generated.append(next_id)
        current.append(next_id)
    return generated


def summarize_long_context_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "exact_match_accuracy": None, "teacher_forced_token_accuracy": None}
    return {
        "sample_count": len(rows),
        "exact_match_accuracy": sum(1 for row in rows if row["exact_match"]) / len(rows),
        "teacher_forced_token_accuracy": _mean([row.get("teacher_forced_token_accuracy") for row in rows]),
        "truncation_rate": sum(1 for row in rows if row.get("truncated")) / len(rows),
    }


def _make_sample(rng: random.Random, task_family: str, difficulty: str, *, context_length: int) -> LongContextSample:
    key = f"K{rng.randint(10, 99)}"
    value = str(rng.randint(100, 999))
    filler_count = max(2, min(10, context_length // 10))
    filler = [f"n{index}." for index in range(filler_count)]
    insert_at = {
        "far": max(0, filler_count // 5),
        "middle": filler_count // 2,
        "near": max(0, filler_count - 1),
    }[difficulty]
    if task_family == "needle_retrieval":
        parts = [*filler]
        parts.insert(insert_at, f"{key}={value}.")
        return LongContextSample(task_family, difficulty, "ctx " + " ".join(parts) + f" Q {key}? A:", " " + value, key)
    if task_family == "two_hop_tracing":
        link_key = f"L{rng.randint(10, 99)}"
        parts = [*filler]
        parts.insert(insert_at, f"{key}->{link_key}.")
        parts.insert(min(len(parts), insert_at + 2), f"{link_key}={value}.")
        return LongContextSample(task_family, difficulty, "ctx " + " ".join(parts) + f" Q {key}? A:", " " + value, key)
    raise ValueError(f"Unsupported long-context task family: {task_family}")


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos = getattr(tokenizer, "bos_token_id", None)
    return ([int(bos)] if bos is not None else []) + ids


def _load_tokenizer_from_run_config(config: dict[str, Any]) -> Any:
    data_config = config.get("data_config_resolved", {})
    tokenizer_config = data_config.get("tokenizer", {}) if isinstance(data_config, dict) else {}
    return load_tokenizer(
        str(tokenizer_config.get("name", "simple-byte-tokenizer")),
        revision=str(tokenizer_config.get("revision", "main")),
        local_files_only=bool(tokenizer_config.get("local_files_only", True)),
        fallback_to_byte=bool(tokenizer_config.get("fallback_to_byte", True)),
    )


def _context_length(config: dict[str, Any]) -> int:
    data_config = config.get("data_config_resolved", {})
    if isinstance(data_config, dict) and data_config.get("sequence_length"):
        return int(data_config["sequence_length"])
    model_config = config.get("model_config_resolved", {})
    if isinstance(model_config, dict) and model_config.get("context_length"):
        return int(model_config["context_length"])
    if isinstance(model_config, dict) and isinstance(model_config.get("base"), dict):
        return int(model_config["base"].get("context_length", 128))
    return 128


def _token_accuracy(predicted: list[int], expected: list[int]) -> float:
    total = max(1, len(expected))
    return sum(1 for left, right in zip(predicted, expected) if int(left) == int(right)) / total


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: summarize_long_context_rows(group_rows) for name, group_rows in sorted(groups.items())}


def _routing_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in rows for key in row if key.startswith("routing_")})
    return {key.removeprefix("routing_"): _mean([row.get(key) for row in rows]) for key in keys}


def _global_kv_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        "global_attention_mass": _mean([row.get("routing_global_attention_mass") for row in rows]),
        "global_read_gate_mean": _mean([row.get("routing_global_read_gate_mean") for row in rows]),
        "global_cache_slots_mean": _mean([row.get("routing_global_cache_slots_mean") for row in rows]),
    }


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
