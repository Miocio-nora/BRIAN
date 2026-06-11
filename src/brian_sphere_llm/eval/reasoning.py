from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.tokenize import load_tokenizer
from brian_sphere_llm.eval.difficulty_report import _checkpoint_step, _forward_routed_for_eval, _load_model_for_run
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class ReasoningSample:
    task_family: str
    difficulty: str
    prompt: str
    answer: str


def make_reasoning_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    sample_output_path: str | Path | None = None,
    sample_count: int = 24,
    seed: int = 1,
    checkpoint: str = "checkpoint_best",
    device_name: str = "auto",
    task_families: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for reasoning eval.")
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    tokenizer = _load_tokenizer_from_run_config(config)
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    route_mode = train_mode_for_stage(str(config["stage"]))
    global_step = _checkpoint_step(run_dir, checkpoint)
    context_length = _context_length(config)
    samples = list(generate_reasoning_samples(sample_count, seed=seed, task_families=task_families, difficulties=difficulties))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            row = evaluate_reasoning_sample(
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
            rows.append(row)

    if output_path is None:
        output_path = run_dir / "reasoning_report.json"
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
        "overall": summarize_reasoning_rows(rows),
        "by_task_family": _group_summary(rows, "task_family"),
        "by_difficulty": _group_summary(rows, "difficulty"),
        "routing": _routing_summary(rows),
    }
    write_json(report, output_path)
    return output_path


def generate_reasoning_samples(
    count: int,
    *,
    seed: int = 1,
    task_families: list[str] | None = None,
    difficulties: list[str] | None = None,
) -> list[ReasoningSample]:
    rng = random.Random(seed)
    families = task_families or ["copy", "reverse", "transform", "arithmetic", "rewrite"]
    difficulty_values = difficulties or ["easy", "medium", "hard"]
    samples = []
    for index in range(count):
        family = families[index % len(families)]
        difficulty = difficulty_values[index % len(difficulty_values)]
        samples.append(_make_sample(rng, family, difficulty))
    return samples


def evaluate_reasoning_sample(
    model: Any,
    tokenizer: Any,
    sample: ReasoningSample,
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
    if not answer_ids:
        raise ValueError("Reasoning sample produced an empty answer.")
    full_ids = (prompt_ids + answer_ids)[-context_length:]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    outputs = _forward_routed_for_eval(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
    logits = outputs["logits"]
    prompt_len = min(len(prompt_ids), len(full_ids) - len(answer_ids))
    start = max(0, prompt_len - 1)
    end = start + len(answer_ids)
    teacher_predictions = logits[0, start:end].argmax(dim=-1).detach().cpu().tolist()
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
    exact_match = normalize_answer(generated_text) == normalize_answer(sample.answer)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "task_family": sample.task_family,
        "difficulty": sample.difficulty,
        "prompt": sample.prompt,
        "expected_answer": sample.answer,
        "generated_answer": generated_text,
        "exact_match": bool(exact_match),
        "teacher_forced_token_accuracy": teacher_accuracy,
        "generated_token_count": len(generated_ids),
        "answer_token_count": len(answer_ids),
        "visible_cot_tokens": _visible_cot_token_count(generated_ids, answer_ids),
    }
    for key, value in outputs.get("routing_summary", {}).items():
        number = _num(value)
        if number is not None:
            row[f"routing_{key}"] = number
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
    generated: list[int] = []
    current = list(prompt_ids)
    for _ in range(new_tokens):
        window = current[-context_length:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        outputs = _forward_routed_for_eval(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
        next_id = int(outputs["logits"][0, -1].argmax().detach().cpu())
        generated.append(next_id)
        current.append(next_id)
    return generated


def summarize_reasoning_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "exact_match_accuracy": None, "teacher_forced_token_accuracy": None}
    return {
        "sample_count": len(rows),
        "exact_match_accuracy": sum(1 for row in rows if row["exact_match"]) / len(rows),
        "teacher_forced_token_accuracy": _mean([row.get("teacher_forced_token_accuracy") for row in rows]),
        "generated_tokens_mean": _mean([row.get("generated_token_count") for row in rows]),
        "answer_tokens_mean": _mean([row.get("answer_token_count") for row in rows]),
        "visible_cot_tokens_mean": _mean([row.get("visible_cot_tokens") for row in rows]),
    }


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().split())


def _make_sample(rng: random.Random, task_family: str, difficulty: str) -> ReasoningSample:
    if task_family in {"copy", "reverse", "transform"}:
        length = {"easy": 4, "medium": 8, "hard": 16}[difficulty]
        values = [str(rng.randint(0, 9)) for _ in range(length)]
        if task_family == "copy":
            answer_values = values
        elif task_family == "reverse":
            answer_values = list(reversed(values))
        else:
            answer_values = [str((int(value) + 1) % 10) for value in values]
        return ReasoningSample(task_family, difficulty, f"{task_family}: {' '.join(values)} ->", " " + " ".join(answer_values))
    if task_family == "arithmetic":
        terms = {"easy": 2, "medium": 4, "hard": 8}[difficulty]
        values = [rng.randint(0, 20) for _ in range(terms)]
        return ReasoningSample(task_family, difficulty, f"sum: {' + '.join(map(str, values))} =", f" {sum(values)}")
    if task_family == "rewrite":
        length = {"easy": 6, "medium": 10, "hard": 18}[difficulty]
        chars = [rng.choice(["a", "b", "c"]) for _ in range(length)]
        rewritten = [char.upper() if char in {"a", "c"} else char for char in chars]
        return ReasoningSample(task_family, difficulty, f"rewrite a,c upper: {''.join(chars)} ->", " " + "".join(rewritten))
    raise ValueError(f"Unsupported reasoning task family: {task_family}")


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


def _decode(tokenizer: Any, ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(ids)
    byte_values = bytes(int(value) for value in ids if 0 <= int(value) < 256)
    return byte_values.decode("utf-8", errors="ignore")


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


def _visible_cot_token_count(generated_ids: list[int], answer_ids: list[int]) -> int:
    if not generated_ids:
        return 0
    if len(generated_ids) >= len(answer_ids) and generated_ids[-len(answer_ids) :] == answer_ids:
        return max(0, len(generated_ids) - len(answer_ids))
    return max(0, len(generated_ids) - _longest_suffix_prefix_overlap(generated_ids, answer_ids))


def _longest_suffix_prefix_overlap(generated_ids: list[int], answer_ids: list[int]) -> int:
    max_len = min(len(generated_ids), len(answer_ids))
    for length in range(max_len, 0, -1):
        if generated_ids[-length:] == answer_ids[:length]:
            return length
    return 0


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: summarize_reasoning_rows(group_rows) for name, group_rows in sorted(groups.items())}


def _routing_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in rows for key in row if key.startswith("routing_")})
    return {key.removeprefix("routing_"): _mean([row.get(key) for row in rows]) for key in keys}


def _mean(values: list[Any]) -> float | None:
    numeric = [_num(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _device(name: str) -> "torch.device":
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
