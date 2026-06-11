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


TASK_FAMILY_ALIASES = {
    "synthetic_multihop_tracing": "two_hop_tracing",
    "ruler_subset": "ruler_needle",
    "longbench_subset": "longbench_qa",
    "long_program_trace": "program_trace",
}


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
    expected_task_families = task_families or ["needle_retrieval", "two_hop_tracing"]
    expected_difficulties = difficulties or ["near", "middle", "far"]
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
        "coverage": _coverage_summary(rows, expected_task_families, expected_difficulties),
        "by_task_family": _group_summary(rows, "task_family"),
        "by_difficulty": _group_summary(rows, "difficulty"),
        "routing": _routing_summary(rows),
        "global_kv": _global_kv_summary(rows),
        "memory_budget": _memory_budget_summary(config, rows),
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
    canonical_family = TASK_FAMILY_ALIASES.get(task_family, task_family)
    key = f"K{rng.randint(10, 99)}"
    value = str(rng.randint(100, 999))
    filler_count = max(2, min(10, context_length // 10))
    filler = [f"n{index}." for index in range(filler_count)]
    insert_at = _insert_index(difficulty, filler_count)
    if canonical_family == "needle_retrieval":
        parts = [*filler]
        parts.insert(insert_at, f"{key}={value}.")
        return LongContextSample(task_family, difficulty, "ctx " + " ".join(parts) + f" Q {key}? A:", " " + value, key)
    if canonical_family == "two_hop_tracing":
        link_key = f"L{rng.randint(10, 99)}"
        parts = [*filler]
        parts.insert(insert_at, f"{key}->{link_key}.")
        parts.insert(min(len(parts), insert_at + 2), f"{link_key}={value}.")
        return LongContextSample(task_family, difficulty, "ctx " + " ".join(parts) + f" Q {key}? A:", " " + value, key)
    if canonical_family == "ruler_needle":
        parts = [*filler]
        for index in range(3):
            parts.insert(min(len(parts), insert_at + index), f"D{index}={rng.randint(100, 999)}.")
        parts.insert(insert_at, f"{key}={value}.")
        prompt = "ruler " + " ".join(parts) + f" Query value for {key}. Answer:"
        return LongContextSample(task_family, difficulty, prompt, " " + value, key)
    if canonical_family == "longbench_qa":
        subject = f"agent{rng.randint(10, 99)}"
        parts = [*filler]
        parts.insert(insert_at, f"The dossier says {subject} carried token {value}.")
        prompt = "document " + " ".join(parts) + f" Question: Which token did {subject} carry? Answer:"
        return LongContextSample(task_family, difficulty, prompt, " " + value, subject)
    if canonical_family == "long_arithmetic_trace":
        values = [rng.randint(1, 9) for _ in range(3)]
        total = sum(values)
        parts = [*filler]
        parts.insert(insert_at, f"trace A={values[0]}; B={values[1]}; C={values[2]}.")
        prompt = "arith " + " ".join(parts) + " Q sum A B C? A:"
        return LongContextSample(task_family, difficulty, prompt, " " + str(total), "A+B+C")
    if canonical_family == "program_trace":
        start = rng.randint(1, 9)
        increment = rng.randint(1, 9)
        total = start + increment
        parts = [*filler]
        parts.insert(insert_at, f"program x={start}; x=x+{increment}; return x.")
        prompt = "code " + " ".join(parts) + " Q final x? A:"
        return LongContextSample(task_family, difficulty, prompt, " " + str(total), "x")
    raise ValueError(f"Unsupported long-context task family: {task_family}")


def _insert_index(difficulty: str, filler_count: int) -> int:
    if difficulty == "far":
        return max(0, filler_count // 5)
    if difficulty == "middle":
        return filler_count // 2
    if difficulty == "near":
        return max(0, filler_count - 1)
    raise ValueError(f"Unsupported long-context difficulty: {difficulty}")


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


def _coverage_summary(
    rows: list[dict[str, Any]],
    expected_task_families: list[str],
    expected_difficulties: list[str],
) -> dict[str, Any]:
    observed_task_families = sorted({str(row.get("task_family")) for row in rows})
    observed_difficulties = sorted({str(row.get("difficulty")) for row in rows})
    missing_task_families = [family for family in expected_task_families if family not in observed_task_families]
    missing_difficulties = [difficulty for difficulty in expected_difficulties if difficulty not in observed_difficulties]
    return {
        "expected_task_families": expected_task_families,
        "observed_task_families": observed_task_families,
        "missing_task_families": missing_task_families,
        "task_family_coverage_passed": not missing_task_families,
        "expected_difficulties": expected_difficulties,
        "observed_difficulties": observed_difficulties,
        "missing_difficulties": missing_difficulties,
        "difficulty_coverage_passed": not missing_difficulties,
    }


def _routing_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in rows for key in row if key.startswith("routing_")})
    return {key.removeprefix("routing_"): _mean([row.get(key) for row in rows]) for key in keys}


def _global_kv_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    read_gate = _mean([row.get("routing_global_read_gate_mean") for row in rows])
    read_gate = min(1.0, max(0.0, read_gate)) if read_gate is not None else None
    local_fraction = 1.0 - read_gate if read_gate is not None else None
    return {
        "global_attention_mass": _mean([row.get("routing_global_attention_mass") for row in rows]),
        "global_sink_attention_mass": _mean([row.get("routing_global_sink_attention_mass") for row in rows]),
        "global_window_attention_mass": _mean([row.get("routing_global_window_attention_mass") for row in rows]),
        "global_read_gate_mean": read_gate,
        "local_read_fraction_mean": local_fraction,
        "global_to_local_read_ratio": _bounded_ratio(read_gate, local_fraction),
        "local_to_global_read_ratio": _bounded_ratio(local_fraction, read_gate),
        "global_cache_slots_mean": _mean([row.get("routing_global_cache_slots_mean") for row in rows]),
    }


def _memory_budget_summary(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_config = config.get("model_config_resolved", {})
    model_config = model_config if isinstance(model_config, dict) else {}
    base_config = _base_model_config(model_config)
    context_length = _context_length(config)
    layer_count = _num(base_config.get("layers"))
    if layer_count is None:
        layer_count = _num(model_config.get("layers"))
    if layer_count is None:
        pre = _num(model_config.get("pre_blocks")) or 0.0
        route = _num(model_config.get("route_pool_blocks")) or 0.0
        post = _num(model_config.get("post_blocks")) or 0.0
        layer_count = pre + route + post if pre + route + post > 0 else None
    d_model = _num(base_config.get("d_model"))
    if d_model is None:
        d_model = _num(model_config.get("d_model"))
    dtype_bytes = 2.0
    local_bytes_per_token = layer_count * d_model * 2.0 * dtype_bytes if layer_count and d_model else None
    local_context_bytes = local_bytes_per_token * context_length if local_bytes_per_token is not None else None
    global_enabled = _as_bool(model_config.get("global_kv", False))
    global_code_dim = _num(model_config.get("global_code_dim"))
    global_sink_slots = _num(model_config.get("global_sink_slots")) or 0.0
    global_window_slots = _num(model_config.get("global_window_slots")) or 0.0
    global_capacity_slots = global_sink_slots + global_window_slots if global_enabled else 0.0
    global_capacity_bytes = (
        global_capacity_slots * global_code_dim * dtype_bytes if global_enabled and global_code_dim is not None else None
    )
    global_mean_slots = _global_kv_summary(rows).get("global_cache_slots_mean")
    global_mean_bytes = (
        global_mean_slots * global_code_dim * dtype_bytes
        if global_enabled and global_mean_slots is not None and global_code_dim is not None
        else None
    )
    global_window_used_slots = (
        max(0.0, global_mean_slots - global_sink_slots)
        if global_enabled and global_mean_slots is not None
        else None
    )
    return {
        "estimation": "fp16_kv_and_global_code_bytes",
        "context_length": context_length,
        "base_layer_count": layer_count,
        "base_d_model": d_model,
        "estimated_local_raw_kv_bytes_per_token_fp16": local_bytes_per_token,
        "estimated_local_raw_kv_context_bytes_fp16": local_context_bytes,
        "global_kv_enabled": global_enabled,
        "global_code_dim": global_code_dim,
        "global_sink_slots": global_sink_slots,
        "global_window_slots": global_window_slots,
        "estimated_global_cache_capacity_slots": global_capacity_slots,
        "estimated_global_cache_window_used_slots": global_window_used_slots,
        "estimated_global_cache_capacity_bytes_fp16": global_capacity_bytes,
        "estimated_global_cache_mean_bytes_fp16": global_mean_bytes,
        "estimated_global_cache_window_utilization": _ratio(global_window_used_slots, global_window_slots),
        "estimated_global_cache_capacity_utilization": _ratio(global_mean_slots, global_capacity_slots),
        "estimated_global_cache_capacity_to_local_context_ratio": _ratio(global_capacity_bytes, local_context_bytes),
        "estimated_global_cache_mean_to_local_context_ratio": _ratio(global_mean_bytes, local_context_bytes),
    }


def _base_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    base = model_config.get("base")
    if isinstance(base, dict):
        return base
    base_config = model_config.get("base_config")
    if not base_config:
        return {}
    base_path = Path(str(base_config))
    candidates = []
    if base_path.is_absolute():
        candidates.append(base_path)
    candidates.append(Path(__file__).resolve().parents[3] / "configs" / "model" / base_path.name)
    for candidate in candidates:
        if candidate.is_file():
            return load_config(candidate)
    return {}


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _bounded_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(1e-9, denominator)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
