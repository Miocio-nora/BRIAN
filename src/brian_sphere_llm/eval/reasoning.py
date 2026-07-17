from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.tokenize import load_tokenizer
from brian_sphere_llm.eval.difficulty_report import (
    _checkpoint_step,
    _forward_routed_for_eval,
    _forward_routed_incremental_for_eval,
    _forward_routed_stream_chunk_for_eval,
    _load_model_for_run,
)
from brian_sphere_llm.routing.metrics import summarize_routes
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json, write_jsonl

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


@dataclass(frozen=True)
class _PreparedReasoningSample:
    sample: ReasoningSample
    answer_ids: tuple[int, ...]
    full_ids: tuple[int, ...]
    teacher_start: int


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
    generation_mode: str = "reference",
    generation_batch_size: int = 1,
    teacher_mode: str = "reference",
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
    samples = list(
        generate_reasoning_samples(
            sample_count,
            seed=seed,
            task_families=task_families,
            difficulties=difficulties,
        )
    )
    if generation_mode not in {"reference", "batched_incremental"}:
        raise ValueError("generation_mode must be 'reference' or 'batched_incremental'.")
    if teacher_mode not in {"reference", "exact_length_batch"}:
        raise ValueError("teacher_mode must be 'reference' or 'exact_length_batch'.")
    if teacher_mode == "exact_length_batch" and generation_mode != "batched_incremental":
        raise ValueError("exact_length_batch teacher mode requires batched_incremental generation.")
    if (
        isinstance(generation_batch_size, bool)
        or not isinstance(generation_batch_size, int)
        or generation_batch_size < 1
    ):
        raise ValueError("generation_batch_size must be a positive integer.")
    evaluation_started = time.perf_counter()
    generated_by_sample: list[list[int]] | None = None
    generation_seconds: float | None = None
    if generation_mode == "batched_incremental":
        prompt_ids = [_prompt_ids(tokenizer, sample.prompt)[-context_length:] for sample in samples]
        answer_lengths = [len(tokenizer.encode(sample.answer, add_special_tokens=False)) for sample in samples]
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated_by_sample = batched_greedy_generate(
                model,
                prompt_ids,
                new_token_counts=answer_lengths,
                batch_size=generation_batch_size,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                context_length=context_length,
                device=device,
            )
        generation_seconds = time.perf_counter() - generation_started
    teacher_started = time.perf_counter()
    with torch.inference_mode():
        if generated_by_sample is not None and teacher_mode == "exact_length_batch":
            rows = evaluate_reasoning_samples_batched(
                model,
                tokenizer,
                samples,
                generated_by_sample=generated_by_sample,
                batch_size=generation_batch_size,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                context_length=context_length,
                device=device,
            )
        else:
            rows = [
                evaluate_reasoning_sample(
                    model,
                    tokenizer,
                    sample,
                    config=config,
                    route_mode=route_mode,
                    global_step=global_step,
                    context_length=context_length,
                    sample_id=index,
                    device=device,
                    generated_ids=None if generated_by_sample is None else generated_by_sample[index],
                )
                for index, sample in enumerate(samples)
            ]
    teacher_elapsed = time.perf_counter() - teacher_started
    teacher_seconds = teacher_elapsed if generated_by_sample is not None else None
    evaluation_seconds = time.perf_counter() - evaluation_started

    if output_path is None:
        output_path = run_dir / "reasoning_report.json"
    output_path = Path(output_path)
    if sample_output_path is None:
        sample_output_path = output_path.with_name(output_path.stem + "_samples.jsonl")
    sample_output_path = Path(sample_output_path)
    _write_jsonl(rows, sample_output_path)
    overall = summarize_reasoning_rows(rows)
    checks = _report_checks(overall)
    report = {
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "route_mode": route_mode,
        "hard_exit": _hard_exit_enabled(config),
        "checkpoint": str(checkpoint),
        "sample_count": len(rows),
        "seed": seed,
        "context_length": context_length,
        "inference": {
            "generation_mode": generation_mode,
            "generation_batch_size": generation_batch_size,
            "teacher_mode": teacher_mode,
            "routing_equivalence": "numeric" if teacher_mode == "exact_length_batch" else "reference",
            "generation_seconds": generation_seconds,
            "teacher_seconds": teacher_seconds,
            "evaluation_seconds": evaluation_seconds,
            "incremental_cache": bool(
                generation_mode == "batched_incremental"
                and route_mode != "baseline"
                and hasattr(model, "forward_stream_chunk")
                and hasattr(model, "forward_incremental")
            ),
        },
        "samples_path": str(sample_output_path),
        "overall": overall,
        "by_task_family": _group_summary(rows, "task_family"),
        "by_difficulty": _group_summary(rows, "difficulty"),
        "routing": _routing_summary(rows),
        "checks": checks,
        "overall_status": _overall_status(checks),
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
    generated_ids: list[int] | None = None,
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
    if generated_ids is None:
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
    return _reasoning_row(
        tokenizer,
        sample,
        sample_id=sample_id,
        answer_ids=answer_ids,
        generated_ids=generated_ids,
        teacher_accuracy=teacher_accuracy,
        routing_summary=outputs.get("routing_summary", {}),
    )


def evaluate_reasoning_samples_batched(
    model: Any,
    tokenizer: Any,
    samples: list[ReasoningSample],
    *,
    generated_by_sample: list[list[int]],
    batch_size: int,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    context_length: int,
    device: "torch.device",
) -> list[dict[str, Any]]:
    if len(samples) != len(generated_by_sample):
        raise ValueError("samples and generated_by_sample must have the same length.")
    prepared = [_prepare_reasoning_sample(tokenizer, sample, context_length) for sample in samples]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(prepared):
        groups[len(item.full_ids)].append(index)
    rows: list[dict[str, Any] | None] = [None] * len(samples)
    num_internal_blocks = int(getattr(getattr(model, "config", None), "route_pool_blocks", 0))

    for indexes in groups.values():
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            batch_items = [prepared[index] for index in batch_indexes]
            input_ids = torch.tensor(
                [item.full_ids for item in batch_items],
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
            predictions = outputs["logits"].argmax(dim=-1).detach().cpu()
            routing_summaries = _per_sample_routing_summaries(
                outputs.get("route_info"),
                batch_size=len(batch_items),
                num_internal_blocks=num_internal_blocks,
            )
            for local_index, (sample_index, item) in enumerate(
                zip(batch_indexes, batch_items, strict=True)
            ):
                answer_ids = list(item.answer_ids)
                teacher_predictions = predictions[
                    local_index,
                    item.teacher_start : item.teacher_start + len(answer_ids),
                ].tolist()
                rows[sample_index] = _reasoning_row(
                    tokenizer,
                    item.sample,
                    sample_id=sample_index,
                    answer_ids=answer_ids,
                    generated_ids=generated_by_sample[sample_index],
                    teacher_accuracy=_token_accuracy(teacher_predictions, answer_ids),
                    routing_summary=routing_summaries[local_index],
                )
    if any(row is None for row in rows):
        raise RuntimeError("Batched reasoning evaluation did not produce every row.")
    return [row for row in rows if row is not None]


def _reasoning_row(
    tokenizer: Any,
    sample: ReasoningSample,
    *,
    sample_id: int,
    answer_ids: list[int],
    generated_ids: list[int],
    teacher_accuracy: float,
    routing_summary: Mapping[str, Any],
) -> dict[str, Any]:
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
    for key, value in routing_summary.items():
        number = _num(value)
        if number is not None:
            row[f"routing_{key}"] = number
    return row


def _prepare_reasoning_sample(
    tokenizer: Any,
    sample: ReasoningSample,
    context_length: int,
) -> _PreparedReasoningSample:
    prompt_ids = _prompt_ids(tokenizer, sample.prompt)
    answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Reasoning sample produced an empty answer.")
    full_ids = (prompt_ids + answer_ids)[-context_length:]
    prompt_len = min(len(prompt_ids), len(full_ids) - len(answer_ids))
    return _PreparedReasoningSample(
        sample=sample,
        answer_ids=tuple(answer_ids),
        full_ids=tuple(full_ids),
        teacher_start=max(0, prompt_len - 1),
    )


def _per_sample_routing_summaries(
    route_info: Mapping[str, Any] | None,
    *,
    batch_size: int,
    num_internal_blocks: int,
) -> list[dict[str, Any]]:
    if not route_info or num_internal_blocks < 1:
        return [{} for _ in range(batch_size)]
    cpu_info = _route_info_to_cpu(route_info)
    summaries: list[dict[str, Any]] = []
    for sample_index in range(batch_size):
        sample_info: dict[str, Any] = {}
        for key, value in cpu_info.items():
            if key in {"random_route_override_count", "self_recur_cap_count"}:
                continue
            if isinstance(value, list):
                sample_info[key] = [
                    item[sample_index : sample_index + 1]
                    if isinstance(item, torch.Tensor) and item.dim() > 0 and item.size(0) == batch_size
                    else item
                    for item in value
                ]
            elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == batch_size:
                sample_info[key] = value[sample_index : sample_index + 1]
            else:
                sample_info[key] = value
        summaries.append(summarize_routes(sample_info, num_internal_blocks))
    return summaries


def _route_info_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _route_info_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_route_info_to_cpu(item) for item in value]
    return value


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
        outputs = _forward_routed_for_eval(
            model,
            input_ids,
            config=config,
            route_mode=route_mode,
            global_step=global_step,
        )
        next_id = int(outputs["logits"][0, -1].argmax().detach().cpu())
        generated.append(next_id)
        current.append(next_id)
    return generated


def batched_greedy_generate(
    model: Any,
    prompt_ids: list[list[int]],
    *,
    new_token_counts: list[int],
    batch_size: int,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    context_length: int,
    device: "torch.device",
) -> list[list[int]]:
    """Generate equal-shape groups together and reuse RC-KV state when available."""

    if len(prompt_ids) != len(new_token_counts):
        raise ValueError("prompt_ids and new_token_counts must have the same length.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    generated: list[list[int] | None] = [None] * len(prompt_ids)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    fallback: list[int] = []
    for index, (prompt, count) in enumerate(zip(prompt_ids, new_token_counts, strict=True)):
        if not prompt:
            raise ValueError("Batched generation requires non-empty prompts.")
        if count < 1:
            raise ValueError("Batched generation requires positive new-token counts.")
        if len(prompt) + count > context_length:
            fallback.append(index)
        else:
            groups[(len(prompt), count)].append(index)

    use_incremental_cache = (
        route_mode != "baseline"
        and hasattr(model, "forward_stream_chunk")
        and hasattr(model, "forward_incremental")
    )
    for (_, count), indexes in groups.items():
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            batch_prompts = torch.tensor(
                [prompt_ids[index] for index in batch_indexes],
                dtype=torch.long,
                device=device,
            )
            batch_generated = _greedy_generate_equal_shape_batch(
                model,
                batch_prompts,
                new_tokens=count,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                use_incremental_cache=use_incremental_cache,
            )
            for index, token_ids in zip(batch_indexes, batch_generated, strict=True):
                generated[index] = token_ids

    for index in fallback:
        generated[index] = greedy_generate(
            model,
            prompt_ids[index],
            new_tokens=new_token_counts[index],
            config=config,
            route_mode=route_mode,
            global_step=global_step,
            context_length=context_length,
            device=device,
        )
    if any(value is None for value in generated):
        raise RuntimeError("Batched generation did not produce every requested sample.")
    return [list(value) for value in generated if value is not None]


def _greedy_generate_equal_shape_batch(
    model: Any,
    prompt_batch: "torch.Tensor",
    *,
    new_tokens: int,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    use_incremental_cache: bool,
) -> list[list[int]]:
    generated: list[torch.Tensor] = []
    if use_incremental_cache:
        outputs = _forward_routed_stream_chunk_for_eval(
            model,
            prompt_batch,
            None,
            config=config,
            route_mode=route_mode,
            global_step=global_step,
            summarize_routing=False,
        )
        state = outputs["incremental_state"]
        next_ids = outputs["logits"][:, -1].argmax(dim=-1)
        generated.append(next_ids)
        for _ in range(1, new_tokens):
            outputs = _forward_routed_incremental_for_eval(
                model,
                next_ids.unsqueeze(1),
                state,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                summarize_routing=False,
            )
            state = outputs["incremental_state"]
            next_ids = outputs["logits"][:, -1].argmax(dim=-1)
            generated.append(next_ids)
    else:
        current = prompt_batch
        for _ in range(new_tokens):
            outputs = _forward_routed_for_eval(
                model,
                current,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                summarize_routing=False,
            )
            next_ids = outputs["logits"][:, -1].argmax(dim=-1)
            generated.append(next_ids)
            current = torch.cat((current, next_ids.unsqueeze(1)), dim=1)
    return torch.stack(generated, dim=1).detach().cpu().tolist()


def summarize_reasoning_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "exact_match_accuracy": None,
            "teacher_forced_token_accuracy": None,
            "generated_tokens_mean": None,
            "answer_tokens_mean": None,
            "visible_cot_tokens_mean": None,
        }
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
    tokenizer_config = _tokenizer_config(config)
    return load_tokenizer(
        str(tokenizer_config.get("name", "simple-byte-tokenizer")),
        revision=str(tokenizer_config.get("revision", "main")),
        local_files_only=_bool_mapping_value(
            tokenizer_config,
            "local_files_only",
            default=True,
            name="tokenizer.local_files_only",
        ),
        fallback_to_byte=_bool_mapping_value(
            tokenizer_config,
            "fallback_to_byte",
            default=True,
            name="tokenizer.fallback_to_byte",
        ),
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
    if isinstance(data_config, Mapping) and data_config.get("sequence_length") is not None:
        return _int_value(data_config["sequence_length"], "data_config_resolved.sequence_length", minimum=1)
    model_config = config.get("model_config_resolved", {})
    if isinstance(model_config, Mapping) and model_config.get("context_length") is not None:
        return _int_value(model_config["context_length"], "model_config_resolved.context_length", minimum=1)
    if isinstance(model_config, Mapping) and isinstance(model_config.get("base"), Mapping):
        base_config = model_config["base"]
        return _int_value(base_config.get("context_length", 128), "model_config_resolved.base.context_length", minimum=1)
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


def _report_checks(overall: dict[str, Any]) -> dict[str, bool]:
    return {
        "samples_present": _positive_number(overall.get("sample_count")),
        "exact_match_accuracy_present": _num(overall.get("exact_match_accuracy")) is not None,
        "teacher_forced_token_accuracy_present": _num(overall.get("teacher_forced_token_accuracy")) is not None,
        "visible_cot_tokens_present": _num(overall.get("visible_cot_tokens_mean")) is not None,
    }


def _overall_status(checks: dict[str, bool]) -> str:
    required = [
        "samples_present",
        "exact_match_accuracy_present",
        "teacher_forced_token_accuracy_present",
    ]
    if not all(checks.get(key) is True for key in required):
        return "fail"
    if all(value is True for value in checks.values()):
        return "pass"
    return "warn"


def _hard_exit_enabled(config: dict[str, Any]) -> bool | None:
    routing_config = config.get("routing", {})
    if isinstance(routing_config, Mapping) and isinstance(routing_config.get("hard_exit"), bool):
        return bool(routing_config["hard_exit"])
    if str(config.get("stage", "")) == "stage4_output_action":
        return True
    return None


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


def _positive_number(value: Any) -> bool:
    number = _num(value)
    return number is not None and number > 0.0


def _tokenizer_config(config: dict[str, Any]) -> Mapping[str, Any]:
    data_config = config.get("data_config_resolved", {})
    if data_config is None:
        return {}
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config_resolved must be a mapping.")
    tokenizer_config = data_config.get("tokenizer", {})
    if tokenizer_config is None:
        return {}
    if not isinstance(tokenizer_config, Mapping):
        raise ValueError("data_config_resolved.tokenizer must be a mapping.")
    return tokenizer_config


def _bool_mapping_value(mapping: Mapping[str, Any], key: str, *, default: bool, name: str) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _int_value(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def _device(name: str) -> "torch.device":
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    write_jsonl(rows, path)
