#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.difficulty_report import _device, _load_model_for_run
from brian_sphere_llm.eval.reasoning import (
    _load_tokenizer_from_run_config,
    generate_reasoning_samples,
    normalize_answer,
)
from brian_sphere_llm.memory.attention_global_cache import AttentionGlobalKVState
from brian_sphere_llm.model.llama_backbone import CausalSelfAttention
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config


CHECKPOINTS = {
    "hidden": {
        "run": "runs/corrected_global_kv_r125_5b_balanced_slow_noise",
        "checkpoints": ["checkpoint_step_00015000", "checkpoint_step_00030000", "checkpoint_step_00045000"],
    },
    "attention": {
        "run": "runs/corrected_attention_global_kv_r125_5b_balanced_slow_noise",
        "checkpoints": [
            "checkpoint_step_00015000",
            "checkpoint_step_00030000",
            "checkpoint_step_00045000",
            "checkpoint_step_00060000",
        ],
    },
}
SCALES = [0.0, 0.25, 0.5, 1.0, 1.5]
INTERVENTIONS = ["default", "global_off", "sink_only", "window_only", "slot_shuffle", "batch_memory_swap"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Global KV checkpoint diagnostics without training.")
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--reasoning-samples", type=int, default=8)
    parser.add_argument("--public-samples", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--incremental-positions", default="16,32,48,64,80,95")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    batches = _load_eval_batches(batch_size=args.batch_size, max_batches=max(2, args.val_batches), device=device)
    tokenizer_cache: dict[str, Any] = {}

    suffix_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []

    for family, spec in CHECKPOINTS.items():
        run_dir = Path(spec["run"])
        config = load_config(run_dir / "config_resolved.yaml")
        tokenizer_cache[str(run_dir)] = tokenizer_cache.get(str(run_dir)) or _load_tokenizer_from_run_config(config)
        route_mode = train_mode_for_stage(str(config["stage"]))
        for checkpoint in spec["checkpoints"]:
            print(f"[diagnostics] loading {family} {checkpoint}", flush=True)
            model = _load_model_for_run(run_dir, checkpoint, device)
            model.eval()
            global_step = _checkpoint_step(run_dir, checkpoint)
            tokenizer = tokenizer_cache[str(run_dir)]

            suffix_rows.append(
                p0_suffix_invariance(
                    model,
                    config,
                    family=family,
                    checkpoint=checkpoint,
                    route_mode=route_mode,
                    global_step=global_step,
                    batch_a=batches[0],
                    batch_b=batches[1],
                    prefix_len=args.prefix_len,
                    seq_len=args.seq_len,
                )
            )
            incremental_rows.extend(
                p0_full_vs_incremental(
                    model,
                    config,
                    family=family,
                    checkpoint=checkpoint,
                    route_mode=route_mode,
                    global_step=global_step,
                    batch=batches[0],
                    positions=[int(value) for value in args.incremental_positions.split(",") if value.strip()],
                )
            )
            norm_rows.append(
                p1_norm_audit(
                    model,
                    config,
                    family=family,
                    checkpoint=checkpoint,
                    route_mode=route_mode,
                    global_step=global_step,
                    batch=batches[0],
                )
            )
            for scale in SCALES:
                sweep_rows.append(
                    evaluate_checkpoint_variant(
                        model,
                        config,
                        tokenizer,
                        family=family,
                        checkpoint=checkpoint,
                        route_mode=route_mode,
                        global_step=global_step,
                        batches=batches[: args.val_batches],
                        scale=scale,
                        intervention="default",
                        reasoning_samples=args.reasoning_samples,
                        public_samples=args.public_samples,
                    )
                )
            for intervention in INTERVENTIONS:
                memory_rows.append(
                    evaluate_checkpoint_variant(
                        model,
                        config,
                        tokenizer,
                        family=family,
                        checkpoint=checkpoint,
                        route_mode=route_mode,
                        global_step=global_step,
                        batches=batches[: args.val_batches],
                        scale=1.0,
                        intervention=intervention,
                        reasoning_samples=args.reasoning_samples,
                        public_samples=args.public_samples,
                    )
                )
            route_rows.extend(
                p1_route_interventions(
                    model,
                    config,
                    tokenizer,
                    family=family,
                    checkpoint=checkpoint,
                    route_mode=route_mode,
                    global_step=global_step,
                    batch=batches[0],
                    reasoning_samples=args.reasoning_samples,
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(output_dir / "p0_suffix_invariance.csv", suffix_rows)
    _write_csv(output_dir / "p0_full_vs_incremental.csv", incremental_rows)
    _write_csv(output_dir / "p0_global_sweep.csv", sweep_rows)
    _write_csv(output_dir / "p0_memory_intervention.csv", memory_rows)
    _write_csv(output_dir / "p1_norm_audit.csv", norm_rows)
    _write_csv(output_dir / "p1_route_intervention.csv", route_rows)
    matrix_rows = checkpoint_selection_matrix(sweep_rows, norm_rows)
    _write_csv(output_dir / "checkpoint_selection_matrix.csv", matrix_rows)
    write_summary(output_dir / "summary.md", suffix_rows, incremental_rows, sweep_rows, memory_rows, norm_rows, route_rows, matrix_rows)


def _load_eval_batches(*, batch_size: int, max_batches: int, device: torch.device) -> list[torch.Tensor]:
    loader = build_dataloader(
        tokenized_dir=REPO_ROOT / "data/tokenized/r125_main_5b_balanced",
        split="val",
        batch_size=batch_size,
        shuffle=False,
    )
    batches = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batches.append(batch.to(device))
    if len(batches) < max_batches:
        raise RuntimeError(f"Needed {max_batches} validation batches, got {len(batches)}")
    return batches


def _checkpoint_step(run_dir: Path, checkpoint: str) -> int:
    payload = torch.load(run_dir / checkpoint / "state.pt", map_location="cpu")
    return int(payload.get("step", 0))


def forward_model(
    model: Any,
    batch: torch.Tensor,
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
    targets: torch.Tensor | None = None,
    pseudo_policy_override: str | None = None,
    route_mode_override: str | None = None,
) -> dict[str, Any]:
    if route_mode == "baseline":
        return model(batch, targets=targets)
    routing_cfg = dict(config.get("routing") or {})
    loss_weights = dict(config.get("loss_weights") or {})
    actual_route_mode = route_mode_override or route_mode
    router_probability = None
    if actual_route_mode == "scheduled":
        router_probability = _scheduled_value(routing_cfg.get("schedule", []), global_step, "router_probability", 0.0)
        loss_weights["route"] = _scheduled_value(
            routing_cfg.get("schedule", []),
            global_step,
            "lambda_route",
            float(loss_weights.get("route", 0.0)),
        )
    return model(
        batch,
        targets=targets,
        route_mode=actual_route_mode,
        pseudo_policy=pseudo_policy_override or str(routing_cfg.get("pseudo_policy", "sequential")),
        loss_weights=loss_weights,
        routing_constraints=dict(routing_cfg.get("constraints") or {}),
        routing_options=routing_cfg,
        hard_exit=bool(routing_cfg.get("hard_exit", True)),
        router_probability=router_probability,
        global_step=global_step,
    )


def p0_suffix_invariance(
    model: Any,
    config: dict[str, Any],
    *,
    family: str,
    checkpoint: str,
    route_mode: str,
    global_step: int,
    batch_a: torch.Tensor,
    batch_b: torch.Tensor,
    prefix_len: int,
    seq_len: int,
) -> dict[str, Any]:
    tokens_a = batch_a[:1, :seq_len].clone()
    tokens_b = tokens_a.clone()
    tokens_b[:, prefix_len:seq_len] = batch_b[:1, prefix_len:seq_len]
    with diagnostic_context(model, family, scale=1.0, intervention="default"):
        out_a = forward_model(model, tokens_a, config=config, route_mode=route_mode, global_step=global_step)
        out_b = forward_model(model, tokens_b, config=config, route_mode=route_mode, global_step=global_step)
    logits_diff = (out_a["logits"][:, :prefix_len] - out_b["logits"][:, :prefix_len]).float().abs()
    route_diff = _route_logits_diff(out_a.get("route_info", {}), out_b.get("route_info", {}))
    summary_a = out_a.get("routing_summary", {})
    summary_b = out_b.get("routing_summary", {})
    return {
        "family": family,
        "checkpoint": checkpoint,
        "prefix_len": prefix_len,
        "seq_len": seq_len,
        "prefix_logits_max_diff": _float(logits_diff.max()),
        "prefix_logits_mean_diff": _float(logits_diff.mean()),
        "route_logits_max_diff": route_diff["max"],
        "route_logits_mean_diff": route_diff["mean"],
        "global_mass_abs_diff": abs(_summary_global_mass(summary_a) - _summary_global_mass(summary_b)),
        "route_entropy_a": _route_entropy(out_a.get("route_info", {})),
        "route_entropy_b": _route_entropy(out_b.get("route_info", {})),
    }


def p0_full_vs_incremental(
    model: Any,
    config: dict[str, Any],
    *,
    family: str,
    checkpoint: str,
    route_mode: str,
    global_step: int,
    batch: torch.Tensor,
    positions: list[int],
) -> list[dict[str, Any]]:
    full = batch[:1]
    with diagnostic_context(model, family, scale=1.0, intervention="default"):
        full_out = forward_model(model, full, config=config, route_mode=route_mode, global_step=global_step)
    rows = []
    full_actions = _selected_actions(full_out.get("route_info", {}))
    for position in positions:
        position = max(1, min(int(position), full.size(1) - 1))
        prefix = full[:, : position + 1]
        with diagnostic_context(model, family, scale=1.0, intervention="default"):
            prefix_out = forward_model(model, prefix, config=config, route_mode=route_mode, global_step=global_step)
        diff = (full_out["logits"][:, position] - prefix_out["logits"][:, -1]).float().abs()
        prefix_actions = _selected_actions(prefix_out.get("route_info", {}))
        rows.append(
            {
                "family": family,
                "checkpoint": checkpoint,
                "position": position,
                "logits_max_diff": _float(diff.max()),
                "logits_mean_diff": _float(diff.mean()),
                "route_action_mismatch_rate": _action_mismatch_rate(full_actions, prefix_actions),
                "full_global_mass": _summary_global_mass(full_out.get("routing_summary", {})),
                "prefix_global_mass": _summary_global_mass(prefix_out.get("routing_summary", {})),
            }
        )
    return rows


def evaluate_checkpoint_variant(
    model: Any,
    config: dict[str, Any],
    tokenizer: Any,
    *,
    family: str,
    checkpoint: str,
    route_mode: str,
    global_step: int,
    batches: list[torch.Tensor],
    scale: float,
    intervention: str,
    reasoning_samples: int,
    public_samples: int,
) -> dict[str, Any]:
    with diagnostic_context(model, family, scale=scale, intervention=intervention) as state:
        losses = []
        summaries = []
        route_entropies = []
        for batch in batches:
            out = forward_model(
                model,
                batch,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                targets=batch,
            )
            losses.append(_float(out["loss"]))
            summaries.append(out.get("routing_summary", {}))
            route_entropies.append(_route_entropy(out.get("route_info", {})))
        reasoning = quick_reasoning_eval(
            model,
            tokenizer,
            config,
            route_mode=route_mode,
            global_step=global_step,
            sample_count=reasoning_samples,
            device=batches[0].device,
        )
        public_avg = None
        if public_samples > 0:
            public_avg = quick_public_eval(
                model,
                tokenizer,
                config,
                route_mode=route_mode,
                global_step=global_step,
                sample_count=public_samples,
                device=batches[0].device,
            )
    loss = sum(losses) / max(1, len(losses))
    merged = _merge_summaries(summaries)
    return {
        "family": family,
        "checkpoint": checkpoint,
        "global_scale": scale,
        "intervention": intervention,
        "validation_loss": loss,
        "perplexity": math.exp(min(20.0, loss)),
        "reason_exact": reasoning["exact_match_accuracy"],
        "teacher_acc": reasoning["teacher_forced_token_accuracy"],
        "public_avg": public_avg,
        "route_entropy": _mean(route_entropies),
        "global_mass": _summary_global_mass(merged),
        "global_read_gate": _num_or_none(merged.get("global_read_gate_mean")),
        "global_local_ratio": _mean(state.delta_ratios),
        "attention_global_mass": _num_or_none(merged.get("attention_global_kv_last_token_mass")),
        "attention_global_logit_bias": _num_or_none(merged.get("attention_global_kv_logit_bias_mean")),
    }


def p1_norm_audit(
    model: Any,
    config: dict[str, Any],
    *,
    family: str,
    checkpoint: str,
    route_mode: str,
    global_step: int,
    batch: torch.Tensor,
) -> dict[str, Any]:
    with diagnostic_context(model, family, scale=1.0, intervention="default") as state:
        out = forward_model(model, batch, config=config, route_mode=route_mode, global_step=global_step, targets=batch)
    row: dict[str, Any] = {
        "family": family,
        "checkpoint": checkpoint,
        "validation_loss_one_batch": _float(out["loss"]),
        "route_entropy": _route_entropy(out.get("route_info", {})),
        "global_local_ratio": _mean(state.delta_ratios),
    }
    row.update(_global_parameter_norms(model, family))
    row.update(_flatten_summary(out.get("routing_summary", {})))
    return row


def p1_route_interventions(
    model: Any,
    config: dict[str, Any],
    tokenizer: Any,
    *,
    family: str,
    checkpoint: str,
    route_mode: str,
    global_step: int,
    batch: torch.Tensor,
    reasoning_samples: int,
) -> list[dict[str, Any]]:
    with diagnostic_context(model, family, scale=1.0, intervention="default"):
        base = forward_model(model, batch, config=config, route_mode=route_mode, global_step=global_step, targets=batch)
    top_block = _most_used_block(base.get("route_info", {}), getattr(model.config, "route_pool_blocks", 0))
    variants = [
        ("default", None, None, 1.0),
        ("sequential_route", "pseudo", "sequential", 1.0),
        ("random_legal_route", "pseudo", "random_internal", 1.0),
        ("global_off", None, None, 0.0),
        ("no_top_block", None, None, 1.0),
    ]
    rows = []
    for name, mode_override, policy, scale in variants:
        block_ctx = no_top_block_context(model, top_block) if name == "no_top_block" and top_block is not None else nullcontext()
        with block_ctx:
            with diagnostic_context(model, family, scale=scale, intervention="default"):
                out = forward_model(
                    model,
                    batch,
                    config=config,
                    route_mode=route_mode,
                    route_mode_override=mode_override,
                    pseudo_policy_override=policy,
                    global_step=global_step,
                    targets=batch,
                )
                reasoning = quick_reasoning_eval(
                    model,
                    tokenizer,
                    config,
                    route_mode=route_mode if mode_override is None else mode_override,
                    global_step=global_step,
                    sample_count=reasoning_samples,
                    device=batch.device,
                    pseudo_policy_override=policy,
                )
        diff = (out["logits"] - base["logits"]).float().abs()
        rows.append(
            {
                "family": family,
                "checkpoint": checkpoint,
                "route_intervention": name,
                "disabled_top_block": top_block,
                "validation_loss_one_batch": _float(out["loss"]),
                "logits_mean_abs_diff_vs_default": _float(diff.mean()),
                "logits_max_abs_diff_vs_default": _float(diff.max()),
                "route_entropy": _route_entropy(out.get("route_info", {})),
                "reason_exact": reasoning["exact_match_accuracy"],
                "teacher_acc": reasoning["teacher_forced_token_accuracy"],
            }
        )
    return rows


def quick_reasoning_eval(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    *,
    route_mode: str,
    global_step: int,
    sample_count: int,
    device: torch.device,
    pseudo_policy_override: str | None = None,
) -> dict[str, float | None]:
    if sample_count <= 0:
        return {"exact_match_accuracy": None, "teacher_forced_token_accuracy": None}
    context_length = _context_length(config)
    samples = generate_reasoning_samples(
        sample_count,
        seed=1,
        task_families=["copy", "reverse", "arithmetic", "rewrite"],
        difficulties=["easy", "medium", "hard"],
    )
    exact = 0
    teacher_scores = []
    for sample in samples:
        prompt_ids = _prompt_ids(tokenizer, sample.prompt)
        answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)
        full_ids = (prompt_ids + answer_ids)[-context_length:]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = forward_model(
            model,
            input_ids,
            config=config,
            route_mode=route_mode,
            global_step=global_step,
            pseudo_policy_override=pseudo_policy_override,
        )
        prompt_len = min(len(prompt_ids), len(full_ids) - len(answer_ids))
        start = max(0, prompt_len - 1)
        end = start + len(answer_ids)
        preds = out["logits"][0, start:end].argmax(dim=-1).detach().cpu().tolist()
        teacher_scores.append(sum(int(a == b) for a, b in zip(preds, answer_ids)) / max(1, len(answer_ids)))
        generated = []
        current = list(prompt_ids[-context_length:])
        for _ in answer_ids:
            window = current[-context_length:]
            step_input = torch.tensor([window], dtype=torch.long, device=device)
            step_out = forward_model(
                model,
                step_input,
                config=config,
                route_mode=route_mode,
                global_step=global_step,
                pseudo_policy_override=pseudo_policy_override,
            )
            next_id = int(step_out["logits"][0, -1].argmax().detach().cpu())
            generated.append(next_id)
            current.append(next_id)
        if normalize_answer(tokenizer.decode(generated)) == normalize_answer(sample.answer):
            exact += 1
    return {
        "exact_match_accuracy": exact / sample_count,
        "teacher_forced_token_accuracy": _mean(teacher_scores),
    }


def quick_public_eval(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    *,
    route_mode: str,
    global_step: int,
    sample_count: int,
    device: torch.device,
) -> float | None:
    try:
        from scripts.public_benchmark import TASKS, _load_examples
    except Exception:
        return None
    import random

    context_length = _context_length(config)
    rng = random.Random(1)
    correct = 0
    total = 0
    for task in TASKS:
        examples = _load_examples(task, sample_count=sample_count, rng=rng)
        for example in examples:
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
                )
                for choice in example["choices"]
            ]
            pred = max(range(len(scores)), key=lambda index: scores[index])
            correct += int(pred == int(example["label"]))
            total += 1
    return correct / total if total else None


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
    device: torch.device,
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
    out = forward_model(model, input_ids, config=config, route_mode=route_mode, global_step=global_step)
    target = input_ids[0, start:]
    pred_logits = out["logits"][0, start - 1 : input_ids.size(1) - 1]
    token_scores = F.log_softmax(pred_logits.float(), dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
    return float((token_scores.sum() / max(1, int(target.numel()))).detach().cpu())


class DiagnosticState:
    def __init__(self) -> None:
        self.delta_ratios: list[float] = []


@contextmanager
def diagnostic_context(model: Any, family: str, *, scale: float, intervention: str) -> Iterator[DiagnosticState]:
    state = DiagnosticState()
    if family == "hidden":
        with hidden_global_context(model, scale=scale, intervention=intervention, state=state):
            yield state
    elif family == "attention":
        with attention_global_context(model, scale=scale, intervention=intervention):
            yield state
    else:
        yield state


@contextmanager
def hidden_global_context(model: Any, *, scale: float, intervention: str, state: DiagnosticState) -> Iterator[None]:
    original = model._global_read

    def wrapped(hidden: torch.Tensor, codes: torch.Tensor, *, sink_slots: int, actions: torch.Tensor | None = None):
        if intervention == "global_off" or scale <= 0.0:
            zero = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
            return hidden, {
                "global_attention_mass": zero,
                "global_sink_attention_mass": zero,
                "global_window_attention_mass": zero,
                "global_read_gate": zero,
            }
        modified = _intervene_hidden_codes(codes, sink_slots=sink_slots, intervention=intervention)
        effective_sink = sink_slots if intervention != "window_only" else 0
        updated, metrics = original(hidden, modified, sink_slots=effective_sink, actions=actions)
        delta = updated - hidden
        ratio = float((delta.float().norm() / hidden.float().norm().clamp_min(1e-9)).detach().cpu())
        state.delta_ratios.append(ratio * float(scale))
        return hidden + delta * float(scale), metrics

    model._global_read = wrapped
    try:
        yield
    finally:
        model._global_read = original


def _intervene_hidden_codes(codes: torch.Tensor, *, sink_slots: int, intervention: str) -> torch.Tensor:
    if codes.size(1) == 0 or intervention == "default":
        return codes
    if intervention == "sink_only":
        return codes[:, :sink_slots, :]
    if intervention == "window_only":
        return codes[:, sink_slots:, :]
    if intervention == "slot_shuffle":
        perm = torch.arange(codes.size(1), device=codes.device)
        perm = torch.roll(perm, shifts=1)
        return codes.index_select(1, perm)
    if intervention == "batch_memory_swap":
        return torch.roll(codes, shifts=1, dims=0) if codes.size(0) > 1 else codes
    return codes


@contextmanager
def attention_global_context(model: Any, *, scale: float, intervention: str) -> Iterator[None]:
    original_index_select = AttentionGlobalKVState.index_select
    original_biases: list[tuple[CausalSelfAttention, torch.Tensor]] = []
    sink_slots = int(getattr(model.config, "attention_global_sink_slots", 0))
    actual_intervention = "global_off" if scale <= 0.0 else intervention
    bias_delta = math.log(float(scale)) if scale > 0.0 else 0.0
    for module in model.modules():
        if isinstance(module, CausalSelfAttention) and module.global_logit_bias is not None:
            original_biases.append((module, module.global_logit_bias.detach().clone()))
            if scale > 0.0:
                module.global_logit_bias.data.add_(bias_delta)

    def patched_index_select(self: AttentionGlobalKVState, indices: torch.Tensor) -> AttentionGlobalKVState:
        selected = original_index_select(self, indices)
        return _intervene_attention_state(selected, sink_slots=sink_slots, intervention=actual_intervention)

    AttentionGlobalKVState.index_select = patched_index_select
    try:
        yield
    finally:
        AttentionGlobalKVState.index_select = original_index_select
        for module, value in original_biases:
            module.global_logit_bias.data.copy_(value)


def _intervene_attention_state(
    state: AttentionGlobalKVState,
    *,
    sink_slots: int,
    intervention: str,
) -> AttentionGlobalKVState:
    slots = state.slots
    if slots == 0 or intervention == "default":
        return state
    if intervention == "global_off":
        return AttentionGlobalKVState(state.keys[:, :, :0, :], state.values[:, :, :0, :], state.valid[:, :0])
    if intervention == "sink_only":
        keep = slice(0, min(sink_slots, slots))
        return AttentionGlobalKVState(state.keys[:, :, keep, :], state.values[:, :, keep, :], state.valid[:, keep])
    if intervention == "window_only":
        keep = slice(min(sink_slots, slots), slots)
        return AttentionGlobalKVState(state.keys[:, :, keep, :], state.values[:, :, keep, :], state.valid[:, keep])
    if intervention == "slot_shuffle":
        perm = torch.roll(torch.arange(slots, device=state.keys.device), shifts=1)
        return AttentionGlobalKVState(
            state.keys.index_select(2, perm),
            state.values.index_select(2, perm),
            state.valid.index_select(1, perm),
        )
    if intervention == "batch_memory_swap" and state.keys.size(0) > 1:
        return AttentionGlobalKVState(
            torch.roll(state.keys, shifts=1, dims=0),
            torch.roll(state.values, shifts=1, dims=0),
            torch.roll(state.valid, shifts=1, dims=0),
        )
    return state


@contextmanager
def no_top_block_context(model: Any, block_index: int | None) -> Iterator[None]:
    if block_index is None:
        yield
        return
    original = model._apply_route_constraints

    def wrapped(logits: torch.Tensor, step: int, max_steps: int, constraints: dict[str, Any]) -> torch.Tensor:
        adjusted = original(logits, step, max_steps, constraints)
        if 0 <= block_index < adjusted.size(-1):
            adjusted = adjusted.clone()
            adjusted[:, block_index] = adjusted.new_tensor(-1.0e4)
        return adjusted

    model._apply_route_constraints = wrapped
    try:
        yield
    finally:
        model._apply_route_constraints = original


def checkpoint_selection_matrix(sweep_rows: list[dict[str, Any]], norm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    norm_by_key = {(row["family"], row["checkpoint"]): row for row in norm_rows}
    rows = []
    for row in sweep_rows:
        if row["intervention"] != "default" or float(row["global_scale"]) != 1.0:
            continue
        report_values = _cached_benchmark_values(str(row["family"]), str(row["checkpoint"]))
        out = dict(row)
        out.update(report_values)
        norm = norm_by_key.get((row["family"], row["checkpoint"]), {})
        out["norm_global_local_ratio"] = norm.get("global_local_ratio")
        rows.append(out)
    return rows


def _cached_benchmark_values(family: str, checkpoint: str) -> dict[str, Any]:
    prefix = "s_balanced_global_kv_slow_noise" if family == "hidden" else "s_balanced_attention_global_kv_slow_noise"
    step = checkpoint.replace("checkpoint_step_", "step")
    reasoning = REPO_ROOT / "reports/package_a_benchmarks" / f"{prefix}_{step}.reasoning_s600.json"
    public = REPO_ROOT / "reports/package_a_benchmarks" / f"public_{prefix}_{step}_s200.json"
    values: dict[str, Any] = {}
    if reasoning.exists():
        data = json.loads(reasoning.read_text(encoding="utf-8"))
        overall = data.get("overall") or {}
        values["cached_s600_reason_exact"] = overall.get("exact_match_accuracy")
        values["cached_s600_teacher_acc"] = overall.get("teacher_forced_token_accuracy")
    if public.exists():
        data = json.loads(public.read_text(encoding="utf-8"))
        values["cached_public_s200_avg"] = (data.get("overall") or {}).get("accuracy")
    return values


def write_summary(
    path: Path,
    suffix_rows: list[dict[str, Any]],
    incremental_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
) -> None:
    leakage_failures = [row for row in suffix_rows if float(row["prefix_logits_max_diff"]) > 1e-3]
    incremental_failures = [row for row in incremental_rows if float(row["logits_max_diff"]) > 1e-3]
    takeover_notes = _global_takeover_notes(sweep_rows, memory_rows)
    best_reason = sorted(
        [row for row in matrix_rows if row.get("cached_s600_reason_exact") is not None],
        key=lambda row: float(row["cached_s600_reason_exact"]),
        reverse=True,
    )
    lines = [
        "# Global KV Checkpoint Diagnostics",
        "",
        "Scope: existing checkpoints only; no retraining. Future-work training recipes are intentionally excluded.",
        "",
        "## Clean validation set",
        "",
        "- Active `data/tokenized/r125_main_5b_balanced/val.bin` is now exact-token clean against train.",
        "- Legacy validation is preserved as `val_legacy.bin` / `val_legacy.idx`.",
        "- Detailed filtering stats are in `data/tokenized/r125_main_5b_balanced/val_clean_stats.json`.",
        "",
        "## Answers",
        "",
        f"1. Causal leakage: {'YES' if leakage_failures else 'not detected'} by suffix invariance. "
        f"{len(leakage_failures)}/{len(suffix_rows)} checkpoints changed prefix logits when only suffix changed.",
        f"2. Full forward vs incremental: {'NOT CONSISTENT' if incremental_failures else 'consistent within tolerance'}. "
        f"{len(incremental_failures)}/{len(incremental_rows)} tested positions exceeded 1e-3 max-logit diff.",
        f"3. Global KV role: {takeover_notes}",
        "4. Priority: if P0.1/P0.2 fail, fix causal/cache interface before treating this as regular overfitting.",
        "",
        "## Best cached checkpoints by s600 reasoning",
        "",
    ]
    if best_reason:
        lines.append("| rank | family | checkpoint | s600 exact | s600 teacher | public s200 | clean val ppl | global mass |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for index, row in enumerate(best_reason[:8], 1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        str(row["family"]),
                        str(row["checkpoint"]),
                        _fmt(row.get("cached_s600_reason_exact")),
                        _fmt(row.get("cached_s600_teacher_acc")),
                        _fmt(row.get("cached_public_s200_avg")),
                        _fmt(row.get("perplexity")),
                        _fmt(row.get("global_mass")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `p0_suffix_invariance.csv`",
            "- `p0_full_vs_incremental.csv`",
            "- `p0_global_sweep.csv`",
            "- `p0_memory_intervention.csv`",
            "- `p1_norm_audit.csv`",
            "- `p1_route_intervention.csv`",
            "- `checkpoint_selection_matrix.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _global_takeover_notes(sweep_rows: list[dict[str, Any]], memory_rows: list[dict[str, Any]]) -> str:
    notes = []
    for family in ["hidden", "attention"]:
        defaults = {
            row["checkpoint"]: row
            for row in sweep_rows
            if row["family"] == family and row["intervention"] == "default" and float(row["global_scale"]) == 1.0
        }
        offs = {
            row["checkpoint"]: row
            for row in sweep_rows
            if row["family"] == family and row["intervention"] == "default" and float(row["global_scale"]) == 0.0
        }
        recoveries = []
        for checkpoint, base in defaults.items():
            off = offs.get(checkpoint)
            if not off:
                continue
            base_reason = _num_or_none(base.get("reason_exact"))
            off_reason = _num_or_none(off.get("reason_exact"))
            if base_reason is not None and off_reason is not None and off_reason > base_reason:
                recoveries.append(checkpoint)
        if recoveries:
            notes.append(f"{family} improves under global-off at {', '.join(recoveries)}")
    return "; ".join(notes) if notes else "no strong global-off recovery in the lightweight diagnostics"


def _global_parameter_norms(model: Any, family: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if family == "hidden":
        modules = model.global_read if isinstance(model.global_read, torch.nn.ModuleList) else [model.global_read]
        writes = model.global_write if isinstance(model.global_write, torch.nn.ModuleList) else [model.global_write]
        row["global_read_query_norm_max"] = max(_param_norm(module.query.weight) for module in modules if module is not None)
        row["global_read_out_norm_max"] = max(_param_norm(module.out.weight) for module in modules if module is not None)
        row["global_read_gate_max"] = max(float(torch.sigmoid(module.gate).detach().cpu()) for module in modules if module is not None)
        row["global_write_proj_norm_max"] = max(_param_norm(module.proj.weight) for module in writes if module is not None)
    else:
        biases = []
        key_read = []
        key_write = []
        for module in model.modules():
            if isinstance(module, CausalSelfAttention):
                if module.global_logit_bias is not None:
                    biases.append(float(module.global_logit_bias.detach().cpu()))
                if module.global_key_read is not None:
                    key_read.append(_param_norm(module.global_key_read.weight))
                if module.global_key_write is not None:
                    key_write.append(_param_norm(module.global_key_write.weight))
        row["attention_global_logit_bias_max"] = max(biases) if biases else None
        row["attention_global_logit_bias_mean"] = sum(biases) / len(biases) if biases else None
        row["attention_global_key_read_norm_max"] = max(key_read) if key_read else None
        row["attention_global_key_write_norm_max"] = max(key_write) if key_write else None
    return row


def _param_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu())


def _route_logits_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    logits_a = a.get("route_logits") or []
    logits_b = b.get("route_logits") or []
    diffs = []
    for left, right in zip(logits_a, logits_b):
        diffs.append((left - right).float().abs())
    if not diffs:
        return {"max": None, "mean": None}
    flat = torch.cat([diff.flatten() for diff in diffs])
    return {"max": _float(flat.max()), "mean": _float(flat.mean())}


def _selected_actions(route_info: dict[str, Any]) -> list[int]:
    actions = route_info.get("selected_actions") or []
    values = []
    for tensor in actions:
        values.append(int(tensor.detach().flatten()[0].cpu()))
    return values


def _action_mismatch_rate(left: list[int], right: list[int]) -> float | None:
    if not left or not right:
        return None
    count = min(len(left), len(right))
    return sum(int(left[i] != right[i]) for i in range(count)) / count


def _route_entropy(route_info: dict[str, Any]) -> float | None:
    probs = route_info.get("route_probs") or []
    if not probs:
        return None
    values = []
    for prob in probs:
        p = prob.float().clamp_min(1e-9)
        values.append(float((-(p * p.log()).sum(dim=-1)).mean().detach().cpu()))
    return _mean(values)


def _most_used_block(route_info: dict[str, Any], block_count: int) -> int | None:
    actions = route_info.get("selected_actions") or []
    counts = [0 for _ in range(block_count)]
    for tensor in actions:
        for value in tensor.detach().cpu().flatten().tolist():
            if 0 <= int(value) < block_count:
                counts[int(value)] += 1
    if not counts or max(counts) <= 0:
        return None
    return int(max(range(len(counts)), key=lambda index: counts[index]))


def _merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for summary in summaries for key in summary})
    merged: dict[str, Any] = {}
    for key in keys:
        values = [_num_or_none(summary.get(key)) for summary in summaries]
        values = [value for value in values if value is not None]
        if values:
            merged[key] = sum(values) / len(values)
    return merged


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in summary.items():
        number = _num_or_none(value)
        if number is not None:
            out[f"summary_{key}"] = number
    return out


def _summary_global_mass(summary: dict[str, Any]) -> float:
    for key in ["global_attention_mass_mean", "attention_global_kv_last_token_mass"]:
        value = _num_or_none(summary.get(key))
        if value is not None:
            return value
    return 0.0


def _scheduled_value(schedule: list[dict[str, Any]], step: int, key: str, default: float) -> float:
    value = default
    for item in schedule:
        if step <= int(item.get("max_step", step)):
            return float(item.get(key, value))
        value = float(item.get(key, value))
    return value


def _context_length(config: dict[str, Any]) -> int:
    model_config = config.get("model_config_resolved", {})
    base = model_config.get("base") if isinstance(model_config.get("base"), dict) else {}
    return int(model_config.get("context_length") or base.get("context_length") or 2048)


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos = getattr(tokenizer, "bos_token_id", None)
    return ([int(bos)] if bos is not None else []) + ids


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def _num_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu())
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: list[Any]) -> float | None:
    numbers = [_num_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _fmt(value: Any) -> str:
    number = _num_or_none(value)
    if number is None:
        return ""
    return f"{number:.4f}"


if __name__ == "__main__":
    main()
