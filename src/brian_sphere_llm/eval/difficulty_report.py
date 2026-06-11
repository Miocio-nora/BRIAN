from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.difficulty import summarize_difficulty_samples
from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.routing.schedule import scheduled_value
from brian_sphere_llm.train.checkpoint import load_checkpoint
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json, write_jsonl

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_baseline_difficulty_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    sample_output_path: str | Path | None = None,
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int = 8,
    device_name: str = "auto",
    checkpoint: str = "checkpoint_best",
    difficulty_bins: list[str] | None = None,
) -> Path:
    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for baseline difficulty reports.")

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    data_config = config.get("data_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")

    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    effective_batch_size = _effective_batch_size(batch_size, config)
    effective_max_batches = _int_value(max_batches, "max_batches", minimum=1)
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=effective_batch_size,
        shuffle=False,
    )
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= effective_max_batches:
                break
            batch = batch.to(device)
            outputs = model(batch)
            losses = causal_lm_sample_losses(outputs["logits"], batch)
            offset = batch_index * effective_batch_size
            for sample_index in range(batch.size(0)):
                samples.append(
                    {
                        "sample_id": offset + sample_index,
                        "baseline_cross_entropy": float(losses[sample_index].detach().cpu()),
                    }
                )

    bins = difficulty_bins or ["easy", "medium", "hard"]
    _assign_difficulty_bins(samples, bins)
    if output_path is None:
        output_path = run_dir / "baseline_difficulty_report.json"
    output_path = Path(output_path)
    if sample_output_path is None:
        sample_output_path = output_path.with_name(output_path.stem + "_samples.jsonl")
    sample_output_path = Path(sample_output_path)
    _write_jsonl(samples, sample_output_path)
    summary = {
        **_summarize_baseline_difficulty_samples(samples, bins),
        "run_dir": str(run_dir),
        "split": split,
        "batch_size": effective_batch_size,
        "max_batches": effective_max_batches,
        "checkpoint": str(_checkpoint_dir(run_dir, checkpoint)),
        "samples_path": str(sample_output_path),
    }
    write_json(summary, output_path)
    return output_path


def make_difficulty_report(
    baseline_run: str | Path,
    routed_run: str | Path,
    *,
    output_path: str | Path | None = None,
    sample_output_path: str | Path | None = None,
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int = 8,
    device_name: str = "auto",
    baseline_checkpoint: str = "checkpoint_best",
    routed_checkpoint: str = "checkpoint_best",
) -> Path:
    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for difficulty reports.")

    baseline_run = Path(baseline_run)
    routed_run = Path(routed_run)
    baseline_config = load_config(baseline_run / "config_resolved.yaml")
    routed_config = load_config(routed_run / "config_resolved.yaml")
    data_config = routed_config.get("data_config_resolved") or baseline_config.get("data_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")

    device = _device(device_name)
    baseline_model = _load_model_for_run(baseline_run, baseline_checkpoint, device)
    routed_model = _load_model_for_run(routed_run, routed_checkpoint, device)
    baseline_model.eval()
    routed_model.eval()

    effective_batch_size = _effective_batch_size(
        batch_size,
        routed_config,
        fallback_config=baseline_config,
    )
    effective_max_batches = _int_value(max_batches, "max_batches", minimum=1)
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=effective_batch_size,
        shuffle=False,
    )
    routed_mode = train_mode_for_stage(str(routed_config["stage"]))
    routed_step = _checkpoint_step(routed_run, routed_checkpoint)
    samples: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= effective_max_batches:
                break
            batch = batch.to(device)
            baseline_outputs = baseline_model(batch)
            routed_outputs = _forward_routed_for_eval(
                routed_model,
                batch,
                config=routed_config,
                route_mode=routed_mode,
                global_step=routed_step,
            )
            baseline_losses = causal_lm_sample_losses(baseline_outputs["logits"], batch)
            routed_losses = causal_lm_sample_losses(routed_outputs["logits"], batch)
            steps = route_steps_per_sample(
                routed_outputs.get("route_info", {}),
                batch_size=batch.size(0),
                out_action=getattr(routed_model, "out_action", None),
            )
            out_probs = output_probability_per_sample(
                routed_outputs.get("route_info", {}),
                out_action=getattr(routed_model, "out_action", None),
                batch_size=batch.size(0),
            )
            offset = batch_index * effective_batch_size
            for sample_index in range(batch.size(0)):
                row: dict[str, float | int] = {
                    "sample_id": offset + sample_index,
                    "baseline_cross_entropy": float(baseline_losses[sample_index].detach().cpu()),
                    "routed_cross_entropy": float(routed_losses[sample_index].detach().cpu()),
                    "route_steps": int(steps[sample_index]),
                }
                if out_probs[sample_index] is not None:
                    row["p_output_mean"] = float(out_probs[sample_index])
                samples.append(row)

    if output_path is None:
        output_path = routed_run / "difficulty_step_report.json"
    output_path = Path(output_path)
    if sample_output_path is None:
        sample_output_path = output_path.with_name(output_path.stem + "_samples.jsonl")
    sample_output_path = Path(sample_output_path)
    _write_jsonl(samples, sample_output_path)
    summary = {
        **summarize_difficulty_samples(samples),
        "baseline_run": str(baseline_run),
        "routed_run": str(routed_run),
        "split": split,
        "batch_size": effective_batch_size,
        "max_batches": effective_max_batches,
        "baseline_checkpoint": str(_checkpoint_dir(baseline_run, baseline_checkpoint)),
        "routed_checkpoint": str(_checkpoint_dir(routed_run, routed_checkpoint)),
        "routed_eval_step": routed_step,
        "samples_path": str(sample_output_path),
    }
    write_json(summary, output_path)
    return output_path


def causal_lm_sample_losses(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    )
    return token_losses.reshape(targets.size(0), -1).mean(dim=1)


def route_steps_per_sample(route_info: dict[str, Any], *, batch_size: int, out_action: int | None = None) -> list[int]:
    actions = route_info.get("selected_actions") or []
    if not actions:
        return [0 for _ in range(batch_size)]
    if out_action is not None:
        stacked_actions = torch.stack(actions).detach().cpu()
        return [int((stacked_actions[:, sample_index] != out_action).sum().item()) for sample_index in range(batch_size)]
    exit_flags = route_info.get("exit_flags") or []
    if not exit_flags:
        return [len(actions) for _ in range(batch_size)]
    stacked = torch.stack(exit_flags).detach().cpu().bool()
    steps: list[int] = []
    for sample_index in range(batch_size):
        hits = torch.nonzero(stacked[:, sample_index], as_tuple=False)
        steps.append(int(hits[0].item() + 1) if hits.numel() else len(actions))
    return steps


def output_probability_per_sample(route_info: dict[str, Any], *, out_action: int | None, batch_size: int) -> list[float | None]:
    probs = route_info.get("route_probs") or []
    if out_action is None or not probs:
        return [None for _ in range(batch_size)]
    stacked = torch.stack(probs).detach().cpu()
    values = stacked[..., out_action].mean(dim=0)
    return [float(value) for value in values]


def _assign_difficulty_bins(samples: list[dict[str, Any]], bins: list[str]) -> None:
    if not samples or not bins:
        return
    ranked_indexes = sorted(
        range(len(samples)),
        key=lambda index: (float(samples[index]["baseline_cross_entropy"]), int(samples[index]["sample_id"])),
    )
    for rank, sample_index in enumerate(ranked_indexes):
        bin_index = min(len(bins) - 1, rank * len(bins) // len(samples))
        samples[sample_index]["difficulty_bin"] = bins[bin_index]


def _summarize_baseline_difficulty_samples(samples: list[dict[str, Any]], bins: list[str]) -> dict[str, Any]:
    losses = [float(sample["baseline_cross_entropy"]) for sample in samples]
    by_difficulty = {label: _baseline_bin_summary(samples, label) for label in bins}
    nonempty_bin_count = sum(1 for summary in by_difficulty.values() if summary["sample_count"] > 0)
    return {
        "sample_count": len(samples),
        "difficulty_bins": bins,
        "difficulty_bin_count": nonempty_bin_count,
        "mean_baseline_cross_entropy": _mean(losses),
        "min_baseline_cross_entropy": min(losses) if losses else None,
        "max_baseline_cross_entropy": max(losses) if losses else None,
        "by_difficulty": by_difficulty,
    }


def _baseline_bin_summary(samples: list[dict[str, Any]], label: str) -> dict[str, float | int | None]:
    losses = [float(sample["baseline_cross_entropy"]) for sample in samples if sample.get("difficulty_bin") == label]
    return {
        "sample_count": len(losses),
        "mean_baseline_cross_entropy": _mean(losses),
        "min_baseline_cross_entropy": min(losses) if losses else None,
        "max_baseline_cross_entropy": max(losses) if losses else None,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _forward_routed_for_eval(
    model: Any,
    batch: "torch.Tensor",
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
) -> dict[str, Any]:
    if route_mode == "baseline":
        return model(batch)
    routing_cfg = _mapping_config(config, "routing")
    loss_weights = dict(_mapping_config(config, "loss_weights"))
    router_probability = None
    if route_mode == "scheduled":
        schedule = routing_cfg.get("schedule", [])
        router_probability = scheduled_value(schedule, global_step, "router_probability", 0.0)
        loss_weights["route"] = scheduled_value(schedule, global_step, "lambda_route", loss_weights.get("route", 0.0))
    return model(
        batch,
        targets=None,
        route_mode=route_mode,
        pseudo_policy=str(routing_cfg.get("pseudo_policy", "sequential")),
        loss_weights=loss_weights,
        hard_exit=_bool_mapping_value(
            routing_cfg,
            "hard_exit",
            default=str(config.get("stage")) == "stage4_output_action",
            name="routing.hard_exit",
        ),
        router_probability=router_probability,
        global_step=global_step,
    )


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


def _mapping_config(config: dict[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _effective_batch_size(
    batch_size: int | None,
    config: dict[str, Any],
    *,
    fallback_config: dict[str, Any] | None = None,
) -> int:
    if batch_size is not None:
        return _int_value(batch_size, "batch_size", minimum=1)
    value = config.get("batch_size")
    if value is None and fallback_config is not None:
        value = fallback_config.get("batch_size")
    if value is None:
        value = 1
    return _int_value(value, "batch_size", minimum=1)


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


def _load_model_for_run(run_dir: Path, checkpoint: str, device: "torch.device") -> Any:
    model = _build_model_from_run(run_dir)
    load_checkpoint(_checkpoint_dir(run_dir, checkpoint), model=model)
    model.to(device)
    return model


def _build_model_from_run(run_dir: Path) -> Any:
    config = load_config(run_dir / "config_resolved.yaml")
    model_config = dict(config.get("model_config_resolved") or {})
    architecture = model_config.get("architecture")
    if architecture == "decoder_only_llama_like":
        return BaselineLM(BaselineConfig.from_dict(model_config))
    if architecture == "brian_route_core":
        if "base" not in model_config:
            model_config["base"] = _resolve_brian_base_config(config, model_config)
        return BrianRouteCore(BrianRouteConfig.from_dict(model_config))
    raise ValueError(f"Unknown run architecture: {architecture}")


def _resolve_brian_base_config(run_config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    base_config = str(model_config.get("base_config", ""))
    candidates: list[Path] = []
    model_path = run_config.get("model_config")
    if model_path:
        candidates.append((REPO_ROOT / "configs" / "train" / str(model_path)).resolve().parent / base_config)
    if base_config:
        candidates.extend(
            [
                (REPO_ROOT / "configs" / "model" / base_config).resolve(),
                Path(base_config).resolve(),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return load_config(candidate)
    raise ValueError(f"Could not resolve Brian base_config: {base_config}")


def _checkpoint_dir(run_dir: Path, checkpoint: str) -> Path:
    aliases = {
        "best": "checkpoint_best",
        "latest": "checkpoint_latest",
    }
    name = aliases.get(checkpoint, checkpoint)
    candidate = run_dir / name
    if (candidate / "state.pt").exists():
        return candidate
    fallback_names = ["checkpoint_best", "checkpoint_latest"]
    for fallback in fallback_names:
        fallback_path = run_dir / fallback
        if (fallback_path / "state.pt").exists():
            return fallback_path
    raise FileNotFoundError(f"No checkpoint state.pt found in {run_dir}")


def _checkpoint_step(run_dir: Path, checkpoint: str) -> int:
    if torch is None:
        return 0
    payload = torch.load(_checkpoint_dir(run_dir, checkpoint) / "state.pt", map_location="cpu")
    return int(payload.get("step", 0))


def _device(name: str) -> "torch.device":
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    write_jsonl(rows, path)
