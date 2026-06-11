from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.difficulty_report import _checkpoint_dir, _load_model_for_run, causal_lm_sample_losses
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def make_fixed_route_stability_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int = 8,
    checkpoint: str = "checkpoint_best",
    device_name: str = "auto",
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for fixed-route stability reports.")

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    data_config = config.get("data_config_resolved")
    model_config = config.get("model_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")
    if not isinstance(model_config, dict):
        raise ValueError("Run config must include model_config_resolved.")

    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    effective_batch_size = int(batch_size or config.get("batch_size", 1))
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=effective_batch_size,
        shuffle=False,
    )
    vocab_size = _model_vocab_size(model, model_config)
    losses: list[float] = []
    routing_summaries: list[dict[str, Any]] = []
    fixed_route_match = True
    logits_finite = True
    logits_shape_matches = True
    batch_count = 0
    sample_count = 0
    token_count = 0
    max_abs_logit: float | None = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = batch.to(device)
            output = model(
                batch,
                targets=batch,
                route_mode="fixed",
                pseudo_policy=str(config.get("routing", {}).get("pseudo_policy", "sequential")),
            )
            logits = output["logits"]
            expected_shape = (batch.size(0), batch.size(1), vocab_size)
            logits_shape_matches = logits_shape_matches and tuple(logits.shape) == expected_shape
            logits_finite = logits_finite and bool(torch.isfinite(logits).all().item())
            max_value = float(logits.detach().abs().max().cpu())
            max_abs_logit = max_value if max_abs_logit is None else max(max_abs_logit, max_value)
            sample_losses = causal_lm_sample_losses(logits, batch)
            losses.extend(float(value) for value in sample_losses.detach().cpu().tolist())
            routing_summaries.append(dict(output.get("routing_summary", {})))
            fixed_route_match = fixed_route_match and _route_matches_targets(output.get("route_info", {}))
            batch_count += 1
            sample_count += int(batch.size(0))
            token_count += int(batch.numel())

    routing_summary = _mean_numeric_summaries(routing_summaries)
    mean_loss = _mean(losses)
    checks = {
        "forward_completed": batch_count > 0,
        "logits_shape_matches": logits_shape_matches,
        "logits_finite": logits_finite,
        "sample_losses_finite": all(math.isfinite(value) for value in losses) if losses else False,
        "fixed_route_matches_targets": fixed_route_match,
        "route_imitation_accuracy_is_one": routing_summary.get("route_imitation_accuracy") == 1.0,
        "position_norm_finite": _finite(routing_summary.get("position_norm_mean")),
        "routing_summary_finite": _all_finite(routing_summary),
    }
    report = {
        "run_dir": str(run_dir),
        "split": split,
        "batch_size": effective_batch_size,
        "max_batches": max_batches,
        "checkpoint": str(_checkpoint_dir(run_dir, checkpoint)),
        "batch_count": batch_count,
        "sample_count": sample_count,
        "token_count": token_count,
        "mean_sample_cross_entropy": mean_loss,
        "max_abs_logit": max_abs_logit,
        "routing_summary": routing_summary,
        "checks": checks,
        "overall_status": "pass" if all(checks.values()) else "fail",
    }
    if output_path is None:
        output_path = run_dir / "fixed_route_stability_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _route_matches_targets(route_info: dict[str, Any]) -> bool:
    actions = route_info.get("selected_actions") or []
    targets = route_info.get("route_targets") or []
    if not actions or len(actions) != len(targets):
        return False
    for selected, target in zip(actions, targets):
        if not bool(torch.equal(selected.detach().cpu(), target.detach().cpu())):
            return False
    return True


def _mean_numeric_summaries(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if _finite(value):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def _model_vocab_size(model: Any, model_config: dict[str, Any]) -> int:
    if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "num_embeddings"):
        return int(model.token_embedding.num_embeddings)
    if isinstance(model_config.get("base"), dict):
        return int(model_config["base"]["vocab_size"])
    return int(model_config["vocab_size"])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _all_finite(values: dict[str, float]) -> bool:
    return bool(values) and all(_finite(value) for value in values.values())


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _device(name: str) -> "torch.device":
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
