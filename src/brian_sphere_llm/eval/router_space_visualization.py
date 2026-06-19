from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json
import math

import numpy as np

from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.eval.difficulty_report import (
    _checkpoint_dir,
    _checkpoint_step,
    _device,
    _effective_batch_size,
    _load_model_for_run,
)
from brian_sphere_llm.eval.route_path_visualization import _forward_routed_for_visualization
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def make_router_space_visualization(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    checkpoint: str = "checkpoint_latest",
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int = 4,
    device_name: str = "auto",
    max_points: int = 2048,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for router-space visualization.")
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    route_mode = train_mode_for_stage(str(config["stage"]))
    if route_mode == "baseline":
        raise ValueError("Router-space visualization requires a routed run, not a baseline run.")
    data_config = config.get("data_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")

    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=_effective_batch_size(batch_size, config),
        shuffle=False,
    )
    routed_step = _checkpoint_step(run_dir, checkpoint)
    payload: dict[str, Any] = {"records": [], "source": "checkpoint", "out_action": None}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max(1, int(max_batches)):
                break
            output = _forward_routed_for_visualization(
                model,
                batch.to(device),
                config=config,
                route_mode=route_mode,
                global_step=routed_step,
                collect_router_space=True,
            )
            batch_payload = output.get("router_space", {})
            if payload["out_action"] is None:
                payload["out_action"] = batch_payload.get("out_action")
            for record in batch_payload.get("records", []):
                row = dict(record)
                row["batch_index"] = batch_index
                payload["records"].append(row)

    output_path = Path(output_path or run_dir / "router_space_visualization.html")
    return make_router_space_visualization_from_payload(
        payload,
        model,
        output_path=output_path,
        step=routed_step,
        max_points=max_points,
        metadata={
            "run_dir": str(run_dir),
            "source": "checkpoint",
            "checkpoint": str(_checkpoint_dir(run_dir, checkpoint)),
            "split": split,
            "max_batches": max_batches,
        },
    )


def make_router_space_visualization_from_payload(
    payload: MappingLike,
    model: Any,
    *,
    output_path: str | Path,
    step: int | None = None,
    max_points: int = 2048,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for router-space visualization.")
    output_path = Path(output_path)
    sidecar_path = output_path.with_suffix(".json")
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Router-space visualization requires at least one collected router record.")
    experts = _expert_vectors(model)
    bias = _expert_bias(model)
    arrays = _records_to_arrays(records, out_action=_out_action(payload, experts.shape[0]))
    sampled_indexes = _sample_indexes(arrays["embeddings"].shape[0], max_points)
    sampled = {
        key: value[sampled_indexes]
        for key, value in arrays.items()
        if isinstance(value, np.ndarray) and value.shape[0] == arrays["embeddings"].shape[0]
    }
    nodes, points, projection = _project_router_space(
        sampled["embeddings"],
        experts,
        selected_actions=sampled["selected_actions"],
        raw_top_actions=sampled["raw_top_actions"],
        effective_top_actions=sampled["effective_top_actions"],
        step_indices=sampled["step_indices"],
        max_probs=sampled["max_probs"],
        margins=sampled["margins"],
        out_action=arrays["out_action"],
    )
    report: dict[str, Any] = {
        "html_path": str(output_path),
        "sidecar_json": str(sidecar_path),
        "step": step,
        "metadata": metadata or {},
        "projection": projection,
        "experts": nodes,
        "points": points,
        "metrics": _router_space_metrics(arrays, experts, bias),
        "counts": {
            "total_points": int(arrays["embeddings"].shape[0]),
            "plotted_points": int(len(points)),
            "records": len(records),
        },
    }
    report["checks"] = {
        "experts_present": bool(nodes),
        "points_present": bool(points),
        "not_single_selected_action": report["metrics"]["selected_action_unique_count"] > 1,
        "not_single_raw_top_action": report["metrics"]["raw_top_action_unique_count"] > 1,
        "router_entropy_nonzero": report["metrics"]["effective_entropy_mean"] > 1e-4,
    }
    report["overall_status"] = "pass" if all(report["checks"].values()) else "warn"
    write_json(report, sidecar_path)
    _write_html(report, output_path)
    return output_path


MappingLike = dict[str, Any]


def _expert_vectors(model: Any) -> np.ndarray:
    router = getattr(model, "router", None)
    if router is None:
        raise ValueError("Model does not expose a router.")
    if hasattr(router, "expert_vectors"):
        values = router.expert_vectors()
    else:
        values = router.net[-1].weight
    return values.detach().float().cpu().numpy()


def _expert_bias(model: Any) -> np.ndarray:
    router = getattr(model, "router", None)
    bias = getattr(router.net[-1], "bias", None)
    if bias is None:
        return np.zeros(_expert_vectors(model).shape[0], dtype=np.float64)
    return bias.detach().float().cpu().numpy()


def _out_action(payload: MappingLike, action_count: int) -> int:
    value = payload.get("out_action")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return action_count - 1


def _records_to_arrays(records: list[MappingLike], *, out_action: int) -> dict[str, np.ndarray | int]:
    embeddings = []
    raw_logits = []
    effective_logits = []
    probs = []
    selected_actions = []
    step_indices = []
    batch_indices = []
    sample_indices = []
    for record_index, record in enumerate(records):
        embedding = _tensor_to_numpy(record["embedding"])
        batch = embedding.shape[0]
        embeddings.append(embedding)
        raw_logits.append(_tensor_to_numpy(record["raw_logits"]))
        effective_logits.append(_tensor_to_numpy(record["effective_logits"]))
        probs.append(_tensor_to_numpy(record["probs"]))
        selected_actions.append(_tensor_to_numpy(record["selected_actions"]).astype(np.int64))
        step_indices.append(np.full(batch, int(record.get("step", record_index)), dtype=np.int64))
        batch_indices.append(np.full(batch, int(record.get("batch_index", 0)), dtype=np.int64))
        sample_indices.append(np.arange(batch, dtype=np.int64))
    embedding_array = np.concatenate(embeddings, axis=0)
    raw_array = np.concatenate(raw_logits, axis=0)
    effective_array = np.concatenate(effective_logits, axis=0)
    prob_array = np.concatenate(probs, axis=0)
    selected_array = np.concatenate(selected_actions, axis=0)
    raw_top = raw_array.argmax(axis=1).astype(np.int64)
    effective_top = effective_array.argmax(axis=1).astype(np.int64)
    sorted_probs = np.sort(prob_array, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else sorted_probs[:, -1]
    return {
        "embeddings": embedding_array,
        "raw_logits": raw_array,
        "effective_logits": effective_array,
        "probs": prob_array,
        "selected_actions": selected_array,
        "raw_top_actions": raw_top,
        "effective_top_actions": effective_top,
        "step_indices": np.concatenate(step_indices, axis=0),
        "batch_indices": np.concatenate(batch_indices, axis=0),
        "sample_indices": np.concatenate(sample_indices, axis=0),
        "max_probs": prob_array.max(axis=1),
        "margins": margins,
        "out_action": out_action,
    }


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sample_indexes(total: int, max_points: int) -> np.ndarray:
    if total <= max(1, int(max_points)):
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=max(1, int(max_points)), dtype=np.int64)


def _project_router_space(
    embeddings: np.ndarray,
    experts: np.ndarray,
    *,
    selected_actions: np.ndarray,
    raw_top_actions: np.ndarray,
    effective_top_actions: np.ndarray,
    step_indices: np.ndarray,
    max_probs: np.ndarray,
    margins: np.ndarray,
    out_action: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    combined = np.concatenate([embeddings, experts], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    if centered.shape[1] == 0:
        coords = np.zeros((combined.shape[0], 2), dtype=np.float64)
        explained = [0.0, 0.0]
    else:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        dims = min(2, vt.shape[0])
        coords = centered @ vt[:dims].T if dims else np.zeros((combined.shape[0], 0), dtype=np.float64)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        variance = singular_values**2
        total = float(variance.sum())
        explained = [float(value / total) if total > 0.0 else 0.0 for value in variance[:2]]
        while len(explained) < 2:
            explained.append(0.0)
    point_coords = coords[: embeddings.shape[0]]
    expert_coords = coords[embeddings.shape[0] :]
    nodes = []
    for action, coord in enumerate(expert_coords):
        nodes.append(
            {
                "action": int(action),
                "label": "OUT" if action == out_action else f"B{action}",
                "kind": "out" if action == out_action else "block",
                "x": float(coord[0]),
                "y": float(coord[1]),
                "norm": float(np.linalg.norm(experts[action])),
            }
        )
    points = []
    for index, coord in enumerate(point_coords):
        selected = int(selected_actions[index])
        points.append(
            {
                "index": int(index),
                "step": int(step_indices[index]),
                "selected_action": selected,
                "selected_label": "OUT" if selected == out_action else f"B{selected}",
                "raw_top_action": int(raw_top_actions[index]),
                "effective_top_action": int(effective_top_actions[index]),
                "max_prob": float(max_probs[index]),
                "margin": float(margins[index]),
                "x": float(coord[0]),
                "y": float(coord[1]),
            }
        )
    return nodes, points, {"method": "pca", "explained_variance_ratio": explained}


def _router_space_metrics(arrays: dict[str, Any], experts: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
    action_count = experts.shape[0]
    out_action = int(arrays["out_action"])
    selected = arrays["selected_actions"].astype(np.int64)
    raw_top = arrays["raw_top_actions"].astype(np.int64)
    effective_top = arrays["effective_top_actions"].astype(np.int64)
    probs = arrays["probs"]
    metrics: dict[str, Any] = {
        "selected_action_counts": _count_actions(selected, action_count),
        "raw_top_action_counts": _count_actions(raw_top, action_count),
        "effective_top_action_counts": _count_actions(effective_top, action_count),
        "selected_action_unique_count": int(np.unique(selected).size),
        "raw_top_action_unique_count": int(np.unique(raw_top).size),
        "effective_top_action_unique_count": int(np.unique(effective_top).size),
        "selected_domination_fraction": _dominant_fraction(selected),
        "raw_top_domination_fraction": _dominant_fraction(raw_top),
        "effective_top_domination_fraction": _dominant_fraction(effective_top),
        "mean_action_probability": {str(i): float(probs[:, i].mean()) for i in range(action_count)},
        "effective_entropy_mean": _entropy(probs),
        "max_probability_mean": float(probs.max(axis=1).mean()),
        "probability_margin_mean": float(arrays["margins"].mean()),
        "expert_norms": {str(i): float(np.linalg.norm(experts[i])) for i in range(action_count)},
        "expert_bias": {str(i): float(bias[i]) for i in range(action_count)},
        "expert_geometry": _expert_geometry(experts),
        "self_recur_ratio": _self_recur_ratio(
            selected,
            arrays["step_indices"].astype(np.int64),
            arrays["batch_indices"].astype(np.int64),
            arrays["sample_indices"].astype(np.int64),
            out_action=out_action,
        ),
        "out_action": out_action,
    }
    metrics["dead_selected_actions"] = [
        int(action) for action, count in metrics["selected_action_counts"].items() if int(count) == 0
    ]
    metrics["dead_raw_top_actions"] = [
        int(action) for action, count in metrics["raw_top_action_counts"].items() if int(count) == 0
    ]
    return metrics


def _count_actions(actions: np.ndarray, action_count: int) -> dict[str, int]:
    counts = np.bincount(actions, minlength=action_count)
    return {str(index): int(counts[index]) for index in range(action_count)}


def _dominant_fraction(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 0.0
    counts = Counter(int(value) for value in actions.tolist())
    return max(counts.values()) / max(1, actions.size)


def _entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float((-(clipped * np.log(clipped)).sum(axis=1)).mean())


def _expert_geometry(experts: np.ndarray) -> dict[str, float]:
    if experts.shape[0] <= 1:
        return {"min_euclidean_distance": 0.0, "mean_euclidean_distance": 0.0, "max_cosine_similarity": 1.0}
    distances = []
    cosines = []
    norms = np.linalg.norm(experts, axis=1, keepdims=True)
    normalized = experts / np.clip(norms, 1e-12, None)
    for i in range(experts.shape[0]):
        for j in range(i + 1, experts.shape[0]):
            distances.append(float(np.linalg.norm(experts[i] - experts[j])))
            cosines.append(float(np.dot(normalized[i], normalized[j])))
    return {
        "min_euclidean_distance": float(min(distances)),
        "mean_euclidean_distance": float(sum(distances) / len(distances)),
        "max_cosine_similarity": float(max(cosines)),
        "mean_cosine_similarity": float(sum(cosines) / len(cosines)),
    }


def _self_recur_ratio(
    selected: np.ndarray,
    step_indices: np.ndarray,
    batch_indices: np.ndarray,
    sample_indices: np.ndarray,
    *,
    out_action: int,
) -> float:
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    repeat = 0
    denom = 0
    order = np.lexsort((step_indices, sample_indices, batch_indices))
    for index in order:
        key = (int(batch_indices[index]), int(sample_indices[index]))
        step = int(step_indices[index])
        action = int(selected[index])
        last = previous.get(key)
        if last is not None and step > last[0]:
            denom += 1
            if action == last[1] and action != out_action:
                repeat += 1
        previous[key] = (step, action)
    return repeat / max(1, denom)


def _write_html(report: dict[str, Any], output_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Plotly is required for router-space visualization. Install plotly>=5.22.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = report["points"]
    experts = report["experts"]
    fig = go.Figure()
    actions = sorted({int(point["selected_action"]) for point in points})
    for action in actions:
        rows = [point for point in points if int(point["selected_action"]) == action]
        label = rows[0]["selected_label"] if rows else str(action)
        fig.add_trace(
            go.Scattergl(
                x=[row["x"] for row in rows],
                y=[row["y"] for row in rows],
                mode="markers",
                name=f"selected {label}",
                marker={"size": 5, "opacity": 0.55},
                text=[
                    f"step={row['step']} selected={row['selected_label']} raw_top={row['raw_top_action']} "
                    f"eff_top={row['effective_top_action']} p={row['max_prob']:.4f} margin={row['margin']:.4f}"
                    for row in rows
                ],
                hoverinfo="text",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[row["x"] for row in experts],
            y=[row["y"] for row in experts],
            mode="markers+text",
            name="router expert vectors",
            marker={"size": 15, "symbol": "star", "color": "black"},
            text=[row["label"] for row in experts],
            textposition="top center",
            hovertext=[f"{row['label']} norm={row['norm']:.4f}" for row in experts],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Router Space PCA: Hidden+Position Embeddings vs Expert Vectors",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=760,
    )
    plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    summary = {
        "overall_status": report["overall_status"],
        "checks": report["checks"],
        "counts": report["counts"],
        "metrics": report["metrics"],
        "projection": report["projection"],
    }
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Router Space Visualization</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;} pre{background:#f6f8fa;padding:12px;overflow:auto;}</style>"
        "</head><body><h1>Router Space Visualization</h1>"
        f"<p>step: {report.get('step')}</p>"
        f"{plot_html}"
        "<h2>Diagnostics</h2>"
        f"<pre>{json.dumps(summary, indent=2, sort_keys=True)}</pre>"
        "</body></html>"
    )
    output_path.write_text(html, encoding="utf-8")
