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
    _mapping_config,
)
from brian_sphere_llm.routing.schedule import scheduled_value
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def make_route_path_visualization(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    source: str = "checkpoint",
    checkpoint: str = "checkpoint_latest",
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int = 8,
    device_name: str = "auto",
    projection: str = "pca",
    top_paths: int = 64,
    timeline_max_frames: int = 100,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to load route position embeddings for path visualization.")
    run_dir = Path(run_dir)
    source = str(source)
    if source not in {"checkpoint", "train_log", "both"}:
        raise ValueError("source must be one of: checkpoint, train_log, both.")
    if projection != "pca":
        raise ValueError("route_path_visualization currently supports only projection: pca.")

    output_path = Path(output_path or run_dir / "route_path_visualization.html")
    sidecar_path = output_path.with_suffix(".json")
    config = load_config(run_dir / "config_resolved.yaml")
    device = _device(device_name)
    model = _load_model_for_run(run_dir, checkpoint, device)
    model.eval()
    positions = _route_positions(model)
    nodes, projection_report = _project_nodes(positions)
    out_action = len(nodes) - 1

    aggregates: list[dict[str, Any]] = []
    if source in {"checkpoint", "both"}:
        aggregates.append(
            _checkpoint_aggregate(
                run_dir,
                model,
                config=config,
                checkpoint=checkpoint,
                split=split,
                batch_size=batch_size,
                max_batches=max_batches,
                out_action=out_action,
                device=device,
                max_paths=top_paths,
            )
        )
    if source in {"train_log", "both"}:
        aggregates.append(
            _train_log_aggregate(
                run_dir,
                out_action=out_action,
                timeline_max_frames=timeline_max_frames,
                max_paths=top_paths,
            )
        )

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source": source,
        "checkpoint": str(_checkpoint_dir(run_dir, checkpoint)),
        "html_path": str(output_path),
        "sidecar_json": str(sidecar_path),
        "projection": projection_report,
        "nodes": nodes,
        "aggregates": aggregates,
        "checks": _checks(aggregates),
    }
    report["overall_status"] = "pass" if all(report["checks"].values()) else "warn"
    write_json(report, sidecar_path)
    _write_html(report, output_path, top_paths=top_paths)
    return output_path


def make_route_path_visualization_from_train_log(
    run_dir: str | Path,
    model: Any,
    *,
    output_path: str | Path,
    step: int | None = None,
    top_paths: int = 64,
    timeline_max_frames: int = 100,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to load route position embeddings for path visualization.")
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    sidecar_path = output_path.with_suffix(".json")
    positions = _route_positions(model)
    nodes, projection_report = _project_nodes(positions)
    aggregate = _train_log_aggregate(
        run_dir,
        out_action=len(nodes) - 1,
        timeline_max_frames=timeline_max_frames,
        max_paths=top_paths,
    )
    metadata: dict[str, Any] = {}
    if step is not None:
        metadata["step"] = int(step)
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "source": "train_log",
        "checkpoint": None,
        "html_path": str(output_path),
        "sidecar_json": str(sidecar_path),
        "projection": projection_report,
        "nodes": nodes,
        "aggregates": [aggregate],
        "checks": _checks([aggregate]),
        **metadata,
    }
    report["overall_status"] = "pass" if all(report["checks"].values()) else "warn"
    write_json(report, sidecar_path)
    _write_html(report, output_path, top_paths=top_paths)
    return output_path


def _checkpoint_aggregate(
    run_dir: Path,
    model: Any,
    *,
    config: dict[str, Any],
    checkpoint: str,
    split: str,
    batch_size: int | None,
    max_batches: int,
    out_action: int,
    device: "torch.device",
    max_paths: int,
) -> dict[str, Any]:
    data_config = config.get("data_config_resolved")
    if not isinstance(data_config, dict):
        raise ValueError("Run config must include data_config_resolved.")
    route_mode = train_mode_for_stage(str(config["stage"]))
    if route_mode == "baseline":
        raise ValueError("Route path visualization requires a routed run, not a baseline run.")
    effective_batch_size = _effective_batch_size(batch_size, config)
    effective_max_batches = _int_value(max_batches, "max_batches", minimum=1)
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=effective_batch_size,
        shuffle=False,
    )
    step = _checkpoint_step(run_dir, checkpoint)
    path_counts: Counter[tuple[int, ...]] = Counter()
    batch_count = 0
    sample_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= effective_max_batches:
                break
            batch = batch.to(device)
            output = _forward_routed_for_visualization(
                model,
                batch,
                config=config,
                route_mode=route_mode,
                global_step=step,
            )
            for path in _paths_from_route_info(output.get("route_info", {}), batch_size=batch.size(0), out_action=out_action):
                path_counts[tuple(path)] += 1
            batch_count += 1
            sample_count += int(batch.size(0))
    return _aggregate_from_counts(
        name="checkpoint",
        path_counts=path_counts,
        exact=True,
        metadata={
            "split": split,
            "batch_size": effective_batch_size,
            "max_batches": effective_max_batches,
            "batch_count": batch_count,
            "sample_count": sample_count,
            "routed_eval_step": step,
        },
        max_paths=max_paths,
    )


def _train_log_aggregate(
    run_dir: Path,
    *,
    out_action: int,
    timeline_max_frames: int,
    max_paths: int,
) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    selected_indexes = set(_evenly_spaced_indexes(len(rows), max(1, int(timeline_max_frames))))
    cumulative: Counter[tuple[int, ...]] = Counter()
    exact = True
    frame_rows: list[dict[str, Any]] = []
    rows_with_paths = 0
    for row_index, row in enumerate(rows):
        row_counts, row_exact = _path_counts_from_train_row(row, out_action=out_action)
        if row_counts:
            rows_with_paths += 1
        if row_counts and not row_exact:
            exact = False
        cumulative.update(row_counts)
        if row_index in selected_indexes and cumulative:
            frame_rows.append(
                {
                    "step": _safe_int(row.get("step"), default=row_index + 1),
                    **_aggregate_from_counts(
                        name="train_log_frame",
                        path_counts=cumulative,
                        exact=exact,
                        metadata={},
                        max_paths=max_paths,
                    ),
                }
            )
    aggregate = _aggregate_from_counts(
        name="train_log",
        path_counts=cumulative,
        exact=exact,
        metadata={
            "train_row_count": len(rows),
            "rows_with_paths": rows_with_paths,
            "timeline_frame_count": len(frame_rows),
            "path_source": "route_path_counts" if exact else "route_path_examples",
        },
        max_paths=max_paths,
    )
    aggregate["timeline_frames"] = frame_rows
    return aggregate


def _forward_routed_for_visualization(
    model: Any,
    batch: "torch.Tensor",
    *,
    config: dict[str, Any],
    route_mode: str,
    global_step: int,
) -> dict[str, Any]:
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
        routing_constraints=_mapping_config(dict(routing_cfg), "constraints"),
        hard_exit=_bool_mapping_value(
            routing_cfg,
            "hard_exit",
            default=str(config.get("stage")) == "stage4_output_action",
            name="routing.hard_exit",
        ),
        router_probability=router_probability,
        global_step=global_step,
    )


def _route_positions(model: Any) -> np.ndarray:
    table = getattr(model, "position_table", None)
    embeddings = getattr(table, "embeddings", None)
    if embeddings is None:
        raise ValueError("Run model must expose position_table.embeddings.")
    values = embeddings.detach().float().cpu().numpy()
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Route position embeddings must have shape [actions, position_dim].")
    return values


def _project_nodes(positions: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centered = positions - positions.mean(axis=0, keepdims=True)
    if centered.shape[1] == 0:
        coords = np.zeros((centered.shape[0], 3), dtype=np.float64)
        explained = [0.0, 0.0, 0.0]
    else:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        dims = min(3, vt.shape[0])
        coords = centered @ vt[:dims].T if dims else np.zeros((centered.shape[0], 0), dtype=np.float64)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
        variance = singular_values**2
        total = float(variance.sum())
        explained = [float(value / total) if total > 0.0 else 0.0 for value in variance[:3]]
        while len(explained) < 3:
            explained.append(0.0)
    nodes = []
    out_action = positions.shape[0] - 1
    for action, coord in enumerate(coords):
        label = "OUT" if action == out_action else f"B{action}"
        nodes.append(
            {
                "action": int(action),
                "label": label,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            }
        )
    return nodes, {"method": "pca", "explained_variance_ratio": explained}


def _paths_from_route_info(route_info: dict[str, Any], *, batch_size: int, out_action: int) -> list[list[int]]:
    actions = route_info.get("selected_actions") or []
    if not actions:
        return []
    stacked = torch.stack(actions).detach().cpu()
    paths = []
    for sample_index in range(batch_size):
        paths.append(_truncate_at_out([int(value) for value in stacked[:, sample_index].tolist()], out_action=out_action))
    return paths


def _path_counts_from_train_row(row: dict[str, Any], *, out_action: int) -> tuple[Counter[tuple[int, ...]], bool]:
    counts: Counter[tuple[int, ...]] = Counter()
    exact_items = row.get("route_path_counts")
    if isinstance(exact_items, list):
        for item in exact_items:
            if not isinstance(item, dict):
                continue
            actions = _actions_from_value(item.get("actions"))
            count = _safe_int(item.get("count"), default=0)
            if actions and count > 0:
                counts[tuple(_truncate_at_out(actions, out_action=out_action))] += count
        return counts, True
    examples = row.get("route_path_examples")
    if isinstance(examples, list):
        for item in examples:
            if not isinstance(item, dict):
                continue
            actions = _actions_from_value(item.get("actions"))
            if actions:
                counts[tuple(_truncate_at_out(actions, out_action=out_action))] += 1
        return counts, False
    return counts, True


def _aggregate_from_counts(
    *,
    name: str,
    path_counts: Counter[tuple[int, ...]],
    exact: bool,
    metadata: dict[str, Any],
    max_paths: int | None = None,
) -> dict[str, Any]:
    edge_counts = _edge_counts(path_counts)
    node_counts = _node_counts(path_counts)
    sorted_paths = sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
    path_limit = len(sorted_paths) if max_paths is None else max(0, int(max_paths))
    path_rows = [
        {"actions": list(path), "count": int(count)}
        for path, count in sorted_paths[:path_limit]
    ]
    edge_rows = [
        {"source": int(source), "target": int(target), "count": int(count)}
        for (source, target), count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    node_rows = [
        {"action": int(action), "count": int(count)}
        for action, count in sorted(node_counts.items(), key=lambda item: item[0])
    ]
    total = int(sum(path_counts.values()))
    return {
        "name": name,
        "exact": exact,
        **metadata,
        "total_path_count": total,
        "unique_path_count": len(path_counts),
        "path_counts_returned": len(path_rows),
        "path_counts_truncated": len(path_rows) < len(path_counts),
        "unique_edge_count": len(edge_counts),
        "path_counts": path_rows,
        "edge_counts": edge_rows,
        "node_counts": node_rows,
    }


def _edge_counts(path_counts: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for path, count in path_counts.items():
        for source, target in zip(path[:-1], path[1:]):
            counts[(int(source), int(target))] += int(count)
    return counts


def _node_counts(path_counts: Counter[tuple[int, ...]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path, count in path_counts.items():
        for action in path:
            counts[int(action)] += int(count)
    return counts


def _write_html(report: dict[str, Any], output_path: Path, *, top_paths: int) -> None:
    try:
        import plotly.io as pio
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Plotly is required for route path visualization. Install plotly>=5.22.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Route Path Visualization</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .meta{color:#444;} pre{background:#f6f8fa;padding:12px;overflow:auto;}</style>",
        "</head><body>",
        "<h1>Route Path Visualization</h1>",
        f"<p class=\"meta\">run: {report['run_dir']}</p>",
        f"<p class=\"meta\">sidecar JSON: {report['sidecar_json']}</p>",
        "<pre>",
        json.dumps(
            {
                "source": report["source"],
                "projection": report["projection"],
                "overall_status": report["overall_status"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        ),
        "</pre>",
    ]
    include_plotly = "cdn"
    for aggregate in report["aggregates"]:
        fig = _aggregate_figure(report["nodes"], aggregate, top_paths=top_paths)
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=include_plotly))
        include_plotly = False
        frames = aggregate.get("timeline_frames")
        if isinstance(frames, list) and frames:
            timeline = _timeline_figure(report["nodes"], frames)
            parts.append(pio.to_html(timeline, full_html=False, include_plotlyjs=False))
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _aggregate_figure(nodes: list[dict[str, Any]], aggregate: dict[str, Any], *, top_paths: int):
    import plotly.graph_objects as go

    edge_rows = list(aggregate.get("edge_counts") or [])
    path_rows = list(aggregate.get("path_counts") or [])[:top_paths]
    node_counts = _counts_by_action(aggregate.get("node_counts"))
    traces = []
    max_edge_count = max([int(row.get("count", 0)) for row in edge_rows] or [1])
    for row in edge_rows:
        traces.append(_edge_trace(nodes, row, max_edge_count=max_edge_count))
    max_path_count = max([int(row.get("count", 0)) for row in path_rows] or [1])
    for row in path_rows:
        traces.append(_path_trace(nodes, row, max_path_count=max_path_count))
    traces.append(_node_trace(nodes, node_counts))
    title = f"{aggregate['name']} route paths ({aggregate.get('total_path_count', 0)} samples, exact={aggregate.get('exact')})"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        showlegend=False,
    )
    return fig


def _timeline_figure(nodes: list[dict[str, Any]], frames: list[dict[str, Any]]):
    import plotly.graph_objects as go

    edge_keys = sorted(
        {
            (int(edge["source"]), int(edge["target"]))
            for frame in frames
            for edge in frame.get("edge_counts", [])
        }
    )
    first = frames[0]
    max_edge_count = max(
        [int(edge.get("count", 0)) for frame in frames for edge in frame.get("edge_counts", [])] or [1]
    )
    first_edges = _edge_rows_by_key(first.get("edge_counts", []))
    data = [_edge_trace_for_key(nodes, key, first_edges.get(key, 0), max_edge_count=max_edge_count) for key in edge_keys]
    data.append(_node_trace(nodes, _counts_by_action(first.get("node_counts"))))
    plotly_frames = []
    for frame in frames:
        edge_counts = _edge_rows_by_key(frame.get("edge_counts", []))
        plotly_frames.append(
            go.Frame(
                data=[
                    _edge_trace_for_key(nodes, key, edge_counts.get(key, 0), max_edge_count=max_edge_count)
                    for key in edge_keys
                ],
                traces=list(range(len(edge_keys))),
                name=str(frame.get("step")),
            )
        )
    fig = go.Figure(data=data, frames=plotly_frames)
    fig.update_layout(
        title="train_log cumulative route path timeline",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "data"},
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(frame.get("step")),
                        "method": "animate",
                        "args": [[str(frame.get("step"))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for frame in frames
                ]
            }
        ],
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        showlegend=False,
    )
    return fig


def _edge_trace(nodes: list[dict[str, Any]], row: dict[str, Any], *, max_edge_count: int):
    key = (int(row["source"]), int(row["target"]))
    return _edge_trace_for_key(nodes, key, int(row.get("count", 0)), max_edge_count=max_edge_count)


def _edge_trace_for_key(nodes: list[dict[str, Any]], key: tuple[int, int], count: int, *, max_edge_count: int):
    import plotly.graph_objects as go

    if count <= 0:
        return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines")
    source, target = key
    source_node = nodes[source]
    target_node = nodes[target]
    width = 1.0 + 7.0 * math.sqrt(count / max(1, max_edge_count))
    alpha = 0.18 + 0.62 * math.sqrt(count / max(1, max_edge_count))
    if source == target:
        x, y, z = _loop_points(source_node)
    else:
        x = [source_node["x"], target_node["x"]]
        y = [source_node["y"], target_node["y"]]
        z = [source_node["z"], target_node["z"]]
    text = f"{source_node['label']} -> {target_node['label']}<br>count={count}"
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": width, "color": f"rgba(31, 87, 164, {alpha})"},
        hoverinfo="text",
        text=[text for _ in x],
    )


def _path_trace(nodes: list[dict[str, Any]], row: dict[str, Any], *, max_path_count: int):
    import plotly.graph_objects as go

    actions = [int(value) for value in row.get("actions", []) if 0 <= int(value) < len(nodes)]
    if len(actions) < 2:
        return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines")
    count = int(row.get("count", 0))
    width = 1.0 + 3.0 * math.sqrt(count / max(1, max_path_count))
    x = [nodes[action]["x"] for action in actions]
    y = [nodes[action]["y"] for action in actions]
    z = [nodes[action]["z"] for action in actions]
    labels = " -> ".join(nodes[action]["label"] for action in actions)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": width, "color": "rgba(20, 20, 20, 0.18)"},
        hoverinfo="text",
        text=[f"path={labels}<br>count={count}" for _ in actions],
    )


def _node_trace(nodes: list[dict[str, Any]], node_counts: dict[int, int]):
    import plotly.graph_objects as go

    max_count = max(node_counts.values() or [1])
    sizes = [8.0 + 18.0 * math.sqrt(node_counts.get(int(node["action"]), 0) / max(1, max_count)) for node in nodes]
    colors = ["#d1495b" if node["label"] == "OUT" else "#2a9d8f" for node in nodes]
    hover = [f"{node['label']}<br>visits={node_counts.get(int(node['action']), 0)}" for node in nodes]
    return go.Scatter3d(
        x=[node["x"] for node in nodes],
        y=[node["y"] for node in nodes],
        z=[node["z"] for node in nodes],
        mode="markers+text",
        marker={"size": sizes, "color": colors, "line": {"width": 1, "color": "#222"}},
        text=[node["label"] for node in nodes],
        textposition="top center",
        hoverinfo="text",
        hovertext=hover,
    )


def _loop_points(node: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    radius = 0.04
    x0 = float(node["x"])
    y0 = float(node["y"])
    z0 = float(node["z"])
    angles = np.linspace(0.0, 2.0 * math.pi, 24)
    x = [x0 + radius * math.cos(float(angle)) for angle in angles]
    y = [y0 + radius * math.sin(float(angle)) for angle in angles]
    z = [z0 for _ in angles]
    return x, y, z


def _checks(aggregates: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "aggregates_present": bool(aggregates),
        "paths_present": all(int(aggregate.get("total_path_count") or 0) > 0 for aggregate in aggregates),
        "edges_present": all(int(aggregate.get("unique_edge_count") or 0) > 0 for aggregate in aggregates),
    }


def _actions_from_value(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    actions = []
    for item in value:
        if isinstance(item, bool):
            return []
        if isinstance(item, int):
            actions.append(item)
        elif isinstance(item, float) and math.isfinite(item) and item.is_integer():
            actions.append(int(item))
        else:
            return []
    return actions


def _truncate_at_out(actions: list[int], *, out_action: int) -> list[int]:
    truncated: list[int] = []
    for action in actions:
        truncated.append(int(action))
        if action == out_action:
            break
    return truncated


def _counts_by_action(rows: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if isinstance(row, dict):
            counts[int(row.get("action", 0))] = int(row.get("count", 0))
    return counts


def _edge_rows_by_key(rows: Any) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if isinstance(row, dict):
            counts[(int(row.get("source", 0)), int(row.get("target", 0)))] = int(row.get("count", 0))
    return counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _evenly_spaced_indexes(count: int, max_items: int) -> list[int]:
    if count <= 0 or max_items <= 0:
        return []
    if count <= max_items:
        return list(range(count))
    values = np.linspace(0, count - 1, num=max_items)
    return sorted({int(round(value)) for value in values})


def _safe_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return default


def _int_value(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be an integer.")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return number


def _bool_mapping_value(mapping: dict[str, Any] | Any, key: str, *, default: bool, name: str) -> bool:
    value = mapping.get(key, default) if isinstance(mapping, dict) else default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")
