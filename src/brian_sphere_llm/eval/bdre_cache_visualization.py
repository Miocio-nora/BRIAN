from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from brian_sphere_llm.utils.logging import write_json

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def make_bdre_cache_visualization_from_payload(
    payload: Mapping[str, Any],
    model: Any,
    *,
    output_path: str | Path,
    step: int | None = None,
    sample_index: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for BDRE cache visualization.")
    required = {"writer_blocks", "writer_valid", "key_weights", "value_weights"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"BDRE visualization payload is missing: {missing}")
    writer_blocks = _array(payload["writer_blocks"]).astype(np.int64)
    writer_valid = _array(payload["writer_valid"]).astype(bool)
    key_weights = _array(payload["key_weights"]).astype(np.float64)
    value_weights = _array(payload["value_weights"]).astype(np.float64)
    if writer_blocks.ndim != 2 or key_weights.ndim != 3 or value_weights.shape != key_weights.shape:
        raise ValueError("Invalid BDRE visualization tensor shapes.")
    if sample_index < 0 or sample_index >= writer_blocks.shape[0]:
        raise ValueError("BDRE visualization sample_index is out of range.")

    table = model.position_table
    internal_count = int(model.config.route_pool_blocks)
    input_position = F.normalize(table.input_position, dim=-1).detach().float().cpu().numpy()
    action_positions = F.normalize(table.embeddings, dim=-1).detach().float().cpu().numpy()
    ordered_positions = np.concatenate([input_position[None], action_positions], axis=0)
    projected, explained = _pca3(ordered_positions)
    labels = ["IN", *[f"B{index}" for index in range(internal_count)], "OUT"]
    nodes = [
        {"label": label, "x": float(row[0]), "y": float(row[1]), "z": float(row[2])}
        for label, row in zip(labels, projected)
    ]

    valid = writer_valid[sample_index]
    blocks = writer_blocks[sample_index][valid]
    route = [
        {
            "route_step": int(route_step),
            "block": int(block),
            "label": f"B{int(block)}",
            **nodes[int(block) + 1],
        }
        for route_step, block in enumerate(blocks)
    ]
    key = key_weights[sample_index, :, : valid.size][..., valid]
    value = value_weights[sample_index, :, : valid.size][..., valid]
    reader_step_key = _optional_reader_step_weights(payload.get("reader_step_key_weights"), sample_index, valid)
    reader_step_value = _optional_reader_step_weights(payload.get("reader_step_value_weights"), sample_index, valid)
    if (reader_step_key is None) != (reader_step_value is None):
        raise ValueError("BDRE reader-step Key and Value visualization tensors must be provided together.")
    support = ((key > 0) | (value > 0)).sum(axis=-1)
    report: dict[str, Any] = {
        "step": step,
        "sample_index": sample_index,
        "metadata": dict(metadata or {}),
        "nodes": nodes,
        "route": route,
        "reader_labels": [f"B{index}" for index in range(internal_count)],
        "writer_labels": [f"s{index + 1}:B{int(block)}" for index, block in enumerate(blocks)],
        "key_weights": key.tolist(),
        "value_weights": value.tolist(),
        "reader_step_key_weights": reader_step_key.tolist() if reader_step_key is not None else None,
        "reader_step_value_weights": reader_step_value.tolist() if reader_step_value is not None else None,
        "metrics": {
            "writer_steps": int(valid.sum()),
            "key_entropy_mean": _entropy(key),
            "value_entropy_mean": _entropy(value),
            "support_mean": float(support.mean()) if support.size else 0.0,
            "last_step_mass_mean": float(0.5 * (key[:, -1].mean() + value[:, -1].mean())) if key.size else 0.0,
            "position_pca_explained_variance": explained,
        },
        "checks": {
            "writer_route_present": bool(route),
            "one_row_per_reader": key.shape[0] == internal_count,
            "key_value_shapes_match": key.shape == value.shape,
            "weights_finite": bool(np.isfinite(key).all() and np.isfinite(value).all()),
        },
    }
    report["overall_status"] = "pass" if all(report["checks"].values()) else "warn"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, output_path.with_suffix(".json"))
    output_path.write_text(_html(report), encoding="utf-8")
    return output_path


def _array(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _optional_reader_step_weights(value: Any, sample_index: int, valid: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    weights = _array(value).astype(np.float64)
    if weights.ndim != 4:
        raise ValueError("BDRE reader-step weights must have shape [batch, reader_step, reader, writer_step].")
    return weights[sample_index, :, :, : valid.size][..., valid]


def _pca3(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(3, vh.shape[0])
    projected = centered @ vh[:dimensions].T
    if dimensions < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - dimensions)))
    variance = singular**2
    explained = (variance / max(float(variance.sum()), 1e-12))[:3]
    return projected, [float(value) for value in explained]


def _entropy(weights: np.ndarray) -> float:
    if not weights.size:
        return 0.0
    safe = np.clip(weights, 1e-12, None)
    return float((-(safe * np.log(safe)).sum(axis=-1)).mean())


def _html(report: Mapping[str, Any]) -> str:
    data = json.dumps(report, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BRIAN BDRE Cache</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; color: #171717; background: #f6f7f8; font: 14px system-ui, sans-serif; }}
    header {{ padding: 14px 18px; background: #fff; border-bottom: 1px solid #d9dde2; }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(420px, 1.2fr); gap: 12px; padding: 12px; }}
    section {{ min-width: 0; background: #fff; border: 1px solid #d9dde2; border-radius: 6px; }}
    .toolbar {{ display: flex; gap: 16px; align-items: center; padding: 10px 12px; border-bottom: 1px solid #e4e7eb; }}
    .plot {{ height: 520px; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .plot {{ height: 430px; }} }}
  </style>
</head>
<body>
<header><h1>BRIAN BDRE Reader-Compiled Cache</h1></header>
<main>
  <section>
    <div class="toolbar"><label><input id="showInOut" type="checkbox" checked> Show IN/OUT</label></div>
    <div id="geometry" class="plot"></div>
  </section>
  <section>
    <div class="toolbar">
      <label><input type="radio" name="weights" value="key" checked> Key</label>
      <label><input type="radio" name="weights" value="value"> Value</label>
      <label id="readerStepLabel">Reader step <select id="readerStep"></select></label>
    </div>
    <div id="weights" class="plot"></div>
  </section>
</main>
<script>
const report={data};
const layout={{margin:{{l:45,r:20,t:35,b:45}},paper_bgcolor:'#fff',plot_bgcolor:'#fff'}};
function geometry(showInOut) {{
  const nodes=report.nodes.filter(n => showInOut || (n.label !== 'IN' && n.label !== 'OUT'));
  const route=report.route;
  const traces=[{{type:'scatter3d',mode:'markers+text',x:nodes.map(n=>n.x),y:nodes.map(n=>n.y),z:nodes.map(n=>n.z),text:nodes.map(n=>n.label),textposition:'top center',marker:{{size:7,color:nodes.map(n=>n.label==='IN'?'#1f77b4':n.label==='OUT'?'#d62728':'#555')}}}},{{type:'scatter3d',mode:'lines+markers',x:route.map(n=>n.x),y:route.map(n=>n.y),z:route.map(n=>n.z),text:route.map(n=>`s${{n.route_step+1}}:${{n.label}}`),line:{{color:'#17a673',width:6}},marker:{{size:4,color:'#17a673'}}}}];
  Plotly.react('geometry',traces,{{...layout,title:'Position Geometry and Writer Route',scene:{{aspectmode:'data'}}}},{{responsive:true}});
}}
function heatmap(kind) {{
  const readerStep=Number(document.getElementById('readerStep').value || 0);
  const stepWeights=kind==='key'?report.reader_step_key_weights:report.reader_step_value_weights;
  const z=stepWeights?stepWeights[readerStep]:(kind==='key'?report.key_weights:report.value_weights);
  Plotly.react('weights',[{{type:'heatmap',z,x:report.writer_labels,y:report.reader_labels,colorscale:'Viridis',zmin:0,zmax:1,colorbar:{{title:'mass'}}}}],{{...layout,title:`${{kind==='key'?'Key':'Value'}} Compile Weights`,xaxis:{{title:'Writer step'}},yaxis:{{title:'Reader block',autorange:'reversed'}}}},{{responsive:true}});
}}
document.getElementById('showInOut').addEventListener('change',e=>geometry(e.target.checked));
document.querySelectorAll('input[name=weights]').forEach(el=>el.addEventListener('change',e=>heatmap(e.target.value)));
const readerStep=document.getElementById('readerStep');
const stepCount=report.reader_step_key_weights?report.reader_step_key_weights.length:0;
document.getElementById('readerStepLabel').hidden=stepCount===0;
for(let i=0;i<stepCount;i++){{const option=document.createElement('option');option.value=i;option.textContent=String(i+1);readerStep.appendChild(option);}}
readerStep.addEventListener('change',()=>heatmap(document.querySelector('input[name=weights]:checked').value));
geometry(true); heatmap('key');
</script>
</body>
</html>"""
