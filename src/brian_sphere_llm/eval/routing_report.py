from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


ROUTING_KEYS = {
    "route_entropy",
    "block_load_entropy",
    "block_load_entropy_normalized",
    "route_path_count",
    "route_path_diversity",
    "p_output_mean",
    "active_block_evals_per_token",
    "average_route_steps",
    "advance_ratio",
    "skip_ratio",
    "recur_ratio",
    "position_norm_mean",
    "location_distance_mean",
    "route_imitation_accuracy",
    "weighted_fusion_ratio",
    "global_attention_mass",
    "global_read_gate_mean",
    "global_cache_slots_mean",
    "parallel_branch_count_mean",
    "parallel_score_margin_mean",
}


def make_routing_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rows = _read_jsonl(run_dir / "train_log.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    aggregates: dict[str, list[float]] = defaultdict(list)
    latest_histogram: dict[str, Any] | None = None
    latest_exit_distribution: list[int] | None = None
    latest_first_exit_histogram: dict[str, Any] | None = None
    latest_route_path_examples: list[Any] | None = None
    latest_position_norm_trajectory: list[Any] | None = None
    latest_location_distance_trajectory: list[Any] | None = None
    for row in rows:
        for key in ROUTING_KEYS:
            if isinstance(row.get(key), (int, float)):
                aggregates[key].append(float(row[key]))
        if isinstance(row.get("top1_block_histogram"), dict):
            latest_histogram = row["top1_block_histogram"]
        if isinstance(row.get("exit_step_distribution"), list):
            latest_exit_distribution = row["exit_step_distribution"]
        if isinstance(row.get("first_exit_step_histogram"), dict):
            latest_first_exit_histogram = row["first_exit_step_histogram"]
        if isinstance(row.get("route_path_examples"), list):
            latest_route_path_examples = row["route_path_examples"]
        if isinstance(row.get("position_norm_trajectory"), list):
            latest_position_norm_trajectory = row["position_norm_trajectory"]
        if isinstance(row.get("location_distance_trajectory"), list):
            latest_location_distance_trajectory = row["location_distance_trajectory"]
    report = {
        "run_dir": str(run_dir),
        "summary": {key: sum(values) / max(1, len(values)) for key, values in aggregates.items()},
        "latest_block_histogram": latest_histogram or {},
        "latest_exit_step_distribution": latest_exit_distribution or [],
        "latest_first_exit_step_histogram": latest_first_exit_histogram or {},
        "latest_route_path_examples": latest_route_path_examples or [],
        "latest_position_norm_trajectory": latest_position_norm_trajectory or [],
        "latest_location_distance_trajectory": latest_location_distance_trajectory or [],
        "latest_eval": eval_rows[-1] if eval_rows else {},
    }
    output = run_dir / "routing_report.json"
    write_json(report, output)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
