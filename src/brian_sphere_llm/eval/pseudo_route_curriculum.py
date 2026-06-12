from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brian_sphere_llm.routing.pseudo_policy import actions_for_policy
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


DIFFICULTY_TO_ID = {"easy": 0, "medium": 1, "hard": 2}


def make_pseudo_route_curriculum_report(
    run_dir: str | Path,
    *,
    baseline_difficulty_report_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for pseudo-route curriculum reports.")

    run_dir = Path(run_dir)
    config = load_config(run_dir / "config_resolved.yaml")
    model_config = config.get("model_config_resolved")
    if not isinstance(model_config, dict):
        raise ValueError("Run config must include model_config_resolved.")
    route_pool_blocks = int(model_config["route_pool_blocks"])
    max_route_steps = int(model_config["max_route_steps"])
    routing_config = config.get("routing", {})
    stage = str(config.get("stage", ""))
    routing_mode = str(routing_config.get("mode", "")) if isinstance(routing_config, dict) else ""
    pseudo_policy = (
        str(routing_config.get("pseudo_policy", "mixed_skip_recur"))
        if isinstance(routing_config, dict)
        else "mixed_skip_recur"
    )

    baseline_report_path = Path(baseline_difficulty_report_path)
    baseline_report = _read_json(baseline_report_path)
    samples_path = _resolve_samples_path(baseline_report_path, baseline_report)
    samples = _read_jsonl(samples_path)
    samples.sort(key=lambda row: int(row.get("sample_id", 0)))
    difficulty_ids = [_difficulty_id(row.get("difficulty_bin")) for row in samples]
    difficulty = torch.tensor(difficulty_ids, dtype=torch.long)
    actions = actions_for_policy(
        pseudo_policy,
        num_internal_blocks=route_pool_blocks,
        max_route_steps=max_route_steps,
        batch_size=len(samples),
        device=torch.device("cpu"),
        difficulty=difficulty,
    )
    rows = _sample_rows(samples, actions, out_action=route_pool_blocks)
    by_difficulty = {label: _difficulty_summary(rows, label) for label in DIFFICULTY_TO_ID}
    checks = _checks(
        rows,
        by_difficulty,
        route_pool_blocks=route_pool_blocks,
        stage=stage,
        routing_mode=routing_mode,
        pseudo_policy=pseudo_policy,
    )
    report = {
        "run_dir": str(run_dir),
        "stage": stage,
        "routing_mode": routing_mode,
        "baseline_difficulty_report": str(baseline_report_path),
        "samples_path": str(samples_path),
        "pseudo_policy": pseudo_policy,
        "route_pool_blocks": route_pool_blocks,
        "max_route_steps": max_route_steps,
        "sample_count": len(rows),
        "by_difficulty": by_difficulty,
        "checks": checks,
        "overall_status": _overall_status(checks),
    }
    if output_path is None:
        output_path = run_dir / "pseudo_route_curriculum_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _overall_status(checks: dict[str, bool]) -> str:
    return "pass" if all(value is True for value in checks.values()) else "fail"


def _sample_rows(samples: list[dict[str, Any]], actions: list["torch.Tensor"], *, out_action: int) -> list[dict[str, Any]]:
    stacked = torch.stack(actions).detach().cpu()
    rows = []
    for sample_index, sample in enumerate(samples):
        path = [int(value) for value in stacked[:, sample_index].tolist()]
        transitions = _internal_transitions(path, out_action)
        first_exit_step = _first_exit_step(path, out_action)
        rows.append(
            {
                "sample_id": int(sample.get("sample_id", sample_index)),
                "difficulty_bin": str(sample.get("difficulty_bin", "medium")),
                "baseline_cross_entropy": _num(sample.get("baseline_cross_entropy")),
                "actions": path,
                "internal_route_steps": sum(1 for action in path if action != out_action),
                "first_exit_step": first_exit_step,
                "has_exit_supervision": first_exit_step > 0,
                "advance_transition_count": transitions["advance"],
                "skip_transition_count": transitions["skip"],
                "recur_transition_count": transitions["recur"],
            }
        )
    return rows


def _internal_transitions(path: list[int], out_action: int) -> dict[str, int]:
    counts = {"advance": 0, "skip": 0, "recur": 0}
    internal = [action for action in path if action != out_action]
    for current, next_action in zip(internal, internal[1:]):
        diff = next_action - current
        if diff == 0:
            counts["recur"] += 1
        elif diff == 1:
            counts["advance"] += 1
        elif diff > 1:
            counts["skip"] += 1
    return counts


def _first_exit_step(path: list[int], out_action: int) -> int:
    for index, action in enumerate(path):
        if action == out_action:
            return index + 1
    return 0


def _difficulty_summary(rows: list[dict[str, Any]], difficulty: str) -> dict[str, Any]:
    matching = [row for row in rows if row["difficulty_bin"] == difficulty]
    return {
        "sample_count": len(matching),
        "mean_baseline_cross_entropy": _mean([row["baseline_cross_entropy"] for row in matching]),
        "mean_internal_route_steps": _mean([row["internal_route_steps"] for row in matching]),
        "mean_first_exit_step": _mean([row["first_exit_step"] for row in matching if row["first_exit_step"] > 0]),
        "advance_transition_count": sum(int(row["advance_transition_count"]) for row in matching),
        "skip_transition_count": sum(int(row["skip_transition_count"]) for row in matching),
        "recur_transition_count": sum(int(row["recur_transition_count"]) for row in matching),
        "exit_supervision_count": sum(1 for row in matching if row["has_exit_supervision"]),
    }


def _checks(
    rows: list[dict[str, Any]],
    by_difficulty: dict[str, dict[str, Any]],
    *,
    route_pool_blocks: int,
    stage: str,
    routing_mode: str,
    pseudo_policy: str,
) -> dict[str, bool]:
    easy = by_difficulty["easy"]
    medium = by_difficulty["medium"]
    hard = by_difficulty["hard"]
    easy_exit = _num(easy.get("mean_first_exit_step"))
    hard_exit = _num(hard.get("mean_first_exit_step"))
    easy_ce = _num(easy.get("mean_baseline_cross_entropy"))
    medium_ce = _num(medium.get("mean_baseline_cross_entropy"))
    hard_ce = _num(hard.get("mean_baseline_cross_entropy"))
    easy_steps = _num(easy.get("mean_internal_route_steps"))
    medium_steps = _num(medium.get("mean_internal_route_steps"))
    hard_steps = _num(hard.get("mean_internal_route_steps"))
    easy_rows = [row for row in rows if row["difficulty_bin"] == "easy"]
    hard_rows = [row for row in rows if row["difficulty_bin"] == "hard"]
    easy_uses_skip_or_early_exit = bool(easy_rows) and all(
        _has_skip_or_early_exit(row) for row in easy_rows
    )
    hard_uses_recurrence = bool(hard_rows) and all(
        int(row.get("recur_transition_count") or 0) > 0 for row in hard_rows
    )
    out_supervised = bool(rows) and all(row["has_exit_supervision"] for row in rows)
    return {
        "stage3_pseudo_skip_recur_stage": stage == "stage3_pseudo_skip_recur",
        "pseudo_routing_mode": routing_mode == "pseudo",
        "baseline_samples_present": bool(rows),
        "baseline_cross_entropy_numeric": bool(rows) and all(row["baseline_cross_entropy"] is not None for row in rows),
        "baseline_cross_entropy_ordered_by_difficulty": (
            easy_ce is not None
            and medium_ce is not None
            and hard_ce is not None
            and easy_ce <= medium_ce
            and medium_ce <= hard_ce
        ),
        "difficulty_bins_present": all(by_difficulty[label]["sample_count"] > 0 for label in DIFFICULTY_TO_ID),
        "mixed_skip_recur_policy": pseudo_policy == "mixed_skip_recur",
        "easy_has_skip_or_small_pool": bool(easy_rows)
        and (route_pool_blocks <= 2 or int(easy.get("skip_transition_count") or 0) > 0),
        "easy_uses_skip_or_early_exit": easy_uses_skip_or_early_exit,
        "hard_has_recur_transition": int(hard.get("recur_transition_count") or 0) > 0,
        "hard_uses_recurrence": hard_uses_recurrence,
        "exit_action_supervised": out_supervised,
        "out_supervised": out_supervised,
        "all_samples_have_supervised_out": out_supervised,
        "supervised_out_targets_present": out_supervised,
        "easy_exits_no_later_than_hard": easy_exit is not None and hard_exit is not None and easy_exit <= hard_exit,
        "route_length_conditioned_by_difficulty": (
            easy_steps is not None
            and medium_steps is not None
            and hard_steps is not None
            and easy_steps <= medium_steps
            and easy_steps <= hard_steps
        ),
    }


def _has_skip_or_early_exit(row: dict[str, Any]) -> bool:
    first_exit_step = int(row.get("first_exit_step") or 0)
    actions = row.get("actions")
    path_length = len(actions) if isinstance(actions, list) else 0
    return int(row.get("skip_transition_count") or 0) > 0 or (
        first_exit_step > 0 and path_length > 0 and first_exit_step < path_length
    )


def _difficulty_id(value: Any) -> int:
    return DIFFICULTY_TO_ID.get(str(value), DIFFICULTY_TO_ID["medium"])


def _resolve_samples_path(report_path: Path, report: dict[str, Any]) -> Path:
    raw_path = report.get("samples_path")
    if not raw_path:
        raise ValueError("Baseline difficulty report must include samples_path.")
    samples_path = Path(str(raw_path))
    if samples_path.exists():
        return samples_path
    candidate = report_path.parent / samples_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve baseline difficulty samples path: {raw_path}")


def _mean(values: list[float | None]) -> float | None:
    finite = [_num(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
