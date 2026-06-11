from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from brian_sphere_llm.experiments.runner import ExperimentPlan, build_experiment_plan
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json


PROFILE_ALIASES = {
    "auto": "auto",
    "route_core_r125_package": "package_a_r125_route_core",
    "package_a": "package_a_r125_route_core",
    "package_a_r125_route_core": "package_a_r125_route_core",
    "route_core_r350_scaling": "package_b_r350_scaling",
    "package_b": "package_b_r350_scaling",
    "package_b_r350_scaling": "package_b_r350_scaling",
    "route_core_global_kv": "global_kv_ablation",
    "tiny_global_kv": "global_kv_ablation",
    "package_c": "global_kv_ablation",
    "global_kv_ablation": "global_kv_ablation",
    "route_core_parallel_passing": "parallel_passing_beta",
    "tiny_parallel_passing": "parallel_passing_beta",
    "parallel_passing_beta": "parallel_passing_beta",
}


def make_experiment_coverage_report(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: str = "auto",
    include_baseline: bool = False,
) -> Path:
    plan = build_experiment_plan(manifest_path, include_baseline=include_baseline)
    resolved_profile = _resolve_profile(profile, plan.experiment_name)
    entries = [_summarize_entry(entry) for entry in plan.entries]
    requirements = _requirements(resolved_profile, plan, entries)
    checks = {
        "profile_known": resolved_profile != "unknown",
        "manifest_entries_present": bool(plan.entries),
        "train_configs_exist": bool(entries) and all(entry["checks"]["train_config_exists"] for entry in entries),
        "train_configs_load": bool(entries) and all(entry["checks"]["train_config_loads"] for entry in entries),
        "train_modes_resolve": bool(entries) and all(entry["checks"]["train_mode_resolves"] for entry in entries),
        "baseline_train_config_present": plan.baseline_train_config is not None,
        "required_coverage_satisfied": bool(requirements)
        and all(requirement["status"] == "pass" for requirement in requirements),
    }
    report = {
        "overall_status": _overall_status(checks, requirements),
        "profile": resolved_profile,
        "source_plan_section": "21 First Formal Experiment Package",
        "manifest": plan.to_json(),
        "checks": checks,
        "requirements": requirements,
        "entries": entries,
    }
    if output_path is None:
        output_path = Path("reports") / f"{plan.experiment_name}_coverage_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _resolve_profile(profile: str, experiment_name: str) -> str:
    requested = str(profile or "auto")
    if requested == "auto":
        requested = experiment_name
    return PROFILE_ALIASES.get(requested, "unknown")


def _requirements(profile: str, plan: ExperimentPlan, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if profile == "package_a_r125_route_core":
        return _exact_id_requirements(
            entries,
            [
                _req("A0", "fixed Transformer baseline", stage="stage0_baseline", mode="baseline"),
                _req("A1", "fixed route wrapper", stage="stage1_fixed_route", mode="fixed"),
                _req("A2", "sequential router imitation", stage="stage2_router_imitation", mode="pseudo"),
                _req("A3", "skip/recur router imitation", stage="stage3_pseudo_skip_recur", mode="pseudo"),
                _req("A4", "free router + block-position", stage="stage3_scheduled_free_routing", mode="scheduled"),
                _req(
                    "A5",
                    "no block-position ablation",
                    mode="scheduled",
                    model_flags={
                        "block_position_mode": "none",
                        "position_to_router": False,
                        "position_to_blocks": False,
                    },
                ),
                _req("A6", "no output action ablation", stage="stage4_scheduled_free_routing", mode="scheduled"),
                _req("A7", "no location loss ablation", mode="scheduled", loss_weights={"location": 0.0}),
            ],
        )
    if profile == "package_b_r350_scaling":
        return _exact_id_requirements(
            entries,
            [
                _req("B0", "350M fixed baseline", stage="stage0_baseline", mode="baseline"),
                _req("B1", "350M routed main", stage="stage4_output_action", mode="scheduled"),
                _req(
                    "B2",
                    "350M routed no-position",
                    mode="scheduled",
                    model_flags={
                        "block_position_mode": "none",
                        "position_to_router": False,
                        "position_to_blocks": False,
                    },
                ),
                _req("B3", "350M routed no-output-action", stage="stage4_scheduled_free_routing", mode="scheduled"),
                _req("B4", "350M difficulty-conditioned route", mode="pseudo"),
            ],
        )
    if profile == "global_kv_ablation":
        return _global_kv_requirements(plan, entries)
    if profile == "parallel_passing_beta":
        return _parallel_requirements(entries)
    return []


def _req(
    entry_id: str,
    description: str,
    *,
    stage: str | None = None,
    mode: str | None = None,
    model_flags: dict[str, Any] | None = None,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "description": description,
        "stage": stage,
        "mode": mode,
        "model_flags": model_flags or {},
        "loss_weights": loss_weights or {},
    }


def _exact_id_requirements(entries: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {entry["id"]: entry for entry in entries}
    return [_requirement_row(spec, [by_id[spec["entry_id"]]] if spec["entry_id"] in by_id else []) for spec in specs]


def _global_kv_requirements(plan: ExperimentPlan, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    global_entries = [entry for entry in entries if entry["model"].get("global_kv") is True]
    window_entries = [
        entry
        for entry in global_entries
        if _num(entry["model"].get("global_window_slots")) is not None
        and _num(entry["model"].get("global_window_slots")) > 0
    ]
    distinct_windows = sorted(
        {
            int(_num(entry["model"].get("global_window_slots")) or 0)
            for entry in window_entries
            if _num(entry["model"].get("global_window_slots")) is not None
        }
    )
    return [
        _group_requirement(
            "C0",
            "local KV only baseline",
            entries,
            lambda entry: entry["model"].get("global_kv") is not True,
        ),
        _group_requirement(
            "C1",
            "local plus global uncompressed",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_code_dim")) is not None
            and _num(entry["model"].get("base_d_model")) is not None
            and _num(entry["model"].get("global_code_dim")) == _num(entry["model"].get("base_d_model")),
        ),
        _group_requirement(
            "C2",
            "local plus global compressed",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_code_dim")) is not None
            and _num(entry["model"].get("base_d_model")) is not None
            and _num(entry["model"].get("global_code_dim")) < _num(entry["model"].get("base_d_model")),
        ),
        _group_requirement(
            "C3",
            "global KV without sink",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_sink_slots")) == 0,
        ),
        _group_requirement(
            "C4",
            "global KV with sink and window",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and (_num(entry["model"].get("global_sink_slots")) or 0) > 0
            and (_num(entry["model"].get("global_window_slots")) or 0) > 0,
        ),
        {
            "id": "C5",
            "description": "global KV window size sweep",
            "status": "pass" if len(distinct_windows) >= 2 else "fail",
            "matched_entry_ids": [entry["id"] for entry in window_entries],
            "checks": {
                "at_least_two_window_sizes": len(distinct_windows) >= 2,
                "distinct_global_window_slots": distinct_windows,
            },
        },
        _group_requirement(
            "C6",
            "global KV per-block adapter",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and entry["model"].get("global_adapter_scope") == "per_block"
            and (_num(entry["model"].get("global_head_delta_rank")) or 0) == 0,
        ),
        _group_requirement(
            "C7",
            "global KV per-block plus per-head low-rank delta",
            entries,
            lambda entry: entry["model"].get("global_kv") is True
            and entry["model"].get("global_adapter_scope") == "per_block"
            and (_num(entry["model"].get("global_head_delta_rank")) or 0) > 0,
        ),
        {
            "id": "baseline_train_config",
            "description": "local baseline train config is declared for compute comparison",
            "status": "pass" if plan.baseline_train_config is not None else "fail",
            "matched_entry_ids": [],
            "checks": {"baseline_train_config_present": plan.baseline_train_config is not None},
        },
    ]


def _parallel_requirements(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _group_requirement(
            "PP0",
            "top-k weighted fusion single-state baseline",
            entries,
            lambda entry: entry["id"] == "PP0"
            and entry["model"].get("parallel_passing") is not True
            and (_num(entry["model"].get("top_k")) or 0) >= 2,
        ),
        _group_requirement(
            "PP1",
            "beam-2 independent parallel passing",
            entries,
            lambda entry: entry["id"] == "PP1"
            and entry["model"].get("parallel_passing") is True
            and int(_num(entry["model"].get("beam_size")) or 0) == 2
            and (_num(entry["model"].get("branch_cost")) or 0.0) > 0.0,
        ),
        _group_requirement(
            "PP2",
            "beam-4 parallel capacity test",
            entries,
            lambda entry: entry["id"] == "PP2"
            and entry["model"].get("parallel_passing") is True
            and int(_num(entry["model"].get("beam_size")) or 0) == 4,
        ),
        _group_requirement(
            "PP3",
            "branch cost off ablation",
            entries,
            lambda entry: entry["id"] == "PP3"
            and entry["model"].get("parallel_passing") is True
            and _num(entry["model"].get("branch_cost")) == 0.0
            and _num(entry.get("loss_weights", {}).get("cost")) == 0.0,
        ),
        _group_requirement(
            "PP4",
            "branch cost on proposal",
            entries,
            lambda entry: entry["id"] == "PP4"
            and entry["model"].get("parallel_passing") is True
            and (_num(entry["model"].get("branch_cost")) or 0.0) > 0.0
            and (_num(entry.get("loss_weights", {}).get("cost")) or 0.0) > 0.0,
        ),
        _group_requirement(
            "PP5",
            "top-1 OUT terminal rule",
            entries,
            lambda entry: entry["id"] == "PP5"
            and entry["model"].get("parallel_passing") is True
            and entry["model"].get("parallel_exit_policy") == "top1",
        ),
        _group_requirement(
            "PP6",
            "OUT in top-k terminal rule",
            entries,
            lambda entry: entry["id"] == "PP6"
            and entry["model"].get("parallel_passing") is True
            and entry["model"].get("parallel_exit_policy") == "any_topk",
        ),
        _group_requirement(
            "PP7",
            "shared base Global KV plus branch delta memory",
            entries,
            lambda entry: entry["model"].get("parallel_passing") is True
            and entry["model"].get("global_kv") is True
            and (_num(entry["model"].get("global_window_slots")) or 0) > 0,
        ),
    ]


def _group_requirement(
    requirement_id: str,
    description: str,
    entries: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    matched = [entry for entry in entries if predicate(entry)]
    return {
        "id": requirement_id,
        "description": description,
        "status": "pass" if matched else "fail",
        "matched_entry_ids": [entry["id"] for entry in matched],
        "checks": {"matched": bool(matched)},
    }


def _requirement_row(spec: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    entry = matches[0] if matches else None
    checks = {
        "entry_present": entry is not None,
        "stage_matches": _check_stage(entry, spec.get("stage")),
        "mode_matches": _check_mode(entry, spec.get("mode")),
        "model_flags_match": _check_mapping(entry, "model", spec["model_flags"]),
        "loss_weights_match": _check_mapping(entry, "loss_weights", spec["loss_weights"]),
    }
    return {
        "id": spec["entry_id"],
        "description": spec["description"],
        "status": "pass" if all(checks.values()) else "fail",
        "matched_entry_ids": [entry["id"]] if entry else [],
        "checks": checks,
    }


def _check_stage(entry: dict[str, Any] | None, expected: str | None) -> bool:
    return expected is None or bool(entry and entry.get("stage") == expected)


def _check_mode(entry: dict[str, Any] | None, expected: str | None) -> bool:
    return expected is None or bool(entry and entry.get("train_mode") == expected)


def _check_mapping(entry: dict[str, Any] | None, key: str, expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    if not entry:
        return False
    observed = entry.get(key)
    if not isinstance(observed, dict):
        return False
    for expected_key, expected_value in expected.items():
        if _num(expected_value) is not None:
            observed_value = _num(observed.get(expected_key))
            if observed_value is None or observed_value != float(expected_value):
                return False
        elif observed.get(expected_key) != expected_value:
            return False
    return True


def _summarize_entry(entry: Any) -> dict[str, Any]:
    train_config_path = Path(entry.train_config)
    train_config: dict[str, Any] = {}
    train_config_error = None
    if train_config_path.exists():
        try:
            train_config = load_config(train_config_path)
        except Exception as exc:  # pragma: no cover - defensive report evidence
            train_config_error = str(exc)
    stage = str(train_config.get("stage", ""))
    try:
        train_mode = train_mode_for_stage(stage) if stage else None
        train_mode_error = None
    except ValueError as exc:
        train_mode = None
        train_mode_error = str(exc)
    model_config_path = _resolve_optional_reference(train_config.get("model_config"), train_config_path)
    model_config = _load_optional_config(model_config_path)
    routing = train_config.get("routing") if isinstance(train_config.get("routing"), dict) else {}
    loss_weights = train_config.get("loss_weights") if isinstance(train_config.get("loss_weights"), dict) else {}
    return {
        **entry.to_json(),
        "train_config": str(train_config_path),
        "stage": stage or None,
        "train_mode": train_mode,
        "routing_mode": routing.get("mode"),
        "model_config": str(model_config_path) if model_config_path else None,
        "model": _model_summary(model_config, model_config_path),
        "loss_weights": {key: _num(value) for key, value in loss_weights.items()},
        "checks": {
            "train_config_exists": train_config_path.exists(),
            "train_config_loads": train_config_path.exists() and train_config_error is None,
            "train_mode_resolves": train_mode is not None and train_mode_error is None,
            "model_config_exists": model_config_path.exists() if model_config_path else False,
        },
        "errors": {
            "train_config": train_config_error,
            "train_mode": train_mode_error,
        },
    }


def _resolve_optional_reference(value: Any, source_config: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (source_config.parent / path).resolve()


def _load_optional_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return load_config(path)


def _model_summary(config: dict[str, Any], model_config_path: Path | None) -> dict[str, Any]:
    keys = [
        "model_name",
        "architecture",
        "base_config",
        "pre_blocks",
        "route_pool_blocks",
        "post_blocks",
        "block_position_dim",
        "block_position_mode",
        "position_to_router",
        "position_to_blocks",
        "max_route_steps",
        "top_k",
        "later_top_k",
        "hard_exit",
        "global_kv",
        "global_code_dim",
        "global_sink_slots",
        "global_window_slots",
        "global_adapter_scope",
        "global_head_delta_rank",
        "parallel_passing",
        "beam_size",
        "branch_cost",
        "parallel_exit_policy",
    ]
    summary = {key: config.get(key) for key in keys if key in config}
    base_config = config.get("base_config")
    if base_config and model_config_path is not None:
        base_path = (model_config_path.parent / str(base_config)).resolve()
        base = _load_optional_config(base_path)
        if "d_model" in base:
            summary["base_d_model"] = base.get("d_model")
    elif isinstance(config.get("base"), dict):
        base = config["base"]
        if "d_model" in base:
            summary["base_d_model"] = base.get("d_model")
    return summary


def _overall_status(checks: dict[str, bool], requirements: list[dict[str, Any]]) -> str:
    if not checks.get("profile_known", False):
        return "fail"
    if any(requirement["status"] == "fail" for requirement in requirements):
        return "fail"
    if checks and all(checks.values()):
        return "pass"
    if any(checks.values()):
        return "warn"
    return "fail"


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
