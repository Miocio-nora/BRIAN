from __future__ import annotations

from collections.abc import Callable
import math
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
    "route_core_position_ablations": "block_position_ablation",
    "tiny_position_ablations": "block_position_ablation",
    "block_position_ablation": "block_position_ablation",
    "route_core_cost_control": "cost_control_sweep",
    "tiny_cost_control": "cost_control_sweep",
    "cost_control_sweep": "cost_control_sweep",
    "route_core_global_kv": "global_kv_ablation",
    "tiny_global_kv": "global_kv_ablation",
    "package_c": "global_kv_ablation",
    "global_kv_ablation": "global_kv_ablation",
    "route_core_parallel_passing": "parallel_passing_beta",
    "tiny_parallel_passing": "parallel_passing_beta",
    "parallel_passing_beta": "parallel_passing_beta",
    "route_core_r1b_pilot": "package_d_r1b_pilot",
    "package_d": "package_d_r1b_pilot",
    "package_d_r1b_pilot": "package_d_r1b_pilot",
    "route_core_r125_5b_followup": "scale_r125_5b_followup",
    "scale_r125_5b_followup": "scale_r125_5b_followup",
    "route_core_r350_30b_followup": "scale_r350_30b_followup",
    "scale_r350_30b_followup": "scale_r350_30b_followup",
    "route_core_r1b_main_validation": "package_d_r1b_main_validation",
    "package_d_r1b_main_validation": "package_d_r1b_main_validation",
}

PLANNED_PARAMETER_RANGES = {
    "125m": (110_000_000, 150_000_000),
    "r125": (110_000_000, 150_000_000),
    "350m": (300_000_000, 400_000_000),
    "r350": (300_000_000, 400_000_000),
    "1b": (800_000_000, 1_300_000_000),
    "r1b": (800_000_000, 1_300_000_000),
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
    baseline = _summarize_baseline(plan.baseline_train_config)
    requirements = _requirements(resolved_profile, plan, entries)
    checks = {
        "profile_known": resolved_profile != "unknown",
        "manifest_entries_present": bool(plan.entries),
        "train_configs_exist": bool(entries) and all(entry["checks"]["train_config_exists"] for entry in entries),
        "train_configs_load": bool(entries) and all(entry["checks"]["train_config_loads"] for entry in entries),
        "train_modes_resolve": bool(entries) and all(entry["checks"]["train_mode_resolves"] for entry in entries),
        "model_configs_exist": bool(entries) and all(entry["checks"]["model_config_exists"] for entry in entries),
        "model_configs_load": bool(entries) and all(entry["checks"]["model_config_loads"] for entry in entries),
        "model_configs_valid": bool(entries) and all(entry["checks"]["model_config_valid"] for entry in entries),
        "model_base_configs_exist": bool(entries) and all(entry["checks"]["model_base_config_exists"] for entry in entries),
        "model_base_configs_load": bool(entries) and all(entry["checks"]["model_base_config_loads"] for entry in entries),
        "data_configs_exist": bool(entries) and all(entry["checks"]["data_config_exists"] for entry in entries),
        "data_configs_load": bool(entries) and all(entry["checks"]["data_config_loads"] for entry in entries),
        "data_configs_consistent": _data_configs_consistent(entries),
        "baseline_train_config_present": plan.baseline_train_config is not None,
        "baseline_train_config_exists": _baseline_check(baseline, "train_config_exists"),
        "baseline_train_config_loads": _baseline_check(baseline, "train_config_loads"),
        "baseline_train_mode_resolves": _baseline_check(baseline, "train_mode_resolves"),
        "baseline_model_config_exists": _baseline_check(baseline, "model_config_exists"),
        "baseline_model_config_loads": _baseline_check(baseline, "model_config_loads"),
        "baseline_model_config_valid": _baseline_check(baseline, "model_config_valid"),
        "baseline_model_base_config_exists": _baseline_check(baseline, "model_base_config_exists"),
        "baseline_model_base_config_loads": _baseline_check(baseline, "model_base_config_loads"),
        "baseline_data_config_exists": _baseline_check(baseline, "data_config_exists"),
        "baseline_data_config_loads": _baseline_check(baseline, "data_config_loads"),
        "baseline_data_config_consistent": _baseline_data_config_consistent(baseline, entries),
        "planned_parameter_estimates_in_range": _planned_parameter_estimates_in_range([baseline, *entries]),
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
        "baseline": baseline,
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
                _req(
                    "A2",
                    "sequential router imitation",
                    stage="stage2_router_imitation",
                    mode="pseudo",
                    routing_flags={"pseudo_policy": "sequential"},
                ),
                _req(
                    "A3",
                    "skip/recur router imitation",
                    stage="stage3_pseudo_skip_recur",
                    mode="pseudo",
                    routing_flags={"pseudo_policy": "mixed_skip_recur"},
                ),
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
                _req("A8", "top-1 routing baseline", mode="scheduled", model_flags={"top_k": 1}),
                _req("A9", "top-2 weighted routing", mode="scheduled", model_flags={"top_k": 2}),
            ],
        )
    if profile == "package_b_r350_scaling":
        return _exact_id_requirements(
            entries,
            [
                _req(
                    "B0",
                    "350M fixed baseline",
                    stage="stage0_baseline",
                    mode="baseline",
                    train_flags={"precision": "bf16"},
                ),
                _req(
                    "B1",
                    "350M routed main",
                    stage="stage4_output_action",
                    mode="scheduled",
                    train_flags={"precision": "bf16"},
                ),
                _req(
                    "B2",
                    "350M routed no-position",
                    mode="scheduled",
                    model_flags={
                        "block_position_mode": "none",
                        "position_to_router": False,
                        "position_to_blocks": False,
                    },
                    train_flags={"precision": "bf16"},
                ),
                _req(
                    "B3",
                    "350M routed no-output-action",
                    stage="stage4_scheduled_free_routing",
                    mode="scheduled",
                    train_flags={"precision": "bf16"},
                ),
                _req(
                    "B4",
                    "350M difficulty-conditioned route",
                    stage="stage3_pseudo_skip_recur",
                    mode="pseudo",
                    routing_flags={"pseudo_policy": "mixed_skip_recur"},
                    train_flags={"precision": "bf16"},
                ),
            ],
        )
    if profile == "block_position_ablation":
        return _position_requirements(entries)
    if profile == "cost_control_sweep":
        return _cost_control_requirements(entries)
    if profile == "global_kv_ablation":
        return _global_kv_requirements(plan, entries)
    if profile == "parallel_passing_beta":
        return _parallel_requirements(entries)
    if profile == "package_d_r1b_pilot":
        return _exact_id_requirements(
            entries,
            [
                _req(
                    "D0",
                    "1B fixed decoder-only baseline",
                    stage="stage0_baseline",
                    mode="baseline",
                    model_flags={"model_name": "baseline_1b"},
                    train_flags={
                        "precision": "bf16",
                        "activation_checkpointing": True,
                        "ddp_find_unused_parameters": False,
                        "gradient_accumulation_steps": 4,
                        "lr_schedule": "linear_warmup_cosine_decay",
                        "warmup_steps": 500,
                    },
                ),
                _req(
                    "D1",
                    "1B Global KV routed pilot",
                    stage="stage5_global_kv",
                    mode="scheduled",
                    model_flags={
                        "model_name": "brian_r1b",
                        "global_kv": True,
                        "parallel_passing": False,
                        "top_k": 2,
                    },
                    train_flags={
                        "precision": "bf16",
                        "activation_checkpointing": True,
                        "ddp_find_unused_parameters": True,
                        "gradient_accumulation_steps": 4,
                        "lr_schedule": "linear_warmup_cosine_decay",
                        "warmup_steps": 500,
                    },
                ),
            ],
        )
    if profile == "scale_r125_5b_followup":
        return _exact_id_requirements(
            entries,
            [
                _req(
                    "A10",
                    "125M fixed decoder-only baseline on 5B data",
                    stage="stage0_baseline",
                    mode="baseline",
                    model_flags={"model_name": "baseline_125m"},
                    data_flags={
                        "recipe_name": "r125_main_5b",
                        "target_tokens": 5_000_000_000,
                        "sequence_length": 2048,
                    },
                ),
                _req(
                    "A11",
                    "125M routed output-action follow-up on 5B data",
                    stage="stage4_output_action",
                    mode="scheduled",
                    model_flags={
                        "model_name": "brian_r125_top2",
                        "top_k": 2,
                        "global_kv": False,
                        "parallel_passing": False,
                    },
                    data_flags={
                        "recipe_name": "r125_main_5b",
                        "target_tokens": 5_000_000_000,
                        "sequence_length": 2048,
                    },
                ),
            ],
        )
    if profile == "scale_r350_30b_followup":
        return _exact_id_requirements(
            entries,
            [
                _req(
                    "B5",
                    "350M fixed decoder-only baseline on 30B data",
                    stage="stage0_baseline",
                    mode="baseline",
                    model_flags={"model_name": "baseline_350m"},
                    data_flags={
                        "recipe_name": "r350_main_30b",
                        "target_tokens": 30_000_000_000,
                        "sequence_length": 4096,
                    },
                ),
                _req(
                    "B6",
                    "350M routed output-action follow-up on 30B data",
                    stage="stage4_output_action",
                    mode="scheduled",
                    model_flags={
                        "model_name": "brian_r350_top2",
                        "top_k": 2,
                        "global_kv": False,
                        "parallel_passing": False,
                    },
                    data_flags={
                        "recipe_name": "r350_main_30b",
                        "target_tokens": 30_000_000_000,
                        "sequence_length": 4096,
                    },
                ),
            ],
        )
    if profile == "package_d_r1b_main_validation":
        return _exact_id_requirements(
            entries,
            [
                _req(
                    "D2",
                    "1B fixed decoder-only baseline on 50B data",
                    stage="stage0_baseline",
                    mode="baseline",
                    model_flags={"model_name": "baseline_1b"},
                    data_flags={
                        "recipe_name": "r1b_main_50b",
                        "target_tokens": 50_000_000_000,
                        "sequence_length": 4096,
                    },
                    train_flags={
                        "precision": "bf16",
                        "activation_checkpointing": True,
                        "ddp_find_unused_parameters": False,
                        "gradient_accumulation_steps": 4,
                        "lr_schedule": "linear_warmup_cosine_decay",
                        "warmup_steps": 2000,
                    },
                ),
                _req(
                    "D3",
                    "1B Global KV main validation on 50B data",
                    stage="stage5_global_kv",
                    mode="scheduled",
                    model_flags={
                        "model_name": "brian_r1b",
                        "global_kv": True,
                        "parallel_passing": False,
                        "top_k": 2,
                    },
                    data_flags={
                        "recipe_name": "r1b_main_50b",
                        "target_tokens": 50_000_000_000,
                        "sequence_length": 4096,
                    },
                    train_flags={
                        "precision": "bf16",
                        "activation_checkpointing": True,
                        "ddp_find_unused_parameters": True,
                        "gradient_accumulation_steps": 4,
                        "lr_schedule": "linear_warmup_cosine_decay",
                        "warmup_steps": 2000,
                    },
                ),
            ],
        )
    return []


def _req(
    entry_id: str,
    description: str,
    *,
    stage: str | None = None,
    mode: str | None = None,
    routing_flags: dict[str, Any] | None = None,
    model_flags: dict[str, Any] | None = None,
    data_flags: dict[str, Any] | None = None,
    loss_weights: dict[str, float] | None = None,
    train_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "description": description,
        "stage": stage,
        "mode": mode,
        "routing_flags": routing_flags or {},
        "model_flags": model_flags or {},
        "data_flags": data_flags or {},
        "loss_weights": loss_weights or {},
        "train_flags": train_flags or {},
    }


def _exact_id_requirements(entries: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {entry["id"]: entry for entry in entries}
    return [_requirement_row(spec, [by_id[spec["entry_id"]]] if spec["entry_id"] in by_id else []) for spec in specs]


def _global_kv_requirements(plan: ExperimentPlan, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    global_entries = [entry for entry in entries if entry["model"].get("global_kv") is True]
    window_sweep_entries = [
        entry
        for entry in global_entries
        if str(entry["id"]).lower().startswith("c5")
        and _num(entry["model"].get("global_window_slots")) is not None
        and _num(entry["model"].get("global_window_slots")) > 0
    ]
    all_window_entries = [
        entry
        for entry in global_entries
        if _num(entry["model"].get("global_window_slots")) is not None
        and _num(entry["model"].get("global_window_slots")) > 0
    ]
    distinct_windows = sorted(
        {
            int(_num(entry["model"].get("global_window_slots")) or 0)
            for entry in window_sweep_entries
            if _num(entry["model"].get("global_window_slots")) is not None
        }
    )
    return [
        _group_requirement(
            "C0",
            "local KV only baseline",
            entries,
            lambda entry: entry["id"] == "C0" and entry["model"].get("global_kv") is not True,
            plan_aliases=["K0"],
        ),
        _group_requirement(
            "C1",
            "local plus global uncompressed",
            entries,
            lambda entry: entry["id"] == "C1"
            and entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_code_dim")) is not None
            and _num(entry["model"].get("base_d_model")) is not None
            and _num(entry["model"].get("global_code_dim")) == _num(entry["model"].get("base_d_model")),
            plan_aliases=["K1"],
        ),
        _group_requirement(
            "C2",
            "local plus global compressed",
            entries,
            lambda entry: entry["id"] == "C2"
            and entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_code_dim")) is not None
            and _num(entry["model"].get("base_d_model")) is not None
            and _num(entry["model"].get("global_code_dim")) < _num(entry["model"].get("base_d_model")),
            plan_aliases=["K2"],
        ),
        _group_requirement(
            "C3",
            "global KV without sink",
            entries,
            lambda entry: entry["id"] == "C3"
            and entry["model"].get("global_kv") is True
            and _num(entry["model"].get("global_sink_slots")) == 0,
            plan_aliases=["K3"],
        ),
        _group_requirement(
            "C4",
            "global KV with sink and window",
            entries,
            lambda entry: entry["id"] == "C4"
            and entry["model"].get("global_kv") is True
            and (_num(entry["model"].get("global_sink_slots")) or 0) > 0
            and (_num(entry["model"].get("global_window_slots")) or 0) > 0,
            plan_aliases=["K4"],
        ),
        {
            "id": "C5",
            "description": "global KV window size sweep",
            "status": "pass" if len(distinct_windows) >= 2 else "fail",
            "matched_entry_ids": [entry["id"] for entry in window_sweep_entries],
            "plan_aliases": ["K5"],
            "checks": {
                "at_least_two_window_sizes": len(distinct_windows) >= 2,
                "distinct_global_window_slots": distinct_windows,
                "global_window_entry_ids": [entry["id"] for entry in all_window_entries],
            },
        },
        _group_requirement(
            "C6",
            "global KV per-block adapter",
            entries,
            lambda entry: entry["id"] == "C6"
            and entry["model"].get("global_kv") is True
            and entry["model"].get("global_adapter_scope") == "per_block"
            and (_num(entry["model"].get("global_head_delta_rank")) or 0) == 0,
            plan_aliases=["K6"],
        ),
        _group_requirement(
            "C7",
            "global KV per-head low-rank delta",
            entries,
            lambda entry: entry["id"] == "C7"
            and entry["model"].get("global_kv") is True
            and entry["model"].get("global_adapter_scope", "shared") == "shared"
            and (_num(entry["model"].get("global_head_delta_rank")) or 0) > 0,
            plan_aliases=["K7"],
        ),
        _group_requirement(
            "C8",
            "global KV per-block plus per-head low-rank delta",
            entries,
            lambda entry: entry["id"] == "C8"
            and entry["model"].get("global_kv") is True
            and entry["model"].get("global_adapter_scope") == "per_block"
            and (_num(entry["model"].get("global_head_delta_rank")) or 0) > 0,
            plan_aliases=["K8"],
        ),
        {
            "id": "baseline_train_config",
            "description": "local baseline train config is declared for compute comparison",
            "status": "pass" if plan.baseline_train_config is not None else "fail",
            "matched_entry_ids": [],
            "checks": {"baseline_train_config_present": plan.baseline_train_config is not None},
        },
    ]


def _position_requirements(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _exact_id_requirements(
        entries,
        [
            _req(
                "P0",
                "no block-position state",
                mode="scheduled",
                model_flags={
                    "block_position_mode": "none",
                    "position_to_router": False,
                    "position_to_blocks": False,
                },
            ),
            _req("P1", "random position initialization", mode="scheduled", model_flags={"block_position_mode": "random"}),
            _req("P2", "open-arc position initialization", mode="scheduled", model_flags={"block_position_mode": "open_arc"}),
            _req("P3", "circular position initialization", mode="scheduled", model_flags={"block_position_mode": "circular"}),
            _req(
                "P4",
                "position only to router",
                mode="scheduled",
                model_flags={"position_to_router": True, "position_to_blocks": False},
            ),
            _req(
                "P5",
                "position to router and blocks",
                mode="scheduled",
                model_flags={"position_to_router": True, "position_to_blocks": True},
            ),
            _req("P6", "no location bias", mode="scheduled", model_flags={"location_bias_weight": 0.0}),
            _req("P7", "no location loss", mode="scheduled", loss_weights={"location": 0.0}),
            _req(
                "P8",
                "direct position-hidden addition",
                mode="scheduled",
                model_flags={"block_position_injection": "direct_add"},
            ),
            _req(
                "P9",
                "separate position state",
                mode="scheduled",
                model_flags={
                    "block_position_injection": "adapter",
                    "position_to_router": True,
                    "position_to_blocks": True,
                },
            ),
        ],
    )


def _cost_control_requirements(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _exact_id_requirements(
        entries,
        [
            _req(
                "C0",
                "no route cost pressure",
                stage="stage4_output_action",
                mode="scheduled",
                loss_weights={"cost": 0.0},
            ),
            _req(
                "C1",
                "light route cost pressure",
                stage="stage4_output_action",
                mode="scheduled",
                loss_weights={"cost": 0.001},
            ),
            _req(
                "C2",
                "default route cost pressure",
                stage="stage4_output_action",
                mode="scheduled",
                loss_weights={"cost": 0.01},
            ),
            _req(
                "C3",
                "strong route cost pressure",
                stage="stage4_output_action",
                mode="scheduled",
                loss_weights={"cost": 0.05},
            ),
        ],
    )


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
            and (_num(entry["model"].get("branch_cost")) or 0.0) > 0.0
            and _branch_score_decay_enabled(entry),
        ),
        _group_requirement(
            "PP2",
            "beam-4 parallel capacity test",
            entries,
            lambda entry: entry["id"] == "PP2"
            and entry["model"].get("parallel_passing") is True
            and int(_num(entry["model"].get("beam_size")) or 0) == 4
            and _branch_score_decay_enabled(entry),
        ),
        _group_requirement(
            "PP3",
            "branch cost off ablation",
            entries,
            lambda entry: entry["id"] == "PP3"
            and entry["model"].get("parallel_passing") is True
            and _num(entry["model"].get("branch_cost")) == 0.0
            and _branch_score_decay_enabled(entry)
            and _num(entry.get("loss_weights", {}).get("cost")) == 0.0,
        ),
        _group_requirement(
            "PP4",
            "branch cost on proposal",
            entries,
            lambda entry: entry["id"] == "PP4"
            and entry["model"].get("parallel_passing") is True
            and (_num(entry["model"].get("branch_cost")) or 0.0) > 0.0
            and _branch_score_decay_enabled(entry)
            and (_num(entry.get("loss_weights", {}).get("cost")) or 0.0) > 0.0,
        ),
        _group_requirement(
            "PP5",
            "top-1 OUT terminal rule",
            entries,
            lambda entry: entry["id"] == "PP5"
            and entry["model"].get("parallel_passing") is True
            and _branch_score_decay_enabled(entry)
            and entry["model"].get("parallel_exit_policy") == "top1",
        ),
        _group_requirement(
            "PP6",
            "OUT in top-k terminal rule",
            entries,
            lambda entry: entry["id"] == "PP6"
            and entry["model"].get("parallel_passing") is True
            and _branch_score_decay_enabled(entry)
            and entry["model"].get("parallel_exit_policy") == "any_topk",
        ),
        _group_requirement(
            "PP7",
            "shared base Global KV plus branch delta memory",
            entries,
            lambda entry: entry["id"] == "PP7"
            and entry["model"].get("parallel_passing") is True
            and _branch_score_decay_enabled(entry)
            and entry["model"].get("global_kv") is True
            and (_num(entry["model"].get("global_window_slots")) or 0) > 0,
        ),
    ]


def _branch_score_decay_enabled(entry: dict[str, Any]) -> bool:
    decay = _num(entry["model"].get("branch_score_decay"))
    return decay is not None and 0.0 < decay < 1.0


def _group_requirement(
    requirement_id: str,
    description: str,
    entries: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    plan_aliases: list[str] | None = None,
) -> dict[str, Any]:
    matched = [entry for entry in entries if predicate(entry)]
    return {
        "id": requirement_id,
        "description": description,
        "status": "pass" if matched else "fail",
        "matched_entry_ids": [entry["id"] for entry in matched],
        "plan_aliases": plan_aliases or [],
        "checks": {"matched": bool(matched)},
    }


def _requirement_row(spec: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    entry = matches[0] if matches else None
    checks = {
        "entry_present": entry is not None,
        "stage_matches": _check_stage(entry, spec.get("stage")),
        "mode_matches": _check_mode(entry, spec.get("mode")),
        "routing_flags_match": _check_mapping(entry, "routing", spec["routing_flags"]),
        "model_flags_match": _check_mapping(entry, "model", spec["model_flags"]),
        "data_flags_match": _check_mapping(entry, "data", spec["data_flags"]),
        "loss_weights_match": _check_mapping(entry, "loss_weights", spec["loss_weights"]),
        "train_flags_match": _check_mapping(entry, "train", spec["train_flags"]),
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
    return {**entry.to_json(), **_summarize_train_config(Path(entry.train_config))}


def _summarize_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"id": "baseline", "role": "baseline", **_summarize_train_config(path)}


def _summarize_train_config(train_config_path: Path) -> dict[str, Any]:
    train_config_path = Path(train_config_path)
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
    data_config_path = _resolve_optional_reference(train_config.get("data_config"), train_config_path)
    model_config, model_config_error = _load_optional_config_with_error(model_config_path)
    data_config, data_config_error = _load_optional_config_with_error(data_config_path)
    model_base_config_path = _resolve_model_base_reference(model_config, model_config_path)
    model_base_config, model_base_config_error = _load_optional_config_with_error(model_base_config_path)
    routing = train_config.get("routing") if isinstance(train_config.get("routing"), dict) else {}
    loss_weights = train_config.get("loss_weights") if isinstance(train_config.get("loss_weights"), dict) else {}
    return {
        "train_config": str(train_config_path),
        "stage": stage or None,
        "train_mode": train_mode,
        "routing_mode": routing.get("mode"),
        "routing": {
            "mode": routing.get("mode"),
            "pseudo_policy": routing.get("pseudo_policy"),
            "hard_exit": routing.get("hard_exit"),
        },
        "train": {
            "precision": train_config.get("precision", "fp32"),
            "activation_checkpointing": train_config.get("activation_checkpointing", False),
            "ddp_find_unused_parameters": train_config.get("ddp_find_unused_parameters", False),
            "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 1),
            "lr_schedule": train_config.get("lr_schedule", "constant"),
            "warmup_steps": train_config.get("warmup_steps", 0),
        },
        "model_config": str(model_config_path) if model_config_path else None,
        "model_base_config": str(model_base_config_path) if model_base_config_path else None,
        "data_config": str(data_config_path) if data_config_path else None,
        "model": _model_summary(model_config, model_base_config),
        "data": _data_summary(data_config),
        "loss_weights": {key: _num(value) for key, value in loss_weights.items()},
        "checks": {
            "train_config_exists": train_config_path.exists(),
            "train_config_loads": train_config_path.exists() and train_config_error is None,
            "train_mode_resolves": train_mode is not None and train_mode_error is None,
            "model_config_exists": model_config_path.exists() if model_config_path else False,
            "model_config_loads": model_config_path.exists() and model_config_error is None if model_config_path else False,
            "model_config_valid": _model_config_valid(model_config),
            "model_base_config_exists": model_base_config_path.exists() if model_base_config_path else True,
            "model_base_config_loads": (
                model_base_config_path.exists() and model_base_config_error is None if model_base_config_path else True
            ),
            "data_config_exists": data_config_path.exists() if data_config_path else False,
            "data_config_loads": data_config_path.exists() and data_config_error is None if data_config_path else False,
        },
        "errors": {
            "train_config": train_config_error,
            "train_mode": train_mode_error,
            "model_config": model_config_error,
            "model_base_config": model_base_config_error,
            "data_config": data_config_error,
        },
    }


def _baseline_check(baseline: dict[str, Any] | None, key: str) -> bool:
    if baseline is None:
        return False
    checks = baseline.get("checks")
    return bool(isinstance(checks, dict) and checks.get(key) is True)


def _resolve_optional_reference(value: Any, source_config: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (source_config.parent / path).resolve()


def _load_optional_config(path: Path | None) -> dict[str, Any]:
    config, _ = _load_optional_config_with_error(path)
    return config


def _load_optional_config_with_error(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None or not path.exists():
        return {}, None
    try:
        return load_config(path), None
    except Exception as exc:  # pragma: no cover - defensive report evidence
        return {}, str(exc)


def _resolve_model_base_reference(config: dict[str, Any], model_config_path: Path | None) -> Path | None:
    if model_config_path is None:
        return None
    return _resolve_optional_reference(config.get("base_config"), model_config_path)


def _model_summary(config: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model_name",
        "architecture",
        "base_config",
        "pre_blocks",
        "route_pool_blocks",
        "post_blocks",
        "block_position_dim",
        "block_position_mode",
        "block_position_injection",
        "position_to_router",
        "position_to_blocks",
        "location_bias_weight",
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
        "branch_score_decay",
        "parallel_exit_policy",
    ]
    summary = {key: config.get(key) for key in keys if key in config}
    if base_config and "d_model" in base_config:
        summary["base_d_model"] = base_config.get("d_model")
    elif isinstance(config.get("base"), dict):
        base = config["base"]
        if "d_model" in base:
            summary["base_d_model"] = base.get("d_model")
    estimated_parameters = _estimate_parameter_count(config, base_config)
    if estimated_parameters is not None:
        summary["estimated_parameter_count"] = estimated_parameters
    planned_range = _planned_parameter_range(summary.get("model_name"))
    if planned_range is not None:
        minimum, maximum = planned_range
        summary["planned_parameter_range"] = {"min": minimum, "max": maximum}
        summary["estimated_parameter_count_in_plan_range"] = (
            estimated_parameters is not None and minimum <= estimated_parameters <= maximum
        )
    return summary


def _estimate_parameter_count(config: dict[str, Any], base_config: dict[str, Any]) -> int | None:
    architecture = config.get("architecture")
    if architecture == "decoder_only_llama_like":
        return _estimate_baseline_parameters(config)
    if architecture != "brian_route_core":
        return None
    base = base_config if base_config else config.get("base")
    if not isinstance(base, dict):
        return None
    base_count = _estimate_baseline_parameters(base)
    route_pool_blocks = _int_or_none(config.get("route_pool_blocks"))
    block_position_dim = _int_or_none(config.get("block_position_dim"))
    if base_count is None or route_pool_blocks is None or block_position_dim is None:
        return None
    d_model = _int_or_none(base.get("d_model"))
    n_heads = _int_or_none(base.get("n_heads"))
    if d_model is None or n_heads is None:
        return None
    position_injection = str(config.get("block_position_injection", "adapter"))
    position_adapter = 0 if position_injection == "direct_add" else block_position_dim * d_model
    num_actions = route_pool_blocks + 1
    route_adapter_params = route_pool_blocks * position_adapter
    exit_block_params = position_adapter + d_model + (d_model * d_model)
    position_table_params = num_actions * block_position_dim
    router_params = ((d_model + block_position_dim) * d_model) + d_model + (d_model * num_actions) + num_actions
    global_params = 0
    if config.get("global_kv") is True:
        code_dim = _int_or_none(config.get("global_code_dim")) or block_position_dim
        head_delta_rank = _int_or_none(config.get("global_head_delta_rank")) or 0
        adapter_count = num_actions if config.get("global_adapter_scope", "shared") == "per_block" else 1
        global_params = adapter_count * _global_adapter_parameters(
            d_model,
            code_dim,
            n_heads=n_heads,
            head_delta_rank=head_delta_rank,
        )
    return base_count + route_adapter_params + exit_block_params + position_table_params + router_params + global_params


def _estimate_baseline_parameters(config: dict[str, Any]) -> int | None:
    vocab_size = _int_or_none(config.get("vocab_size"))
    layers = _int_or_none(config.get("layers"))
    d_model = _int_or_none(config.get("d_model"))
    if vocab_size is None or layers is None or d_model is None:
        return None
    return (vocab_size * d_model) + (layers * _transformer_block_parameters(d_model)) + d_model


def _transformer_block_parameters(d_model: int) -> int:
    hidden_dim = int(math.ceil(4.0 * d_model / 256) * 256)
    if d_model < 256:
        hidden_dim = int(4.0 * d_model)
    return (4 * d_model * d_model) + (3 * d_model * hidden_dim) + (2 * d_model)


def _global_adapter_parameters(d_model: int, code_dim: int, *, n_heads: int, head_delta_rank: int) -> int:
    write = d_model * code_dim
    read = (d_model * code_dim) + (code_dim * d_model) + 1
    if head_delta_rank > 0:
        head_dim = d_model // n_heads
        write += (n_heads * head_dim * head_delta_rank) + (n_heads * head_delta_rank * code_dim)
        read += (n_heads * code_dim * head_delta_rank) + (n_heads * head_delta_rank * head_dim)
    return write + read


def _planned_parameter_range(model_name: Any) -> tuple[int, int] | None:
    name = str(model_name or "").lower()
    for scale, planned_range in PLANNED_PARAMETER_RANGES.items():
        if scale in name:
            return planned_range
    return None


def _planned_parameter_estimates_in_range(items: list[dict[str, Any] | None]) -> bool:
    scoped = [
        item
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("model"), dict)
        and _planned_parameter_range(item["model"].get("model_name")) is not None
    ]
    if not scoped:
        return True
    return all(item["model"].get("estimated_parameter_count_in_plan_range") is True for item in scoped)


def _model_config_valid(config: dict[str, Any]) -> bool:
    architecture = config.get("architecture")
    if architecture == "decoder_only_llama_like":
        required_keys = {
            "model_name",
            "architecture",
            "layers",
            "d_model",
            "n_heads",
            "context_length",
            "vocab_size",
        }
        return required_keys.issubset(config)
    if architecture == "brian_route_core":
        required_keys = {
            "model_name",
            "architecture",
            "pre_blocks",
            "route_pool_blocks",
            "post_blocks",
            "block_position_dim",
            "max_route_steps",
        }
        return required_keys.issubset(config) and ("base_config" in config or isinstance(config.get("base"), dict))
    return False


def _data_summary(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "recipe_name",
        "output_dir",
        "manifest_path",
        "sequence_length",
        "target_tokens",
        "validation_tokens",
    ]
    return {key: config.get(key) for key in keys if key in config}


def _data_configs_consistent(entries: list[dict[str, Any]]) -> bool:
    paths = {entry.get("data_config") for entry in entries if entry.get("data_config")}
    return len(paths) == 1


def _baseline_data_config_consistent(baseline: dict[str, Any] | None, entries: list[dict[str, Any]]) -> bool:
    if baseline is None or not baseline.get("data_config"):
        return False
    paths = {entry.get("data_config") for entry in entries if entry.get("data_config")}
    paths.add(baseline["data_config"])
    return len(paths) == 1


def _overall_status(checks: dict[str, bool], requirements: list[dict[str, Any]]) -> str:
    if checks.get("profile_known") is not True:
        return "fail"
    if any(requirement["status"] == "fail" for requirement in requirements):
        return "fail"
    for critical_check in [
        "train_configs_exist",
        "train_configs_load",
        "train_modes_resolve",
        "model_configs_exist",
        "model_configs_load",
        "model_configs_valid",
        "model_base_configs_exist",
        "model_base_configs_load",
        "data_configs_exist",
        "data_configs_load",
        "data_configs_consistent",
        "baseline_train_config_present",
        "baseline_train_config_exists",
        "baseline_train_config_loads",
        "baseline_train_mode_resolves",
        "baseline_model_config_exists",
        "baseline_model_config_loads",
        "baseline_model_config_valid",
        "baseline_model_base_config_exists",
        "baseline_model_base_config_loads",
        "baseline_data_config_exists",
        "baseline_data_config_loads",
        "baseline_data_config_consistent",
        "planned_parameter_estimates_in_range",
    ]:
        if checks.get(critical_check) is False:
            return "fail"
    if checks and all(value is True for value in checks.values()):
        return "pass"
    if any(value is True for value in checks.values()):
        return "warn"
    return "fail"


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    number = _num(value)
    if number is None or not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)
