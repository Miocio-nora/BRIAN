from __future__ import annotations

from pathlib import Path
from typing import Any

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.utils.config import load_config


def build_model_from_config(model_config_path: str | Path) -> Any:
    model_config_path = Path(model_config_path)
    config = load_config(model_config_path)
    architecture = config.get("architecture")
    if architecture == "decoder_only_llama_like":
        return BaselineLM(BaselineConfig.from_dict(config))
    if architecture == "brian_route_core":
        return BrianRouteCore(BrianRouteConfig.from_dict(config, config_dir=model_config_path.parent))
    raise ValueError(f"Unknown architecture: {architecture}")


def train_mode_for_stage(stage: str) -> str:
    if stage == "stage0_baseline":
        return "baseline"
    if stage == "stage1_fixed_route":
        return "fixed"
    if stage in {"stage2_router_imitation", "stage3_pseudo_skip_recur"}:
        return "pseudo"
    if stage in {
        "stage3_scheduled_free_routing",
        "stage4_scheduled_free_routing",
        "stage4_coverage_free_sphere",
        "stage4_output_action",
        "stage5_output_action",
        "stage5_global_kv",
        "stage5_attention_global_kv",
    }:
        return "scheduled"
    if stage == "stage4_pure_free_sphere":
        return "free"
    if stage in {"stage6_parallel_passing", "stage7_parallel_passing"}:
        return "parallel"
    raise ValueError(f"Unsupported executable stage: {stage}")
