import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.model.brian_model import BrianRouteConfig
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config


CONFIG_ROOT = Path("configs")


def test_all_yaml_configs_load() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT):
        try:
            load_config(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    assert errors == []


def test_model_configs_have_known_architecture_and_base_refs() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT / "model"):
        config = _load_config(path, errors)
        _validate_model_config(config, path, errors)
        if config.get("architecture") == "brian_route_core" and "base_config" in config:
            base_path = _resolve_reference(config, "base_config", path, errors)
            if base_path is not None:
                base_config = _load_config(base_path, errors)
                if base_config.get("architecture") != "decoder_only_llama_like":
                    errors.append(f"{path}: base_config must resolve to a decoder_only_llama_like config")

    assert errors == []


def test_planned_model_architecture_ladder_is_declared() -> None:
    baseline_expected = {
        "baseline_125m.yaml": {"layers": 12, "d_model": 768, "n_heads": 12, "context_length": 2048},
        "baseline_350m.yaml": {"layers": 24, "d_model": 960, "n_heads": 16, "context_length": 4096},
        "baseline_1b.yaml": {"layers": 32, "d_model": 1536, "n_heads": 24, "context_length": 4096},
    }
    route_expected = {
        "brian_r125.yaml": {
            "base_config": "baseline_125m.yaml",
            "pre_blocks": 2,
            "route_pool_blocks": 8,
            "post_blocks": 2,
            "block_position_dim": 64,
            "max_route_steps": 4,
            "top_k": 1,
            "later_top_k": 2,
            "global_kv": False,
            "parallel_passing": False,
        },
        "brian_r350.yaml": {
            "base_config": "baseline_350m.yaml",
            "pre_blocks": 4,
            "route_pool_blocks": 16,
            "post_blocks": 4,
            "block_position_dim": 128,
            "max_route_steps": 8,
            "top_k": 1,
            "later_top_k": 2,
            "global_kv": False,
            "global_kv_phase": "phase_2_only",
            "parallel_passing": False,
        },
        "brian_r1b.yaml": {
            "base_config": "baseline_1b.yaml",
            "pre_blocks": 6,
            "route_pool_blocks": 20,
            "post_blocks": 6,
            "block_position_dim": 256,
            "max_route_steps": 12,
            "top_k": 2,
            "later_top_k": 2,
            "global_kv": True,
            "global_kv_phase": "after_route_core",
            "parallel_passing": False,
            "parallel_passing_phase": "experimental_only",
        },
    }

    for filename, expected in baseline_expected.items():
        config = load_config(CONFIG_ROOT / "model" / filename)
        assert config["architecture"] == "decoder_only_llama_like"
        assert config["ffn_type"] == "swiglu"
        assert config["norm"] == "rmsnorm"
        assert config["token_position"] == "rope"
        assert config["vocab_size"] == 32000
        for key, value in expected.items():
            assert config[key] == value

    for filename, expected in route_expected.items():
        config = load_config(CONFIG_ROOT / "model" / filename)
        assert config["architecture"] == "brian_route_core"
        assert config["block_position_mode"] == "open_arc"
        assert config["block_position_injection"] == "adapter"
        assert config["position_to_router"] is True
        assert config["position_to_blocks"] is True
        assert config["hard_exit"] is False
        for key, value in expected.items():
            assert config[key] == value


def test_parallel_passing_ablation_configs_keep_planned_branch_controls() -> None:
    expected = {
        "stage6_parallel_passing.yaml": {
            "model_name": "brian_r125_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "ablation_pp2_parallel_beam4.yaml": {
            "model_name": "brian_r125_parallel_beam4",
            "beam_size": 4,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "ablation_pp3_parallel_cost_off.yaml": {
            "model_name": "brian_r125_parallel_no_branch_cost",
            "beam_size": 2,
            "branch_cost": 0.0,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.0,
        },
        "ablation_pp4_parallel_cost_on.yaml": {
            "model_name": "brian_r125_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "ablation_pp5_parallel_top1_exit.yaml": {
            "model_name": "brian_r125_parallel_top1_exit",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "top1",
            "loss_cost": 0.01,
        },
        "ablation_pp6_parallel_any_topk_exit.yaml": {
            "model_name": "brian_r125_parallel_any_topk_exit",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "any_topk",
            "loss_cost": 0.01,
        },
        "ablation_pp7_parallel_global_kv_delta.yaml": {
            "model_name": "brian_r125_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "stage6_tiny_debug.yaml": {
            "model_name": "brian_tiny_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "stage6_tiny_parallel_beam4.yaml": {
            "model_name": "brian_tiny_parallel_beam4",
            "beam_size": 4,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "stage6_tiny_parallel_cost_off.yaml": {
            "model_name": "brian_tiny_parallel_no_branch_cost",
            "beam_size": 2,
            "branch_cost": 0.0,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.0,
        },
        "stage6_tiny_parallel_cost_on.yaml": {
            "model_name": "brian_tiny_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
        "stage6_tiny_parallel_top1_exit.yaml": {
            "model_name": "brian_tiny_parallel_top1_exit",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "top1",
            "loss_cost": 0.01,
        },
        "stage6_tiny_parallel_any_topk_exit.yaml": {
            "model_name": "brian_tiny_parallel_any_topk_exit",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "any_topk",
            "loss_cost": 0.01,
        },
        "stage6_tiny_parallel_global_kv_delta.yaml": {
            "model_name": "brian_tiny_parallel",
            "beam_size": 2,
            "branch_cost": 0.01,
            "parallel_exit_policy": "branch",
            "loss_cost": 0.01,
        },
    }

    for filename, expected_values in expected.items():
        train_path = CONFIG_ROOT / "train" / filename
        train_config = load_config(train_path)
        model_config = load_config(train_path.parent / train_config["model_config"])
        assert train_config["stage"] == "stage6_parallel_passing"
        assert train_config["routing"]["mode"] == "parallel"
        assert train_config["routing"]["hard_exit"] is True
        assert train_config["loss_weights"]["route"] == 0.0
        assert train_config["loss_weights"]["cost"] == expected_values["loss_cost"]
        assert model_config["parallel_passing"] is True
        assert model_config["global_kv"] is True
        assert model_config["global_window_slots"] > 0
        assert model_config["model_name"] == expected_values["model_name"]
        assert model_config["beam_size"] == expected_values["beam_size"]
        assert model_config["branch_cost"] == expected_values["branch_cost"]
        assert model_config["branch_score_decay"] == 0.99
        assert model_config.get("parallel_exit_policy", "branch") == expected_values["parallel_exit_policy"]


def test_train_configs_resolve_stage_model_and_data_refs() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT / "train"):
        config = _load_config(path, errors)
        _validate_train_config(config, path, errors)
        for key in ["stage", "model_config", "data_config"]:
            if key not in config:
                errors.append(f"{path}: missing {key}")
        if "stage" in config:
            try:
                train_mode_for_stage(str(config["stage"]))
            except ValueError as exc:
                errors.append(f"{path}: {exc}")

        model_path = _resolve_reference(config, "model_config", path, errors)
        if model_path is not None:
            model_config = _load_config(model_path, errors)
            _validate_model_config(model_config, model_path, errors)

        data_path = _resolve_reference(config, "data_config", path, errors)
        if data_path is not None:
            data_config = _load_config(data_path, errors)
            _validate_data_config(data_config, data_path, errors)

    assert errors == []


def test_first_milestone_train_configs_match_guide_sequence() -> None:
    expected = {
        "stage0_baseline.yaml": {
            "stage": "stage0_baseline",
            "mode": None,
            "pseudo_policy": None,
        },
        "stage1_fixed_route.yaml": {
            "stage": "stage1_fixed_route",
            "mode": "fixed",
            "pseudo_policy": "sequential",
        },
        "stage2_router_imitation.yaml": {
            "stage": "stage2_router_imitation",
            "mode": "pseudo",
            "pseudo_policy": "sequential",
        },
        "stage3_pseudo_skip_recur.yaml": {
            "stage": "stage3_pseudo_skip_recur",
            "mode": "pseudo",
            "pseudo_policy": "mixed_skip_recur",
        },
        "stage3_scheduled_free_routing.yaml": {
            "stage": "stage3_scheduled_free_routing",
            "mode": "scheduled",
            "pseudo_policy": "mixed_skip_recur",
        },
    }

    for filename, values in expected.items():
        config = load_config(CONFIG_ROOT / "train" / filename)
        routing = config.get("routing", {})
        assert config["stage"] == values["stage"]
        assert routing.get("mode") == values["mode"]
        assert routing.get("pseudo_policy") == values["pseudo_policy"]

    guidance = Path("CODEX_GUIDANCE.md").read_text(encoding="utf-8")
    assert "configs/train/stage2_router_imitation.yaml" in guidance
    assert "configs/train/stage3_pseudo_skip_recur.yaml" in guidance
    assert "configs/train/stage3_scheduled_free_routing.yaml" in guidance
    assert "--run <stage3_scheduled_free_routing_run>" in guidance

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "<stage2_sequential_router_imitation_run>" in readme
    assert "<stage3_pseudo_skip_recur_run>" in readme
    assert "<stage3_scheduled_free_routing_run>" in readme
    assert "<stage2_run>" not in readme
    assert "<stage3_run>" not in readme


def test_scaled_train_configs_keep_b200_memory_controls() -> None:
    r350_configs = [
        "stage0_r350_baseline.yaml",
        "stage4_r350_output_action.yaml",
        "stage0_r350_main30b_baseline.yaml",
        "stage4_r350_main30b_output_action.yaml",
    ]
    for filename in r350_configs:
        config = load_config(CONFIG_ROOT / "train" / filename)
        assert config["precision"] == "bf16"
        assert config["batch_size"] == 1
        assert config["lr_schedule"] == "linear_warmup_cosine_decay"
        assert config["warmup_steps"] > 0

    r1b_expected = {
        "stage0_r1b_baseline.yaml": {
            "model_config": "../model/baseline_1b.yaml",
            "data_config": "../data/r1b_pilot.yaml",
            "ddp_find_unused_parameters": False,
            "warmup_steps": 500,
        },
        "stage5_r1b_global_kv_pilot.yaml": {
            "model_config": "../model/brian_r1b.yaml",
            "data_config": "../data/r1b_pilot.yaml",
            "ddp_find_unused_parameters": True,
            "warmup_steps": 500,
        },
        "stage0_r1b_main50b_baseline.yaml": {
            "model_config": "../model/baseline_1b.yaml",
            "data_config": "../data/r1b_main_50b.yaml",
            "ddp_find_unused_parameters": False,
            "warmup_steps": 2000,
        },
        "stage5_r1b_main50b_global_kv.yaml": {
            "model_config": "../model/brian_r1b.yaml",
            "data_config": "../data/r1b_main_50b.yaml",
            "ddp_find_unused_parameters": True,
            "warmup_steps": 2000,
        },
    }
    for filename, expected in r1b_expected.items():
        config = load_config(CONFIG_ROOT / "train" / filename)
        assert config["precision"] == "bf16"
        assert config["batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 4
        assert config["activation_checkpointing"] is True
        assert config["lr_schedule"] == "linear_warmup_cosine_decay"
        model_config = load_config((CONFIG_ROOT / "train" / filename).parent / config["model_config"])
        assert model_config.get("parallel_passing", False) is False
        for key, value in expected.items():
            assert config[key] == value


def test_scheduled_train_configs_keep_curriculum_schedule() -> None:
    errors: list[str] = []
    scheduled_paths: list[Path] = []
    for path in _yaml_files(CONFIG_ROOT / "train"):
        config = load_config(path)
        routing = config.get("routing", {})
        if isinstance(routing, dict) and routing.get("mode") == "scheduled":
            scheduled_paths.append(path)
            _validate_scheduled_routing_config(routing, path, errors)

    assert scheduled_paths
    assert errors == []


def test_output_action_and_later_train_configs_keep_hard_exit_enabled() -> None:
    hard_exit_stages = {
        "stage4_coverage_free_sphere",
        "stage4_pure_free_sphere",
        "stage4_output_action",
        "stage5_output_action",
        "stage5_global_kv",
        "stage5_attention_global_kv",
        "stage6_parallel_passing",
        "stage7_parallel_passing",
    }
    no_hard_exit_stages = {"stage4_scheduled_free_routing"}
    hard_exit_ablations = {
        CONFIG_ROOT / "train" / "corrected_package_a_r125_2b_aout_no_hard_exit.yaml",
    }
    checked_hard_exit: list[Path] = []
    checked_no_hard_exit: list[Path] = []

    for path in _yaml_files(CONFIG_ROOT / "train"):
        config = load_config(path)
        stage = config.get("stage")
        routing = config.get("routing", {})
        assert isinstance(routing, dict), path
        if path in hard_exit_ablations:
            checked_no_hard_exit.append(path)
            assert routing.get("hard_exit", False) is not True, path
            continue
        if stage in hard_exit_stages:
            checked_hard_exit.append(path)
            assert routing.get("hard_exit") is True, path
        if stage in no_hard_exit_stages:
            checked_no_hard_exit.append(path)
            assert routing.get("hard_exit", False) is not True, path

    assert checked_hard_exit
    assert checked_no_hard_exit


def test_experiment_manifests_reference_train_configs() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT / "experiments"):
        config = _load_config(path, errors)
        if not config.get("experiment_name"):
            errors.append(f"{path}: missing experiment_name")
        ablations = config.get("ablations")
        if not isinstance(ablations, list) or not ablations:
            errors.append(f"{path}: ablations must be a non-empty list")
            continue
        baseline_path = _resolve_reference(
            config,
            "baseline_train_config",
            path,
            errors,
            required=False,
            repo_relative=True,
        )
        if baseline_path is not None:
            _load_config(baseline_path, errors)
        for index, entry in enumerate(ablations):
            if not isinstance(entry, dict):
                errors.append(f"{path}: ablations[{index}] must be a mapping")
                continue
            if not entry.get("id"):
                errors.append(f"{path}: ablations[{index}] missing id")
            train_config_path = _resolve_reference(entry, "train_config", path, errors, repo_relative=True)
            if train_config_path is not None:
                _load_config(train_config_path, errors)

    assert errors == []


def test_free_sphere_2x2_configs_match_ablation_matrix() -> None:
    expected = {
        "free_sphere_r125_2b_a0_coverage_no23.yaml": {
            "stage": "stage4_coverage_free_sphere",
            "mode": "scheduled",
            "pseudo_policy": "balanced_coverage",
            "selected_balance": 0.0,
            "transition_diversity": 0.0,
        },
        "free_sphere_r125_2b_a1_coverage_23loss.yaml": {
            "stage": "stage4_coverage_free_sphere",
            "mode": "scheduled",
            "pseudo_policy": "balanced_coverage",
            "selected_balance": 0.02,
            "transition_diversity": 0.005,
        },
        "free_sphere_r125_2b_b0_pure_no23.yaml": {
            "stage": "stage4_pure_free_sphere",
            "mode": "free",
            "pseudo_policy": None,
            "selected_balance": 0.0,
            "transition_diversity": 0.0,
        },
        "free_sphere_r125_2b_b1_pure_23loss.yaml": {
            "stage": "stage4_pure_free_sphere",
            "mode": "free",
            "pseudo_policy": None,
            "selected_balance": 0.02,
            "transition_diversity": 0.005,
        },
    }

    for filename, values in expected.items():
        train_config = load_config(CONFIG_ROOT / "train" / filename)
        model_config = load_config((CONFIG_ROOT / "train" / filename).parent / train_config["model_config"])
        routing = train_config["routing"]
        loss_weights = train_config["loss_weights"]
        route_path_visualization = train_config["route_path_visualization"]
        assert train_config["stage"] == values["stage"]
        assert routing["mode"] == values["mode"]
        assert routing.get("pseudo_policy") == values["pseudo_policy"]
        assert routing["hard_exit"] is True
        assert routing["log_path_counts"] is True
        assert routing["constraints"]["min_exit_step"] == 8
        assert routing["constraints"]["exit_ramp_start"] == 12
        assert routing["constraints"]["force_final_exit"] is True
        assert route_path_visualization["enabled"] is True
        assert route_path_visualization["upload_to_wandb"] is True
        assert route_path_visualization["interval"] == 2500
        assert model_config["route_pool_blocks"] == 8
        assert model_config["max_route_steps"] == 16
        assert loss_weights["cost"] == 0.0002
        assert loss_weights["selected_balance"] == values["selected_balance"]
        assert loss_weights["transition_diversity"] == values["transition_diversity"]


def test_experiment_manifests_keep_shared_data_config_for_validation() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT / "experiments"):
        config = _load_config(path, errors)
        train_refs: list[tuple[str, Path]] = []
        baseline_path = _resolve_reference(
            config,
            "baseline_train_config",
            path,
            errors,
            required=False,
            repo_relative=True,
        )
        if baseline_path is not None:
            train_refs.append(("baseline", baseline_path))
        for index, entry in enumerate(config.get("ablations", [])):
            if not isinstance(entry, dict):
                continue
            train_path = _resolve_reference(entry, "train_config", path, errors, repo_relative=True)
            if train_path is not None:
                train_refs.append((str(entry.get("id", index)), train_path))

        data_refs: dict[Path, list[str]] = {}
        for label, train_path in train_refs:
            train_config = _load_config(train_path, errors)
            data_path = _resolve_reference(train_config, "data_config", train_path, errors)
            if data_path is not None:
                data_refs.setdefault(data_path.resolve(), []).append(label)
        if len(data_refs) != 1:
            formatted = {str(data_path): labels for data_path, labels in data_refs.items()}
            errors.append(f"{path}: experiment entries must share one data_config for validation: {formatted}")

    assert errors == []


def test_eval_configs_declare_eval_names() -> None:
    errors: list[str] = []
    for path in _yaml_files(CONFIG_ROOT / "eval"):
        config = _load_config(path, errors)
        if not isinstance(config.get("eval_name"), str) or not config["eval_name"]:
            errors.append(f"{path}: missing eval_name")

    assert errors == []


def test_planned_data_recipe_ladder_is_declared() -> None:
    expected = {
        "r125_smoke": (100_000_000, 2_048),
        "r125_main_2b": (2_000_000_000, 2_048),
        "r125_main_5b": (5_000_000_000, 2_048),
        "r350_main_10b": (10_000_000_000, 4_096),
        "r350_main_30b": (30_000_000_000, 4_096),
        "r1b_pilot_10b": (10_000_000_000, 4_096),
        "r1b_main_50b": (50_000_000_000, 4_096),
    }
    by_name = {load_config(path)["recipe_name"]: load_config(path) for path in _yaml_files(CONFIG_ROOT / "data")}

    for recipe_name, (target_tokens, sequence_length) in expected.items():
        config = by_name[recipe_name]
        assert config["target_tokens"] == target_tokens
        assert config["sequence_length"] == sequence_length
        assert config["validation_tokens"] > 0
        assert config["output_dir"].endswith(recipe_name)
        assert config["manifest_path"].endswith(f"{recipe_name}.jsonl")


def test_scaled_data_recipes_taper_synthetic_routing_share() -> None:
    by_name = {load_config(path)["recipe_name"]: load_config(path) for path in _yaml_files(CONFIG_ROOT / "data")}

    r125_weights = [
        _mixture_weight(by_name["r125_smoke"], "synthetic_routing"),
        _mixture_weight(by_name["r125_main_2b"], "synthetic_routing"),
        _mixture_weight(by_name["r125_main_5b"], "synthetic_routing"),
    ]
    r350_weights = [
        _mixture_weight(by_name["r350_main_10b"], "synthetic_routing"),
        _mixture_weight(by_name["r350_main_30b"], "synthetic_routing"),
    ]
    r1b_weights = [
        _mixture_weight(by_name["r1b_pilot_10b"], "synthetic_routing"),
        _mixture_weight(by_name["r1b_main_50b"], "synthetic_routing"),
    ]

    assert all(math.isclose(weight, 0.10) for weight in r125_weights)
    assert all(0.05 <= weight <= 0.10 for weight in r350_weights)
    assert all(0.02 <= weight <= 0.05 for weight in r1b_weights)
    assert max(r125_weights) > max(r350_weights) > max(r1b_weights)


def test_non_synthetic_data_recipe_mixtures_are_normalized() -> None:
    for path in _yaml_files(CONFIG_ROOT / "data"):
        config = load_config(path)
        if config.get("synthetic_only", {}).get("enabled") is True:
            continue
        weights = [
            float(item["weight"])
            for item in config.get("mixture", {}).values()
            if isinstance(item, dict) and "weight" in item
        ]

        assert weights, f"{path}: non-synthetic recipes must declare mixture weights"
        assert math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9), path


def test_data_config_validation_rejects_invalid_numeric_and_mixture_values() -> None:
    config = load_config(CONFIG_ROOT / "data" / "r125_smoke.yaml")
    errors: list[str] = []
    _validate_data_config({**config, "target_tokens": True}, Path("bad_bool.yaml"), errors)
    _validate_data_config({**config, "sequence_length": 1}, Path("bad_sequence.yaml"), errors)
    _validate_data_config({**config, "validation_tokens": 0}, Path("bad_validation.yaml"), errors)
    _validate_data_config({**config, "mixture": {"fineweb_edu": {"weight": 0.5}}}, Path("bad_mixture.yaml"), errors)

    assert any("target_tokens must be an integer" in error for error in errors)
    assert any("sequence_length must be >= 2" in error for error in errors)
    assert any("validation_tokens must be >= 1" in error for error in errors)
    assert any("mixture weights must sum to 1.0" in error for error in errors)


def test_train_config_validation_rejects_invalid_precision_value() -> None:
    config = load_config(CONFIG_ROOT / "train" / "stage0_baseline.yaml")
    errors: list[str] = []

    _validate_train_config({**config, "precision": "float16"}, Path("bad_precision.yaml"), errors)

    assert any("precision must be fp32 or bf16" in error for error in errors)


def test_train_config_validation_rejects_invalid_loss_weights() -> None:
    config = load_config(CONFIG_ROOT / "train" / "stage4_output_action.yaml")
    errors: list[str] = []

    _validate_train_config({**config, "loss_weights": [("route", 1.0)]}, Path("bad_loss_mapping.yaml"), errors)
    _validate_train_config(
        {**config, "loss_weights": {"route": True, "balance": 0.01, "cost": 0.01, "location": 0.02}},
        Path("bad_loss_bool.yaml"),
        errors,
    )
    _validate_train_config(
        {**config, "loss_weights": {"route": 0.05, "balance": 0.01, "cost": -0.01, "location": 0.02}},
        Path("bad_loss_negative.yaml"),
        errors,
    )
    _validate_train_config(
        {**config, "loss_weights": {"route": 0.05, "extra": 0.0}},
        Path("bad_loss_extra.yaml"),
        errors,
    )

    assert any("loss_weights must be a mapping" in error for error in errors)
    assert any("loss_weights.route must be a finite numeric value" in error for error in errors)
    assert any("loss_weights.cost must be >= 0.0" in error for error in errors)
    assert any("unknown loss weight" in error for error in errors)


def test_train_config_validation_rejects_invalid_scheduled_routing_values() -> None:
    config = load_config(CONFIG_ROOT / "train" / "stage3_scheduled_free_routing.yaml")
    errors: list[str] = []

    _validate_train_config(
        {
            **config,
            "routing": {
                "mode": "scheduled",
                "schedule": [
                    {"max_step": 10, "router_probability": 0.5, "lambda_route": 0.2},
                    {"max_step": 10, "router_probability": 0.4, "lambda_route": 0.3},
                ],
            },
        },
        Path("bad_schedule.yaml"),
        errors,
    )
    _validate_train_config(
        {
            **config,
            "routing": {
                "mode": "scheduled",
                "schedule": [
                    {"max_step": True, "router_probability": 0.1, "lambda_route": 1.0},
                    {"max_step": 2, "router_probability": 0.9, "lambda_route": 0.05},
                ],
            },
        },
        Path("bad_bool_schedule.yaml"),
        errors,
    )

    assert any("routing.schedule max_step values must be strictly increasing" in error for error in errors)
    assert any("routing.schedule router_probability values must be nondecreasing" in error for error in errors)
    assert any("routing.schedule lambda_route values must be nonincreasing" in error for error in errors)
    assert any("routing.schedule.0.max_step must be an integer, not a boolean" in error for error in errors)


def test_train_config_validation_rejects_invalid_stage_routing_contracts() -> None:
    config = load_config(CONFIG_ROOT / "train" / "stage3_scheduled_free_routing.yaml")
    errors: list[str] = []

    _validate_train_config(
        {
            **config,
            "routing": {
                "mode": "pseudo",
                "pseudo_policy": "mixed_skip_recur",
                "schedule": [{"max_step": 10, "router_probability": 0.5, "lambda_route": 0.2}],
            },
        },
        Path("bad_stage_mode.yaml"),
        errors,
    )
    _validate_train_config(
        {
            **config,
            "routing": {
                "mode": "scheduled",
                "pseudo_policy": "unknown_policy",
                "schedule": [{"max_step": 10, "router_probability": 1.0, "lambda_route": 0.05}],
            },
        },
        Path("bad_policy.yaml"),
        errors,
    )
    _validate_train_config(
        {
            **config,
            "routing": {
                "mode": "scheduled",
                "pseudo_policy": "mixed_skip_recur",
                "hard_exit": "yes",
                "schedule": [{"max_step": 10, "router_probability": 1.0, "lambda_route": 0.05}],
            },
        },
        Path("bad_hard_exit.yaml"),
        errors,
    )
    _validate_train_config(
        {
            **config,
            "stage": "stage6_parallel_passing",
            "routing": {
                "mode": "parallel",
                "pseudo_policy": "mixed_skip_recur",
                "schedule": [{"max_step": 10, "router_probability": 1.0, "lambda_route": 0.05}],
                "hard_exit": True,
            },
        },
        Path("bad_parallel_inherited_schedule.yaml"),
        errors,
    )

    assert any("routing.mode must match executable stage mode scheduled" in error for error in errors)
    assert any("routing.schedule is only valid for scheduled routing" in error for error in errors)
    assert any("routing.pseudo_policy must be one of" in error for error in errors)
    assert any("routing.hard_exit must be a boolean" in error for error in errors)
    assert any("routing.pseudo_policy must be omitted for parallel routing" in error for error in errors)


def _yaml_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml"))


def _load_config(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return load_config(path)
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return {}


def _mixture_weight(config: dict[str, Any], tag: str) -> float:
    mixture = config.get("mixture", {})
    if not isinstance(mixture, dict):
        return 0.0
    item = mixture.get(tag, {})
    if not isinstance(item, dict):
        return 0.0
    value = item.get("weight", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _resolve_reference(
    config: dict[str, Any],
    key: str,
    source_path: Path,
    errors: list[str],
    *,
    required: bool = True,
    repo_relative: bool = False,
) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        if required:
            errors.append(f"{source_path}: missing {key}")
        return None
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [(source_path.parent / path).resolve()]
    if repo_relative and not path.is_absolute():
        candidates.extend([(CONFIG_ROOT.parent / path).resolve(), path.resolve()])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    errors.append(f"{source_path}: {key} does not exist: {candidates[-1]}")
    return None


def _validate_model_config(config: dict[str, Any], path: Path, errors: list[str]) -> None:
    architecture = config.get("architecture")
    if architecture == "decoder_only_llama_like":
        _require_keys(
            config,
            path,
            errors,
            ["model_name", "layers", "d_model", "n_heads", "context_length", "vocab_size"],
        )
        return
    if architecture == "brian_route_core":
        _require_keys(
            config,
            path,
            errors,
            [
                "model_name",
                "pre_blocks",
                "route_pool_blocks",
                "post_blocks",
                "block_position_dim",
                "max_route_steps",
            ],
        )
        if "base_config" not in config and not isinstance(config.get("base"), dict):
            errors.append(f"{path}: brian_route_core requires base_config or base")
        try:
            BrianRouteConfig.from_dict(config, config_dir=path.parent)
        except Exception as exc:
            errors.append(f"{path}: BrianRouteConfig parse failed: {type(exc).__name__}: {exc}")
        _validate_parallel_model_config(config, path, errors)
        return
    errors.append(f"{path}: unknown model architecture {architecture!r}")


def _validate_parallel_model_config(config: dict[str, Any], path: Path, errors: list[str]) -> None:
    if config.get("parallel_passing") is not True:
        return
    if config.get("global_kv") is not True:
        errors.append(f"{path}: parallel_passing configs must enable global_kv for shared base memory")
    if _int_config_value(config.get("beam_size", 2), "beam_size", path, errors, minimum=1) is not None:
        beam_size = int(config.get("beam_size", 2))
        if beam_size > 4:
            errors.append(f"{path}: beam_size must be <= 4 for planned parallel-passing configs")
    if _int_config_value(config.get("global_window_slots", 0), "global_window_slots", path, errors, minimum=1) is None:
        errors.append(f"{path}: parallel_passing configs must keep a positive global_window_slots delta window")
    branch_score_decay = _float_config_value(
        config.get("branch_score_decay", 1.0),
        "branch_score_decay",
        path,
        errors,
        minimum=0.0,
    )
    if branch_score_decay is not None and not 0.0 < branch_score_decay < 1.0:
        errors.append(f"{path}: branch_score_decay must be > 0 and < 1 for planned parallel-passing configs")
    if config.get("parallel_exit_policy", "branch") not in {"branch", "top1", "any_topk"}:
        errors.append(f"{path}: parallel_exit_policy must be branch, top1, or any_topk")


def _validate_data_config(config: dict[str, Any], path: Path, errors: list[str]) -> None:
    _require_keys(
        config,
        path,
        errors,
        ["recipe_name", "target_tokens", "sequence_length", "validation_tokens", "output_dir", "manifest_path"],
    )
    if not isinstance(config.get("recipe_name"), str) or not config.get("recipe_name"):
        errors.append(f"{path}: recipe_name must be a non-empty string")
    for key, minimum in [
        ("target_tokens", 1),
        ("sequence_length", 2),
        ("validation_tokens", 1),
    ]:
        _int_config_value(config.get(key), key, path, errors, minimum=minimum)
    for key in ["output_dir", "manifest_path"]:
        if not isinstance(config.get(key), str) or not config.get(key):
            errors.append(f"{path}: {key} must be a non-empty string")
    synthetic_only = config.get("synthetic_only", {})
    synthetic_enabled = False
    if synthetic_only:
        if not isinstance(synthetic_only, dict):
            errors.append(f"{path}: synthetic_only must be a mapping")
        else:
            enabled = synthetic_only.get("enabled", False)
            if not isinstance(enabled, bool):
                errors.append(f"{path}: synthetic_only.enabled must be a boolean")
            synthetic_enabled = enabled is True
    mixture = config.get("mixture", {})
    if synthetic_enabled:
        return
    if not isinstance(mixture, dict) or not mixture:
        errors.append(f"{path}: mixture must be a non-empty mapping when synthetic_only is disabled")
        return
    weights: list[float] = []
    for tag, item in mixture.items():
        if not isinstance(item, dict):
            errors.append(f"{path}: mixture.{tag} must be a mapping")
            continue
        weight = _float_config_value(item.get("weight"), f"mixture.{tag}.weight", path, errors, minimum=0.0)
        if weight is not None:
            weights.append(weight)
        if not isinstance(item.get("source_dataset"), str) or not item.get("source_dataset"):
            errors.append(f"{path}: mixture.{tag}.source_dataset must be a non-empty string")
    if weights and not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{path}: mixture weights must sum to 1.0")


def _validate_train_config(config: dict[str, Any], path: Path, errors: list[str]) -> None:
    stage_mode: str | None = None
    if isinstance(config.get("stage"), str):
        try:
            stage_mode = train_mode_for_stage(str(config["stage"]))
        except ValueError:
            stage_mode = None
    for key, minimum in [
        ("batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("warmup_steps", 0),
        ("max_steps", 1),
        ("eval_interval", 1),
        ("save_interval", 1),
    ]:
        if key in config:
            _int_config_value(config.get(key), key, path, errors, minimum=minimum)
    for key in ["learning_rate", "min_learning_rate", "weight_decay", "grad_clip"]:
        if key in config:
            _float_config_value(config.get(key), key, path, errors, minimum=0.0)
    if "learning_rate" in config and "min_learning_rate" in config:
        learning_rate = _float_config_value(config.get("learning_rate"), "learning_rate", path, errors, minimum=0.0)
        min_learning_rate = _float_config_value(
            config.get("min_learning_rate"),
            "min_learning_rate",
            path,
            errors,
            minimum=0.0,
        )
        if learning_rate is not None and min_learning_rate is not None and min_learning_rate > learning_rate:
            errors.append(f"{path}: min_learning_rate must be <= learning_rate")
    if "lr_schedule" in config and config.get("lr_schedule") not in {"constant", "linear_warmup_cosine_decay"}:
        errors.append(f"{path}: lr_schedule must be constant or linear_warmup_cosine_decay")
    if "precision" in config and config.get("precision") not in {"fp32", "bf16"}:
        errors.append(f"{path}: precision must be fp32 or bf16")
    for key in ["activation_checkpointing", "ddp_find_unused_parameters", "resume", "write_routing_report_on_checkpoint"]:
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{path}: {key} must be a boolean")
    if "loss_weights" in config:
        _validate_loss_weights_config(config.get("loss_weights"), path, errors)
    if "route_path_visualization" in config:
        _validate_route_path_visualization_config(config.get("route_path_visualization"), path, errors)
    routing = config.get("routing", {})
    if routing:
        if not isinstance(routing, dict):
            errors.append(f"{path}: routing must be a mapping")
        else:
            _validate_routing_config(routing, path, errors, stage_mode=stage_mode)
    elif stage_mode is not None and stage_mode != "baseline":
        errors.append(f"{path}: routing must be declared for {config.get('stage')}")


def _validate_loss_weights_config(loss_weights: Any, path: Path, errors: list[str]) -> None:
    if not isinstance(loss_weights, dict):
        errors.append(f"{path}: loss_weights must be a mapping")
        return
    allowed = {
        "route",
        "balance",
        "cost",
        "location",
        "selected_balance",
        "coverage_floor",
        "transition_diversity",
        "exit_boundary",
        "input_anchor",
    }
    for key, value in loss_weights.items():
        if key not in allowed:
            errors.append(f"{path}: unknown loss weight {key!r}")
            continue
        _float_config_value(value, f"loss_weights.{key}", path, errors, minimum=0.0)


def _validate_route_path_visualization_config(value: Any, path: Path, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: route_path_visualization must be a mapping")
        return
    allowed = {
        "enabled",
        "interval",
        "upload_to_wandb",
        "output_dir",
        "wandb_key",
        "top_paths",
        "timeline_max_frames",
    }
    for key, item in value.items():
        if key not in allowed:
            errors.append(f"{path}: unknown route_path_visualization key {key!r}")
            continue
        if key in {"enabled", "upload_to_wandb"}:
            if not isinstance(item, bool):
                errors.append(f"{path}: route_path_visualization.{key} must be a boolean")
        elif key in {"interval", "top_paths", "timeline_max_frames"}:
            _int_config_value(item, f"route_path_visualization.{key}", path, errors, minimum=1)
        elif not isinstance(item, str) or not item:
            errors.append(f"{path}: route_path_visualization.{key} must be a non-empty string")


def _validate_routing_config(
    routing: dict[str, Any],
    path: Path,
    errors: list[str],
    *,
    stage_mode: str | None,
) -> None:
    mode = routing.get("mode")
    if mode is not None and mode not in {"fixed", "pseudo", "scheduled", "free", "parallel"}:
        errors.append(f"{path}: routing.mode must be fixed, pseudo, scheduled, free, or parallel")
    if stage_mode == "baseline":
        errors.append(f"{path}: baseline train configs must not declare routing")
    elif stage_mode is not None and mode != stage_mode:
        errors.append(f"{path}: routing.mode must match executable stage mode {stage_mode}")
    if "hard_exit" in routing and not isinstance(routing.get("hard_exit"), bool):
        errors.append(f"{path}: routing.hard_exit must be a boolean")
    if "log_path_counts" in routing and not isinstance(routing.get("log_path_counts"), bool):
        errors.append(f"{path}: routing.log_path_counts must be a boolean")
    if "constraints" in routing and routing.get("constraints") is not None:
        _validate_routing_constraints_config(routing.get("constraints"), path, errors)
    _validate_pseudo_policy_config(routing, path, errors)
    if mode == "scheduled":
        _validate_scheduled_routing_config(routing, path, errors)
    elif "schedule" in routing and routing.get("schedule") is not None:
        errors.append(f"{path}: routing.schedule is only valid for scheduled routing")


def _validate_pseudo_policy_config(routing: dict[str, Any], path: Path, errors: list[str]) -> None:
    mode = routing.get("mode")
    policy = routing.get("pseudo_policy")
    allowed = {"sequential", "mixed_skip_recur", "balanced_coverage", "random_internal"}
    if mode == "parallel":
        if policy is not None:
            errors.append(f"{path}: routing.pseudo_policy must be omitted for parallel routing")
        return
    if mode == "free":
        if policy is not None:
            errors.append(f"{path}: routing.pseudo_policy must be omitted for free routing")
        return
    if mode in {"fixed", "pseudo", "scheduled"} and policy is None:
        errors.append(f"{path}: routing.pseudo_policy must be declared for {mode} routing")
        return
    if policy is not None and policy not in allowed:
        errors.append(f"{path}: routing.pseudo_policy must be one of {sorted(allowed)}")
    if mode == "fixed" and policy != "sequential":
        errors.append(f"{path}: fixed routing must use sequential pseudo_policy")


def _validate_routing_constraints_config(constraints: Any, path: Path, errors: list[str]) -> None:
    if not isinstance(constraints, dict):
        errors.append(f"{path}: routing.constraints must be a mapping")
        return
    int_keys = {
        "min_exit_step",
        "exit_ramp_start",
        "self_recur_max_consecutive",
        "self_recur_cap",
        "max_consecutive_self_recur",
    }
    float_keys = {
        "early_exit_logit_penalty",
        "exit_ramp_logit_bias",
        "final_exit_logit_bias",
        "coverage_floor_min",
    }
    bool_keys = {"force_final_exit"}
    allowed = int_keys | float_keys | bool_keys
    for key, value in constraints.items():
        if key not in allowed:
            errors.append(f"{path}: unknown routing constraint {key!r}")
            continue
        if key in int_keys:
            _int_config_value(value, f"routing.constraints.{key}", path, errors, minimum=1)
        elif key in float_keys:
            _float_config_value(value, f"routing.constraints.{key}", path, errors, minimum=0.0)
        elif not isinstance(value, bool):
            errors.append(f"{path}: routing.constraints.{key} must be a boolean")


def _validate_scheduled_routing_config(routing: dict[str, Any], path: Path, errors: list[str]) -> None:
    schedule = routing.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        errors.append(f"{path}: routing.schedule must be a non-empty list for scheduled routing")
        return

    max_steps: list[int] = []
    router_probabilities: list[float] = []
    lambda_routes: list[float] = []
    for index, item in enumerate(schedule):
        if not isinstance(item, dict):
            errors.append(f"{path}: routing.schedule.{index} must be a mapping")
            continue
        max_step = _int_config_value(
            item.get("max_step"),
            f"routing.schedule.{index}.max_step",
            path,
            errors,
            minimum=1,
        )
        router_probability = _float_config_value(
            item.get("router_probability"),
            f"routing.schedule.{index}.router_probability",
            path,
            errors,
            minimum=0.0,
        )
        lambda_route = _float_config_value(
            item.get("lambda_route"),
            f"routing.schedule.{index}.lambda_route",
            path,
            errors,
            minimum=0.0,
        )
        if max_step is not None:
            max_steps.append(max_step)
        if router_probability is not None:
            router_probabilities.append(router_probability)
            if router_probability > 1.0:
                errors.append(f"{path}: routing.schedule.{index}.router_probability must be <= 1.0")
        if lambda_route is not None:
            lambda_routes.append(lambda_route)

    if len(max_steps) != len(schedule) or len(router_probabilities) != len(schedule) or len(lambda_routes) != len(schedule):
        return
    if any(next_value <= value for value, next_value in zip(max_steps, max_steps[1:])):
        errors.append(f"{path}: routing.schedule max_step values must be strictly increasing")
    if any(next_value < value for value, next_value in zip(router_probabilities, router_probabilities[1:])):
        errors.append(f"{path}: routing.schedule router_probability values must be nondecreasing")
    if any(next_value > value for value, next_value in zip(lambda_routes, lambda_routes[1:])):
        errors.append(f"{path}: routing.schedule lambda_route values must be nonincreasing")
    router_probability_increases = len(router_probabilities) >= 2 and router_probabilities[-1] > router_probabilities[0]
    if not router_probability_increases and not all(math.isclose(value, 1.0) for value in router_probabilities):
        errors.append(f"{path}: routing.schedule router_probability must increase")
    if not math.isclose(router_probabilities[-1], 1.0, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{path}: routing.schedule final router_probability must be 1.0")
    lambda_route_decays = len(lambda_routes) >= 2 and lambda_routes[-1] < lambda_routes[0]
    if not lambda_route_decays and not all(math.isclose(value, 0.0) for value in lambda_routes):
        errors.append(f"{path}: routing.schedule lambda_route must decay")
    if lambda_routes[-1] > 0.05:
        errors.append(f"{path}: routing.schedule final lambda_route must be <= 0.05")


def _int_config_value(value: Any, name: str, path: Path, errors: list[str], *, minimum: int) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{path}: {name} must be an integer, not a boolean")
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        errors.append(f"{path}: {name} must be an integer")
        return None
    if number < minimum:
        errors.append(f"{path}: {name} must be >= {minimum}")
        return None
    return number


def _float_config_value(
    value: Any,
    name: str,
    path: Path,
    errors: list[str],
    *,
    minimum: float,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{path}: {name} must be a finite numeric value")
        return None
    number = float(value)
    if number < minimum:
        errors.append(f"{path}: {name} must be >= {minimum}")
        return None
    return number


def _require_keys(config: dict[str, Any], path: Path, errors: list[str], keys: list[str]) -> None:
    for key in keys:
        if key not in config:
            errors.append(f"{path}: missing {key}")
