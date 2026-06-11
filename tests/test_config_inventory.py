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
        return
    errors.append(f"{path}: unknown model architecture {architecture!r}")


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
