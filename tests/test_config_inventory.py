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


def _yaml_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml"))


def _load_config(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return load_config(path)
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return {}


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


def _require_keys(config: dict[str, Any], path: Path, errors: list[str], keys: list[str]) -> None:
    for key in keys:
        if key not in config:
            errors.append(f"{path}: missing {key}")
