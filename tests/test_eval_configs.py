import ast
import importlib.util
from pathlib import Path

import pytest

from brian_sphere_llm.utils.config import load_config


def test_all_eval_configs_have_cli_dispatch() -> None:
    supported = _eval_names_supported_by_cli(Path("scripts/eval.py"))
    configured = {
        load_config(path)["eval_name"]
        for path in Path("configs/eval").glob("*.yaml")
    }

    assert configured <= supported


def test_eval_cli_numeric_helpers_reject_boolean_config_values() -> None:
    module = _load_eval_cli_module()

    with pytest.raises(ValueError, match="max_batches"):
        module._int_config({"max_batches": True}, "max_batches", default=8)
    with pytest.raises(ValueError, match="tolerance"):
        module._float_config({"tolerance": False}, "tolerance", default=1e-8)
    with pytest.raises(ValueError, match="include_baseline"):
        module._bool_config({"include_baseline": 1}, "include_baseline", default=False)


def test_eval_cli_arg_overrides_preserve_zero_values() -> None:
    module = _load_eval_cli_module()

    assert module._int_arg_or_config(0, {"sample_count": 24}, "sample_count", default=24) == 0
    assert module._float_arg_or_config(0.0, {"min_step_delta": 1.0}, "min_step_delta", default=1.0) == 0.0


def _eval_names_supported_by_cli(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    supported = {"routing_eval"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not isinstance(left, ast.Name) or left.id != "eval_name":
            continue
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            if (
                isinstance(operator, ast.Eq)
                and isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
            ):
                supported.add(comparator.value)
    return supported


def _load_eval_cli_module():
    spec = importlib.util.spec_from_file_location("brian_eval_cli_for_tests", Path("scripts/eval.py"))
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
