import ast
from pathlib import Path

from brian_sphere_llm.utils.config import load_config


def test_all_eval_configs_have_cli_dispatch() -> None:
    supported = _eval_names_supported_by_cli(Path("scripts/eval.py"))
    configured = {
        load_config(path)["eval_name"]
        for path in Path("configs/eval").glob("*.yaml")
    }

    assert configured <= supported


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
