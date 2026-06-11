from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTORCH_CU128_PINS = {
    "torch==2.11.0+cu128",
    "torchvision==0.26.0+cu128",
    "torchaudio==2.11.0+cu128",
}


def test_environment_pins_b200_cuda128_wheels() -> None:
    env = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    pip_deps = _pip_dependencies(env["dependencies"])

    assert "--index-url https://download.pytorch.org/whl/cu128" in pip_deps
    assert "--extra-index-url https://pypi.org/simple" in pip_deps
    assert PYTORCH_CU128_PINS.issubset(set(pip_deps))


def test_requirements_pin_b200_cuda128_wheels() -> None:
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert requirements[0] == "--index-url https://download.pytorch.org/whl/cu128"
    assert "--extra-index-url https://pypi.org/simple" in requirements
    assert PYTORCH_CU128_PINS.issubset(set(requirements))


def _pip_dependencies(dependencies: list[object]) -> list[str]:
    for dependency in dependencies:
        if isinstance(dependency, dict) and "pip" in dependency:
            pip_deps = dependency["pip"]
            return [str(item) for item in pip_deps]
    return []
