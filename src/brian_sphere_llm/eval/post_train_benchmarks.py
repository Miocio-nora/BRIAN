from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


def run_post_train_benchmarks(
    run_dir: str | Path,
    train_config: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    cfg = _mapping(train_config.get("post_train_benchmarks"))
    if not _bool(cfg.get("enabled", False)):
        return []

    run_dir = Path(run_dir)
    project_root = Path(project_root or Path.cwd())
    commands = build_post_train_benchmark_commands(run_dir, train_config, project_root=project_root)
    fail_on_error = _bool(cfg.get("fail_on_error", True))
    summary_path = run_dir / str(cfg.get("summary_path", "post_train_benchmarks.json"))
    _release_torch_cache()

    results: list[dict[str, Any]] = []
    env = _subprocess_env(project_root)
    for spec in commands:
        started = time.time()
        completed = subprocess.run(
            spec["command"],
            cwd=project_root,
            env=env,
            check=False,
        )
        results.append(
            {
                "name": spec["name"],
                "command": spec["command"],
                "output_path": spec["output_path"],
                "samples_output_path": spec["samples_output_path"],
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - started,
            }
        )
        write_json({"run_dir": str(run_dir), "results": results}, summary_path)
        if completed.returncode != 0 and fail_on_error:
            raise RuntimeError(f"Post-train benchmark {spec['name']} failed with exit code {completed.returncode}.")
    return results


def build_post_train_benchmark_commands(
    run_dir: str | Path,
    train_config: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    cfg = _mapping(train_config.get("post_train_benchmarks"))
    if not _bool(cfg.get("enabled", False)):
        return []

    run_dir = Path(run_dir)
    project_root = Path(project_root or Path.cwd())
    output_dir = _path(cfg.get("output_dir", run_dir / "benchmarks"), project_root=project_root)
    label = _safe_label(str(cfg.get("label") or run_dir.name))
    checkpoint = str(cfg.get("checkpoint", "checkpoint_latest"))
    commands: list[dict[str, Any]] = []

    reasoning_cfg = _mapping(cfg.get("reasoning"))
    if _bool(reasoning_cfg.get("enabled", True)):
        output_path = _path(
            reasoning_cfg.get("output_path", output_dir / f"{label}.reasoning_s600.json"),
            project_root=project_root,
        )
        samples_output_path = _path(
            reasoning_cfg.get("samples_output_path", output_dir / f"{label}.reasoning_s600_samples.jsonl"),
            project_root=project_root,
        )
        commands.append(
            {
                "name": "reasoning_s600",
                "output_path": str(output_path),
                "samples_output_path": str(samples_output_path),
                "command": [
                    sys.executable,
                    str(project_root / "scripts" / "eval.py"),
                    "--config",
                    str(_path(reasoning_cfg.get("config", "configs/eval/reasoning_eval_s600.yaml"), project_root=project_root)),
                    "--run",
                    str(run_dir),
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(output_path),
                    "--samples-path",
                    str(samples_output_path),
                ],
            }
        )

    public_cfg = _mapping(cfg.get("public"))
    if _bool(public_cfg.get("enabled", True)):
        output_path = _path(
            public_cfg.get("output_path", output_dir / f"public_{label}_s200.json"),
            project_root=project_root,
        )
        samples_output_path = _path(
            public_cfg.get("samples_output_path", output_dir / f"public_{label}_s200_samples.jsonl"),
            project_root=project_root,
        )
        commands.append(
            {
                "name": "public_s600",
                "output_path": str(output_path),
                "samples_output_path": str(samples_output_path),
                "command": [
                    sys.executable,
                    str(project_root / "scripts" / "public_benchmark.py"),
                    "--config",
                    str(_path(public_cfg.get("config", "configs/eval/public_benchmark_s600.yaml"), project_root=project_root)),
                    "--run",
                    str(run_dir),
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(output_path),
                    "--samples-output",
                    str(samples_output_path),
                ],
            }
        )

    return commands


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("post_train_benchmarks entries must be mappings.")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _path(value: Any, *, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "run"


def _subprocess_env(project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str(project_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
