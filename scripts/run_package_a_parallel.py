#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.utils.config import load_config


PACKAGE_A_2B_WAVES = [
    [
        ("A0", "configs/train/package_a_r125_2b_a0_baseline.yaml"),
        ("A1", "configs/train/package_a_r125_2b_a1_fixed_route.yaml"),
        ("A2", "configs/train/package_a_r125_2b_a2_sequential_router_imitation.yaml"),
        ("A3", "configs/train/package_a_r125_2b_a3_skip_recur_router_imitation.yaml"),
    ],
    [
        ("A4", "configs/train/package_a_r125_2b_a4_free_router_block_position.yaml"),
        ("A5", "configs/train/package_a_r125_2b_a5_no_block_position.yaml"),
        ("A6", "configs/train/package_a_r125_2b_a6_no_output_action.yaml"),
        ("A7", "configs/train/package_a_r125_2b_a7_no_location_loss.yaml"),
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Package A R125 2B jobs as two 4-GPU waves.")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated physical GPU ids for each wave.")
    parser.add_argument("--log-dir", default="experiments/generated/route_core_r125_2b_package/launcher_logs")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--rerun-completed", action="store_false", dest="skip_completed")
    parser.add_argument("--skip-wandb-check", action="store_true")
    args = parser.parse_args()

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) != 4:
        raise SystemExit("--gpus must contain exactly four GPU ids.")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _check_wandb_ready(PACKAGE_A_2B_WAVES, skip=args.skip_wandb_check)

    for wave_index, wave in enumerate(PACKAGE_A_2B_WAVES, start=1):
        jobs = []
        for gpu, (entry_id, config_path) in zip(gpus, wave, strict=True):
            config_path_obj = Path(config_path)
            config = load_config(config_path_obj)
            run_dir = _run_dir(config, config_path_obj)
            if args.skip_completed and _is_completed(run_dir, int(config["max_steps"])):
                print(f"[wave {wave_index}] skip completed {entry_id}: {run_dir}", flush=True)
                continue
            jobs.append(_launch(entry_id, config_path_obj, gpu, log_dir))
        _wait_wave(wave_index, jobs)

    print("Package A R125 2B launcher finished.", flush=True)


def _check_wandb_ready(waves: list[list[tuple[str, str]]], *, skip: bool) -> None:
    if skip:
        return
    needs_online = False
    for wave in waves:
        for _entry_id, config_path in wave:
            wandb_cfg = load_config(config_path).get("wandb", {})
            if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled") is True and str(wandb_cfg.get("mode", "online")) == "online":
                needs_online = True
    if not needs_online:
        return
    if _wandb_api_auth_ok():
        return
    try:
        result = subprocess.run(
            ["wandb", "status"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("wandb CLI is not installed in this environment.") from exc
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or '"api_key": null' in output:
        raise SystemExit("W&B online logging is enabled but not authenticated. Run `wandb login --relogin` or set WANDB_API_KEY.")


def _wandb_api_auth_ok() -> bool:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import wandb; api=wandb.Api(); print(bool(api.viewer))",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "True" in result.stdout


def _run_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_root = Path(str(config.get("output_root", "runs")))
    run_name = str(config["run_name"])
    if run_name == "auto":
        raise ValueError(f"{config_path} must define a concrete run_name for package launching.")
    return output_root / run_name


def _is_completed(run_dir: Path, max_steps: int) -> bool:
    train_log = run_dir / "train_log.jsonl"
    latest = run_dir / "checkpoint_latest" / "state.pt"
    if not train_log.exists() or not latest.exists():
        return False
    try:
        last_line = train_log.read_text(encoding="utf-8").splitlines()[-1]
        return int(json.loads(last_line).get("step", 0)) >= max_steps
    except (IndexError, json.JSONDecodeError, ValueError, TypeError):
        return False


def _launch(entry_id: str, config_path: Path, gpu: str, log_dir: Path) -> tuple[str, subprocess.Popen]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = _pythonpath_with_src(env.get("PYTHONPATH", ""))
    log_path = log_dir / f"{entry_id}.log"
    handle = log_path.open("ab")
    cmd = [sys.executable, "scripts/train.py", "--config", str(config_path)]
    print(f"launch {entry_id} on GPU {gpu}: {' '.join(cmd)} > {log_path}", flush=True)
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    process._brian_log_handle = handle  # type: ignore[attr-defined]
    return entry_id, process


def _pythonpath_with_src(existing: str) -> str:
    src = str(Path("src").resolve())
    if not existing:
        return src
    parts = existing.split(os.pathsep)
    if src in parts:
        return existing
    return os.pathsep.join([src, *parts])


def _wait_wave(wave_index: int, jobs: list[tuple[str, subprocess.Popen]]) -> None:
    if not jobs:
        print(f"[wave {wave_index}] no jobs to run.", flush=True)
        return
    try:
        failures: list[tuple[str, int]] = []
        running = dict(jobs)
        while running:
            time.sleep(30)
            for entry_id, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                _close_process_log(process)
                running.pop(entry_id)
                print(f"[wave {wave_index}] {entry_id} exited with code {code}", flush=True)
                if code != 0:
                    failures.append((entry_id, code))
        if failures:
            failed = ", ".join(f"{entry_id}:{code}" for entry_id, code in failures)
            raise SystemExit(f"Wave {wave_index} failed: {failed}")
    except KeyboardInterrupt:
        print(f"interrupt: terminating wave {wave_index}", flush=True)
        for _entry_id, process in jobs:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _entry_id, process in jobs:
            if process.poll() is None:
                process.wait(timeout=30)
            _close_process_log(process)
        raise


def _close_process_log(process: subprocess.Popen) -> None:
    handle = getattr(process, "_brian_log_handle", None)
    if handle is not None:
        handle.close()


if __name__ == "__main__":
    main()
