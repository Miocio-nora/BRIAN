from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.utils.config import load_yaml, save_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_data_cli_runs_tiny_synthetic_config(tmp_path: Path) -> None:
    config = load_yaml(ROOT / "configs/data/r125_tiny_debug.yaml")
    config["output_dir"] = str(tmp_path / "tokenized")
    config["manifest_path"] = str(tmp_path / "manifest.jsonl")
    config["target_tokens"] = 256
    config["validation_tokens"] = 64
    config["synthetic_only"]["sample_count"] = 12
    config_path = tmp_path / "data.yaml"
    save_yaml(config, config_path)

    result = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--config", str(config_path)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    output_dir = Path(result.stdout.strip().splitlines()[-1])
    assert output_dir == Path(config["output_dir"])
    assert (output_dir / "train.bin").exists()
    assert (output_dir / "val.bin").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "stats.json").exists()
    assert Path(config["manifest_path"]).exists()


def test_train_cli_runs_one_step_tiny_baseline(tmp_path: Path) -> None:
    tokenized_dir = tmp_path / "tokenized"
    tokenized_dir.mkdir()
    write_token_bin([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], tokenized_dir / "train.bin")
    write_token_bin([[1, 2, 3, 4], [2, 3, 4, 5]], tokenized_dir / "val.bin")
    write_index(tokenized_dir / "train.idx", sequence_length=4, num_sequences=4)
    write_index(tokenized_dir / "val.idx", sequence_length=4, num_sequences=2)
    (tokenized_dir / "stats.json").write_text(
        json.dumps(
            {
                "recipe_name": "cli_unit_data",
                "sequence_length": 4,
                "num_tokens_train": 16,
                "num_tokens_val": 8,
            }
        ),
        encoding="utf-8",
    )

    model_config = tmp_path / "model.yaml"
    save_yaml(
        {
            "model_name": "cli_baseline_unit",
            "architecture": "decoder_only_llama_like",
            "layers": 1,
            "d_model": 16,
            "n_heads": 4,
            "context_length": 4,
            "vocab_size": 32,
            "dropout": 0.0,
        },
        model_config,
    )
    data_config = tmp_path / "data.yaml"
    save_yaml(
        {
            "recipe_name": "cli_unit_data",
            "output_dir": str(tokenized_dir),
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "sequence_length": 4,
        },
        data_config,
    )
    train_config = tmp_path / "train.yaml"
    save_yaml(
        {
            "stage": "stage0_baseline",
            "run_name": "cli_unit_run",
            "model_config": str(model_config),
            "data_config": str(data_config),
            "output_root": str(tmp_path / "runs"),
            "seed": 1,
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "eval_interval": 1,
            "save_interval": 1,
            "learning_rate": 0.001,
            "resume": False,
        },
        train_config,
    )

    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(train_config)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    run_dir = Path(result.stdout.strip().splitlines()[-1])
    assert run_dir == tmp_path / "runs" / "cli_unit_run"
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "train_log.jsonl").exists()
    assert (run_dir / "eval_log.jsonl").exists()
    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    assert (run_dir / "checkpoint_best" / "state.pt").exists()
    assert (run_dir / "routing_report.json").exists()


def test_lm_eval_cli_writes_validation_report_to_run_dir_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_log.jsonl").write_text(
        json.dumps({"tokens_per_second": 128.0, "active_block_evals_per_token": 0.5}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"validation_loss": 2.0, "perplexity": 7.4}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(
        json.dumps(
            {
                "summary": {"active_block_evals_per_token": 0.5},
                "checks": {},
                "overall_status": "warn",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/eval.py", "--config", "configs/eval/lm_eval.yaml", "--run", str(run_dir)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    report_path = run_dir / "lm_eval_report.json"
    assert Path(result.stdout.strip().splitlines()[-1]) == report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["metrics"]["validation_loss"] == 2.0
    assert report["metrics"]["active_block_evals_per_token"] == 0.5
