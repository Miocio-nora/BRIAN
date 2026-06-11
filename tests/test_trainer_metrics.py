import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.trainer import evaluate, train_from_config
from brian_sphere_llm.utils.config import save_yaml


def test_evaluate_reports_inference_timing_metrics() -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=4, layers=1, d_model=16, n_heads=4))
    val_loader = [
        torch.randint(0, 64, (2, 4)),
        torch.randint(0, 64, (2, 4)),
    ]

    row = evaluate(
        model,
        val_loader,
        config={"stage": "stage0_baseline"},
        device=torch.device("cpu"),
        route_mode="baseline",
        global_step=1,
    )

    assert row["eval_batch_count"] == 2
    assert row["eval_token_count"] == 16
    assert row["inference_time_seconds"] > 0.0
    assert row["inference_tokens_per_second"] > 0.0
    assert row["inference_latency_ms_per_token"] > 0.0
    assert row["validation_loss"] >= 0.0


def test_train_from_config_writes_routing_report_on_checkpoint(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    sequences = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    write_token_bin(sequences, tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=4, num_sequences=len(sequences))
    write_token_bin(sequences, tokenized / "val.bin")
    write_index(tokenized / "val.idx", sequence_length=4, num_sequences=len(sequences))

    model_config = tmp_path / "model.yaml"
    save_yaml(
        {
            "model_name": "baseline_unit",
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
            "recipe_name": "unit_data",
            "output_dir": str(tokenized),
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "sequence_length": 4,
        },
        data_config,
    )
    train_config = tmp_path / "train.yaml"
    save_yaml(
        {
            "stage": "stage0_baseline",
            "run_name": "unit_run",
            "model_config": str(model_config),
            "data_config": str(data_config),
            "output_root": str(tmp_path / "runs"),
            "seed": 1,
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 2,
            "max_steps": 1,
            "eval_interval": 1,
            "save_interval": 1,
            "learning_rate": 0.001,
            "resume": False,
            "write_routing_report_on_checkpoint": True,
        },
        train_config,
    )

    run_dir = train_from_config(train_config)

    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    assert (run_dir / "checkpoint_best" / "state.pt").exists()
    assert (run_dir / "routing_report.json").exists()
    report = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    assert report["latest_eval"]["validation_loss"] >= 0.0
    assert report["cost_quality_curve"]["summary"]["train_point_count"] == 1


def test_train_from_config_records_resume_event(tmp_path: Path) -> None:
    train_config = _write_tiny_train_fixture(tmp_path, max_steps=1, resume=False)
    run_dir = train_from_config(train_config)
    train_config = _write_tiny_train_fixture(tmp_path, max_steps=2, resume=True)

    resumed_run_dir = train_from_config(train_config)

    assert resumed_run_dir == run_dir
    events = [json.loads(line) for line in (run_dir / "resume_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["resumed_from_step"] == 1
    assert events[-1]["target_max_steps"] == 2
    assert events[-1]["optimizer_state_loaded"] is True
    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    train_rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train_rows[-1]["step"] == 2


def _write_tiny_train_fixture(tmp_path: Path, *, max_steps: int, resume: bool) -> Path:
    tokenized = tmp_path / "tokenized_resume"
    sequences = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    write_token_bin(sequences, tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=4, num_sequences=len(sequences))
    write_token_bin(sequences, tokenized / "val.bin")
    write_index(tokenized / "val.idx", sequence_length=4, num_sequences=len(sequences))

    model_config = tmp_path / "model_resume.yaml"
    save_yaml(
        {
            "model_name": "baseline_unit_resume",
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
    data_config = tmp_path / "data_resume.yaml"
    save_yaml(
        {
            "recipe_name": "unit_data_resume",
            "output_dir": str(tokenized),
            "manifest_path": str(tmp_path / "manifest_resume.jsonl"),
            "sequence_length": 4,
        },
        data_config,
    )
    train_config = tmp_path / "train_resume.yaml"
    save_yaml(
        {
            "stage": "stage0_baseline",
            "run_name": "unit_resume_run",
            "model_config": str(model_config),
            "data_config": str(data_config),
            "output_root": str(tmp_path / "runs"),
            "seed": 1,
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 2,
            "max_steps": max_steps,
            "eval_interval": 1,
            "save_interval": 1,
            "learning_rate": 0.001,
            "resume": resume,
        },
        train_config,
    )
    return train_config
