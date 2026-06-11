import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.eval.determinism_report import make_eval_determinism_report
from brian_sphere_llm.train.trainer import train_from_config
from brian_sphere_llm.utils.config import save_yaml


def test_eval_determinism_report_passes_for_repeated_baseline_eval(tmp_path: Path) -> None:
    run_dir = _train_tiny_baseline(tmp_path)
    output = make_eval_determinism_report(
        run_dir,
        output_path=tmp_path / "determinism.json",
        device_name="cpu",
        tolerance=1e-8,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["checks"]["checkpoint_loaded"] is True
    assert report["checks"]["numeric_metrics_within_tolerance"] is True
    assert report["comparison"]["compared_metric_count"] >= 2
    assert not report["comparison"]["mismatched_metrics"]
    metric_names = {item["metric"] for item in report["comparison"]["metrics"]}
    assert "validation_loss" in metric_names


def _train_tiny_baseline(tmp_path: Path) -> Path:
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
        },
        train_config,
    )
    return train_from_config(train_config)
