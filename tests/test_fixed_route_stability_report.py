import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.eval.fixed_route_stability import make_fixed_route_stability_report
from brian_sphere_llm.train.trainer import train_from_config
from brian_sphere_llm.utils.config import load_config, save_yaml


def test_fixed_route_stability_report_passes_for_tiny_stage1_run(tmp_path: Path) -> None:
    run_dir = _train_tiny_fixed_route(tmp_path)

    output = make_fixed_route_stability_report(
        run_dir,
        output_path=tmp_path / "fixed_route.json",
        max_batches=1,
        device_name="cpu",
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["checks"]["logits_shape_matches"] is True
    assert report["checks"]["logits_finite"] is True
    assert report["checks"]["fixed_route_matches_targets"] is True
    assert report["checks"]["route_imitation_accuracy_is_one"] is True
    assert report["routing_summary"]["route_imitation_accuracy"] == 1.0


def test_fixed_route_stability_eval_config_resolves() -> None:
    cfg = load_config("configs/eval/fixed_route_stability.yaml")
    assert cfg["eval_name"] == "fixed_route_stability_report"


def _train_tiny_fixed_route(tmp_path: Path) -> Path:
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

    baseline_config = tmp_path / "baseline.yaml"
    save_yaml(
        {
            "model_name": "baseline_unit",
            "architecture": "decoder_only_llama_like",
            "layers": 4,
            "d_model": 16,
            "n_heads": 4,
            "context_length": 4,
            "vocab_size": 32,
            "dropout": 0.0,
        },
        baseline_config,
    )
    model_config = tmp_path / "brian.yaml"
    save_yaml(
        {
            "model_name": "brian_unit",
            "architecture": "brian_route_core",
            "base_config": str(baseline_config),
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
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
            "stage": "stage1_fixed_route",
            "run_name": "unit_fixed_route",
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
            "routing": {"mode": "fixed", "pseudo_policy": "sequential"},
            "loss_weights": {"route": 1.0, "balance": 0.01, "cost": 0.001, "location": 0.05},
        },
        train_config,
    )
    return train_from_config(train_config)
