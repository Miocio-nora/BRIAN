import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.trainer import (
    _bool_config,
    _float_config,
    _forward_for_stage,
    _int_config,
    _mapping_config,
    _model_stats,
    _schedule_values,
    evaluate,
    run_name,
    train_from_config,
)
from brian_sphere_llm.utils.config import save_yaml


def test_auto_run_name_includes_context_length() -> None:
    name = run_name(
        {"stage": "stage3_scheduled_free_routing", "seed": 7},
        "brian_r125",
        "r125main2b",
        context_length=2048,
    )

    assert name.endswith("_brian_r125_stage3_scheduled_free_routing_r125main2b_ctx2048_seed7")
    assert run_name({"stage": "stage0_baseline", "run_name": "manual"}, "model", "data", context_length=8) == "manual"


def test_schedule_values_rejects_boolean_default_lambda_route() -> None:
    config = {
        "loss_weights": {"route": True},
        "routing": {"schedule": [{"max_step": 1, "router_probability": 0.5}]},
    }

    with pytest.raises(ValueError, match="lambda_route"):
        _schedule_values(config, route_mode="scheduled", global_step=1)


def test_train_config_numeric_helpers_reject_boolean_values() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        _int_config({"max_steps": True}, "max_steps", minimum=1)
    with pytest.raises(ValueError, match="learning_rate"):
        _float_config({"learning_rate": False}, "learning_rate", minimum=0.0)
    with pytest.raises(ValueError, match="grad_clip"):
        _float_config({"grad_clip": True}, "grad_clip", minimum=0.0)


def test_train_config_bool_helper_parses_strings_and_rejects_non_boolean() -> None:
    assert _bool_config({"resume": "false"}, "resume", default=True) is False
    assert _bool_config({"resume": "true"}, "resume", default=False) is True
    with pytest.raises(ValueError, match="resume"):
        _bool_config({"resume": 1}, "resume", default=False)


def test_train_config_mapping_helper_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="routing"):
        _mapping_config({"routing": True}, "routing")


def test_forward_for_stage_parses_string_false_hard_exit() -> None:
    class CaptureModel:
        hard_exit = None

        def __call__(self, *args, **kwargs):
            self.hard_exit = kwargs["hard_exit"]
            return {"loss": torch.tensor(0.0)}

    model = CaptureModel()
    batch = torch.randint(0, 8, (1, 4))

    _forward_for_stage(
        model,
        batch,
        config={"stage": "stage4_output_action", "routing": {"hard_exit": "false"}},
        route_mode="free",
        global_step=1,
    )

    assert model.hard_exit is False


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


def test_model_stats_requires_positive_parameter_count() -> None:
    class MissingStats:
        pass

    class MissingParameterCount:
        def model_stats(self) -> dict:
            return {"model_name": "missing"}

    class InvalidParameterCount:
        def model_stats(self) -> dict:
            return {"model_name": "invalid", "parameter_count": 0}

    class BoolParameterCount:
        def model_stats(self) -> dict:
            return {"model_name": "bool", "parameter_count": True}

    with pytest.raises(ValueError, match="must expose model_stats"):
        _model_stats(MissingStats())
    with pytest.raises(ValueError, match="must include parameter_count"):
        _model_stats(MissingParameterCount())
    with pytest.raises(ValueError, match="positive integer"):
        _model_stats(InvalidParameterCount())
    with pytest.raises(ValueError, match="positive integer"):
        _model_stats(BoolParameterCount())


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
    (tokenized / "stats.json").write_text(
        json.dumps(
            {
                "recipe_name": "unit_data",
                "num_documents": 4,
                "num_tokens_train": 16,
                "num_tokens_val": 16,
                "avg_tokens_per_doc": 8.0,
                "sequence_length": 4,
                "vocab_size": 32,
                "source_mixture_realized": {"unit": 32},
                "sha256_manifest": "abc123",
                "tokenizer": {
                    "name": "unit-tokenizer",
                    "revision": "local",
                    "license": "test",
                    "vocab_size": 32,
                    "special_tokens": {"bos": None, "eos": None, "pad": 0, "unk": None},
                },
            }
        ),
        encoding="utf-8",
    )

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

    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "train_log.jsonl").exists()
    assert (run_dir / "eval_log.jsonl").exists()
    assert (run_dir / "model_stats.json").exists()
    assert (run_dir / "data_manifest_ref.json").exists()
    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    assert (run_dir / "checkpoint_best" / "state.pt").exists()
    assert (run_dir / "routing_report.json").exists()
    model_stats = json.loads((run_dir / "model_stats.json").read_text(encoding="utf-8"))
    assert model_stats["model_name"] == "baseline_unit"
    assert model_stats["parameter_count"] > 0
    manifest_ref = json.loads((run_dir / "data_manifest_ref.json").read_text(encoding="utf-8"))
    assert manifest_ref["path"] == str(tmp_path / "manifest.jsonl")
    assert manifest_ref["sha256_manifest"] == "abc123"
    assert manifest_ref["source_mixture_realized"] == {"unit": 32}
    assert manifest_ref["tokenizer"]["name"] == "unit-tokenizer"
    report = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    assert report["latest_eval"]["validation_loss"] >= 0.0
    assert report["cost_quality_curve"]["summary"]["train_point_count"] == 1


def test_train_from_config_writes_final_routing_report_when_checkpoint_report_disabled(tmp_path: Path) -> None:
    train_config = _write_tiny_train_fixture(
        tmp_path,
        max_steps=1,
        resume=False,
        write_routing_report_on_checkpoint=False,
    )

    run_dir = train_from_config(train_config)

    assert (run_dir / "routing_report.json").exists()
    report = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    assert report["latest_eval"]["validation_loss"] >= 0.0
    assert report["cost_quality_curve"]["summary"]["train_point_count"] == 1


def test_train_from_config_logs_routed_behavior(tmp_path: Path) -> None:
    train_config = _write_tiny_routed_train_fixture(tmp_path)

    run_dir = train_from_config(train_config)

    train_rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(line) for line in (run_dir / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
    train_row = train_rows[-1]
    eval_row = eval_rows[-1]
    for key in [
        "route_entropy",
        "block_load_entropy",
        "active_block_evals_per_token",
        "average_route_steps",
        "route_imitation_accuracy",
    ]:
        assert isinstance(train_row[key], (int, float))
        assert isinstance(eval_row[key], (int, float))
    assert isinstance(train_row["top1_block_histogram"], dict)
    assert train_row["route_path_examples"]
    report = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["route_entropy"] >= 0.0
    assert report["latest_route_path_examples"]
    model_stats = json.loads((run_dir / "model_stats.json").read_text(encoding="utf-8"))
    assert model_stats["model_name"] == "brian_unit_routed"
    assert model_stats["parameter_count"] > 0


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


def _write_tiny_train_fixture(
    tmp_path: Path,
    *,
    max_steps: int,
    resume: bool,
    write_routing_report_on_checkpoint: bool = True,
) -> Path:
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
            "write_routing_report_on_checkpoint": write_routing_report_on_checkpoint,
        },
        train_config,
    )
    return train_config


def _write_tiny_routed_train_fixture(tmp_path: Path) -> Path:
    tokenized = tmp_path / "tokenized_routed"
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

    model_config = tmp_path / "model_routed.yaml"
    save_yaml(
        {
            "model_name": "brian_unit_routed",
            "architecture": "brian_route_core",
            "base": {
                "model_name": "baseline_unit_routed",
                "layers": 4,
                "d_model": 16,
                "n_heads": 4,
                "context_length": 4,
                "vocab_size": 32,
                "dropout": 0.0,
            },
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "top_k": 1,
            "later_top_k": 1,
            "hard_exit": False,
        },
        model_config,
    )
    data_config = tmp_path / "data_routed.yaml"
    save_yaml(
        {
            "recipe_name": "unit_data_routed",
            "output_dir": str(tokenized),
            "manifest_path": str(tmp_path / "manifest_routed.jsonl"),
            "sequence_length": 4,
        },
        data_config,
    )
    train_config = tmp_path / "train_routed.yaml"
    save_yaml(
        {
            "stage": "stage1_fixed_route",
            "run_name": "unit_routed_run",
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
            "routing": {"pseudo_policy": "sequential"},
            "loss_weights": {"route": 1.0, "balance": 0.01, "cost": 0.01, "location": 0.01},
        },
        train_config,
    )
    return train_config
