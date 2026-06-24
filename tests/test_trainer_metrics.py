import json
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.pack import write_index, write_token_bin
from brian_sphere_llm.data.manifest import sha256_text
from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.train.trainer import (
    _bool_config,
    _ddp_no_sync_microbatch_count,
    _device,
    _distributed_mean_metrics,
    _distributed_mean_scalar,
    _float_config,
    _forward_for_stage,
    _gradient_sync_context,
    _global_train_token_count,
    _int_config,
    _learning_rate_for_step,
    _lr_schedule_config,
    _mapping_config,
    _model_stats,
    _next_train_batch,
    _restore_dataloader_position,
    _accumulate_routing_summary,
    _finalize_routing_summary,
    _route_path_visualization_config,
    _route_path_visualization_has_paths,
    _router_space_visualization_config,
    _schedule_values,
    _set_sampler_epoch,
    _wrap_distributed_model,
    _finish_wandb,
    _init_wandb,
    _wandb_log,
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


def test_learning_rate_schedule_supports_warmup_and_cosine_decay() -> None:
    assert (
        _learning_rate_for_step(
            1,
            base_learning_rate=0.1,
            min_learning_rate=0.01,
            max_steps=10,
            warmup_steps=2,
            schedule="linear_warmup_cosine_decay",
        )
        == pytest.approx(0.05)
    )
    assert _learning_rate_for_step(
        2,
        base_learning_rate=0.1,
        min_learning_rate=0.01,
        max_steps=10,
        warmup_steps=2,
        schedule="linear_warmup_cosine_decay",
    ) == pytest.approx(0.1)
    assert _learning_rate_for_step(
        10,
        base_learning_rate=0.1,
        min_learning_rate=0.01,
        max_steps=10,
        warmup_steps=2,
        schedule="linear_warmup_cosine_decay",
    ) == pytest.approx(0.01)
    assert _learning_rate_for_step(
        7,
        base_learning_rate=0.1,
        min_learning_rate=0.01,
        max_steps=10,
        warmup_steps=2,
        schedule="constant",
    ) == pytest.approx(0.1)


def test_learning_rate_schedule_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        _lr_schedule_config({"lr_schedule": "triangular"})


def test_train_config_mapping_helper_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="routing"):
        _mapping_config({"routing": True}, "routing")


def test_wandb_logging_initializes_logs_and_finishes_on_main_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeRun:
        def __init__(self) -> None:
            self.logs: list[tuple[dict[str, object], int | None]] = []
            self.summary: dict[str, object] = {}
            self.finished = False

        def log(self, payload, step=None):
            self.logs.append((payload, step))

        def finish(self):
            self.finished = True

    fake_run = FakeRun()

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))

    run = _init_wandb(
        {
            "wandb": {
                "enabled": True,
                "project": "brian-test",
                "name": "auto",
                "mode": "offline",
                "tags": ["unit"],
            }
        },
        resolved_config={"stage": "stage0_baseline"},
        model_stats={"parameter_count": 1},
        run_dir=tmp_path / "run",
        is_main_process=True,
    )
    _wandb_log(run, "train", {"step": 3, "loss": 1.5, "bad": float("nan"), "label": "ok"})
    _finish_wandb(run, final_step=3, best_eval_loss=1.25)

    assert calls["init"]["project"] == "brian-test"
    assert calls["init"]["name"] == "run"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["tags"] == ["unit"]
    assert fake_run.logs == [({"train/loss": 1.5, "train/label": "ok"}, 3)]
    assert fake_run.summary["final_step"] == 3
    assert fake_run.summary["best_eval_loss"] == 1.25
    assert fake_run.finished is True


def test_wandb_logging_skips_non_main_process(tmp_path: Path) -> None:
    run = _init_wandb(
        {"wandb": {"enabled": True}},
        resolved_config={},
        model_stats={},
        run_dir=tmp_path / "run",
        is_main_process=False,
    )

    assert run is None


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


def test_forward_for_stage_passes_routing_constraints() -> None:
    class CaptureModel:
        routing_constraints = None

        def __call__(self, *args, **kwargs):
            self.routing_constraints = kwargs["routing_constraints"]
            return {"loss": torch.tensor(0.0)}

    model = CaptureModel()
    batch = torch.randint(0, 8, (1, 4))

    _forward_for_stage(
        model,
        batch,
        config={
            "stage": "stage4_pure_free_sphere",
            "routing": {
                "mode": "free",
                "hard_exit": True,
                "constraints": {"min_exit_step": 3, "force_final_exit": True},
            },
        },
        route_mode="free",
        global_step=1,
    )

    assert model.routing_constraints == {"min_exit_step": 3, "force_final_exit": True}


def test_new_free_sphere_stage_modes_resolve() -> None:
    assert train_mode_for_stage("stage4_coverage_free_sphere") == "scheduled"
    assert train_mode_for_stage("stage4_pure_free_sphere") == "free"


def test_route_path_visualization_config_uses_save_interval_default() -> None:
    config = _route_path_visualization_config(
        {"route_path_visualization": {"enabled": True, "top_paths": 12}},
        default_interval=2500,
    )

    assert config["enabled"] is True
    assert config["interval"] == 2500
    assert config["top_paths"] == 12
    assert config["upload_to_wandb"] is True
    assert config["output_dir"] == "route_path_visualizations"


def test_route_path_visualization_has_paths_reads_sidecar(tmp_path: Path) -> None:
    html_path = tmp_path / "route_paths.html"
    html_path.with_suffix(".json").write_text(
        json.dumps({"checks": {"paths_present": True}}),
        encoding="utf-8",
    )

    assert _route_path_visualization_has_paths(html_path) is True

    html_path.with_suffix(".json").write_text(
        json.dumps({"checks": {"paths_present": False}}),
        encoding="utf-8",
    )

    assert _route_path_visualization_has_paths(html_path) is False


def test_route_path_counts_merge_across_microbatches() -> None:
    last_values: dict[str, object] = {}
    numeric_values: dict[str, list[float]] = {}

    _accumulate_routing_summary(
        last_values,
        numeric_values,
        {
            "route_path_counts": [{"actions": [0, 1, 8], "count": 2}],
            "route_transition_counts": [{"source": 0, "target": 1, "count": 2}],
            "route_path_count": 1,
        },
    )
    _accumulate_routing_summary(
        last_values,
        numeric_values,
        {
            "route_path_counts": [
                {"actions": [0, 1, 8], "count": 1},
                {"actions": [2, 3, 8], "count": 4},
            ],
            "route_transition_counts": [
                {"source": 0, "target": 1, "count": 1},
                {"source": 2, "target": 3, "count": 4},
            ],
            "route_path_count": 2,
        },
    )

    summary = _finalize_routing_summary(last_values, numeric_values)

    assert summary["route_path_count"] == pytest.approx(1.5)
    assert summary["route_path_counts"] == [
        {"actions": [2, 3, 8], "count": 4},
        {"actions": [0, 1, 8], "count": 3},
    ]
    assert summary["route_transition_counts"] == [
        {"source": 2, "target": 3, "count": 4},
        {"source": 0, "target": 1, "count": 3},
    ]


def test_router_space_visualization_config_uses_save_interval_default() -> None:
    config = _router_space_visualization_config(
        {"router_space_visualization": {"enabled": True, "max_points": 512}},
        default_interval=2500,
    )

    assert config["enabled"] is True
    assert config["interval"] == 2500
    assert config["max_points"] == 512
    assert config["upload_to_wandb"] is True
    assert config["output_dir"] == "router_space_visualizations"


def test_distributed_cuda_device_uses_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    device = _device("cuda")

    assert device.type == "cuda"
    assert device.index == 1


def test_global_train_token_count_uses_world_size_for_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert _global_train_token_count(128, distributed=True) == 1024
    assert _global_train_token_count(128, distributed=False) == 128


def test_distributed_mean_metrics_reduce_numeric_values_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import brian_sphere_llm.train.trainer as trainer_module

    calls: list[float] = []

    def fake_mean_scalar(value: float, *, device):
        calls.append(value)
        return value + 10.0

    monkeypatch.setattr(trainer_module.dist_utils, "mean_scalar", fake_mean_scalar)
    reduced = _distributed_mean_metrics(
        {
            "loss": 2.0,
            "active_block_evals_per_token": 3,
            "flag": True,
            "histogram": {"0": 2},
            "examples": [[0, 1]],
        },
        device=torch.device("cpu"),
        distributed=True,
    )

    assert calls == [2.0, 3.0]
    assert reduced["loss"] == 12.0
    assert reduced["active_block_evals_per_token"] == 13.0
    assert reduced["flag"] is True
    assert reduced["histogram"] == {"0": 2}
    assert reduced["examples"] == [[0, 1]]


def test_distributed_mean_scalar_noops_when_not_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    import brian_sphere_llm.train.trainer as trainer_module

    def fail_mean_scalar(value: float, *, device):
        raise AssertionError("mean_scalar should not be called")

    monkeypatch.setattr(trainer_module.dist_utils, "mean_scalar", fail_mean_scalar)

    assert _distributed_mean_scalar(7.0, device=torch.device("cpu"), distributed=False) == 7.0


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


def test_set_sampler_epoch_for_distributed_sampler_like_loader() -> None:
    class Sampler:
        epoch = None

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    class Loader:
        sampler = Sampler()

    loader = Loader()
    _set_sampler_epoch(loader, 4)
    assert loader.sampler.epoch == 4


def test_train_batch_helpers_restore_epoch_offset() -> None:
    class Sampler:
        def __init__(self):
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    class Loader:
        def __init__(self):
            self.sampler = Sampler()

        def __len__(self) -> int:
            return 3

        def __iter__(self):
            return iter(["a", "b", "c"])

    loader = Loader()
    iterator, data_epoch, microbatch_in_epoch = _restore_dataloader_position(
        loader,
        data_epoch=2,
        microbatch_in_epoch=1,
    )
    batch, iterator, data_epoch, microbatch_in_epoch = _next_train_batch(
        loader,
        iterator,
        data_epoch=data_epoch,
        microbatch_in_epoch=microbatch_in_epoch,
    )
    assert batch == "b"
    assert data_epoch == 2
    assert microbatch_in_epoch == 2

    batch, iterator, data_epoch, microbatch_in_epoch = _next_train_batch(
        loader,
        iterator,
        data_epoch=data_epoch,
        microbatch_in_epoch=microbatch_in_epoch,
    )
    assert batch == "c"
    assert data_epoch == 3
    assert microbatch_in_epoch == 0
    assert loader.sampler.epochs == [2, 3]


def test_gradient_sync_context_uses_no_sync_only_for_non_final_ddp_microbatch() -> None:
    class FakeNoSync:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            self.model.entered += 1

        def __exit__(self, exc_type, exc, tb):
            self.model.exited += 1

    class FakeModel:
        def __init__(self):
            self.requested = 0
            self.entered = 0
            self.exited = 0

        def no_sync(self):
            self.requested += 1
            return FakeNoSync(self)

    model = FakeModel()

    with _gradient_sync_context(model, distributed=True, should_sync=False):
        pass
    with _gradient_sync_context(model, distributed=True, should_sync=True):
        pass
    with _gradient_sync_context(model, distributed=False, should_sync=False):
        pass

    assert model.requested == 1
    assert model.entered == 1
    assert model.exited == 1
    assert _ddp_no_sync_microbatch_count(model, distributed=True, gradient_accumulation_steps=4) == 3
    assert _ddp_no_sync_microbatch_count(model, distributed=False, gradient_accumulation_steps=4) == 0
    assert _ddp_no_sync_microbatch_count(model, distributed=True, gradient_accumulation_steps=1) == 0


def test_wrap_distributed_model_passes_find_unused_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    import brian_sphere_llm.train.trainer as trainer_module

    captured: dict[str, object] = {}

    class FakeDDP:
        def __init__(self, model, **kwargs):
            self.module = model
            captured.update(kwargs)

    model = object()
    monkeypatch.setattr(trainer_module, "DistributedDataParallel", FakeDDP)

    wrapped = _wrap_distributed_model(
        model,
        torch.device("cpu"),
        distributed=True,
        find_unused_parameters=True,
        static_graph=False,
        gradient_as_bucket_view=False,
    )

    assert wrapped.module is model
    assert captured["find_unused_parameters"] is True


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
    manifest_text = json.dumps({"sample_id": "unit"}) + "\n"
    (tmp_path / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
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
                "source_mixture_expected": {"unit": 1.0},
                "source_mixture_realized": {"unit": 32},
                "source_mixture_realized_share": {"unit": 1.0},
                "sha256_manifest": sha256_text(manifest_text),
                "manifest_row_count": 1,
                "manifest_source_text_hashes_verified": True,
                "manifest_token_hashes_verified": True,
                "manifest_source_text_hash_failure_count": 0,
                "manifest_token_hash_failure_count": 0,
                "tokenizer_artifact_count": 2,
                "tokenizer_artifacts_present": True,
                "tokenizer_artifact_hashes": {
                    "tokenizer.json": "hash-tokenizer",
                    "tokenizer_config.json": "hash-config",
                },
                "tokenizer_artifact_hashes_present": True,
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
            "gradient_accumulation_steps": 2,
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
    assert manifest_ref["path_exists"] is True
    assert manifest_ref["tokenized_dir_exists"] is True
    assert manifest_ref["stats_path_exists"] is True
    assert manifest_ref["tokenized_artifacts_present"] is True
    assert manifest_ref["sha256_manifest"] == sha256_text(manifest_text)
    assert manifest_ref["sha256_manifest_verified"] is True
    assert manifest_ref["num_documents"] == 4
    assert manifest_ref["avg_tokens_per_doc"] == 8.0
    assert manifest_ref["vocab_size"] == 32
    assert manifest_ref["manifest_row_count"] == 1
    assert manifest_ref["manifest_source_text_hashes_verified"] is True
    assert manifest_ref["manifest_token_hashes_verified"] is True
    assert manifest_ref["manifest_source_text_hash_failure_count"] == 0
    assert manifest_ref["manifest_token_hash_failure_count"] == 0
    assert manifest_ref["tokenizer_artifact_count"] == 2
    assert manifest_ref["tokenizer_artifacts_present"] is True
    assert manifest_ref["tokenizer_artifact_hashes"]["tokenizer.json"] == "hash-tokenizer"
    assert manifest_ref["tokenizer_artifact_hashes_present"] is True
    assert manifest_ref["stats_recipe_name_matches_config"] is True
    assert manifest_ref["stats_sequence_length_matches_config"] is True
    assert manifest_ref["source_mixture_expected"] == {"unit": 1.0}
    assert manifest_ref["source_mixture_realized"] == {"unit": 32}
    assert manifest_ref["source_mixture_realized_share"] == {"unit": 1.0}
    assert manifest_ref["tokenizer"]["name"] == "unit-tokenizer"
    assert manifest_ref["tokenizer"]["revision"] == "local"
    assert manifest_ref["tokenizer"]["license"] == "test"
    assert manifest_ref["tokenizer"]["vocab_size"] == 32
    assert manifest_ref["tokenizer"]["special_tokens"] == {"bos": None, "eos": None, "pad": 0, "unk": None}
    report = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    assert report["latest_eval"]["validation_loss"] >= 0.0
    assert report["cost_quality_curve"]["summary"]["train_point_count"] == 1
    train_row = json.loads((run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert train_row["learning_rate"] == pytest.approx(0.001)
    assert train_row["micro_batch_size"] == 2
    assert train_row["gradient_accumulation_steps"] == 2
    assert train_row["local_effective_batch_size"] == 4
    assert train_row["effective_batch_size"] == 4
    assert train_row["local_tokens_per_optimizer_step"] == 16
    assert train_row["tokens_per_optimizer_step"] == 16
    assert train_row["distributed_world_size"] == 1
    assert train_row["ddp_find_unused_parameters"] is False
    assert train_row["ddp_no_sync_microbatches"] == 0
    assert train_row["local_tokens_per_second"] > 0


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


def test_train_from_config_can_skip_best_checkpoint(tmp_path: Path) -> None:
    train_config = _write_tiny_train_fixture(
        tmp_path,
        max_steps=1,
        resume=False,
        save_best_checkpoint=False,
    )

    run_dir = train_from_config(train_config)

    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    assert not (run_dir / "checkpoint_best").exists()


def test_train_from_config_retains_numbered_model_only_checkpoints(tmp_path: Path) -> None:
    train_config = _write_tiny_train_fixture(
        tmp_path,
        max_steps=3,
        resume=False,
        save_best_checkpoint=False,
        checkpoint_retention={"enabled": True, "interval": 1, "keep_last": 2},
    )

    run_dir = train_from_config(train_config)

    assert not (run_dir / "checkpoint_step_00000001").exists()
    assert (run_dir / "checkpoint_step_00000002" / "state.pt").exists()
    assert (run_dir / "checkpoint_step_00000003" / "state.pt").exists()
    payload = torch.load(run_dir / "checkpoint_step_00000003" / "state.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    assert "model" in payload
    assert "optimizer" not in payload


def test_train_from_config_logs_routed_behavior(tmp_path: Path) -> None:
    train_config = _write_tiny_routed_train_fixture(tmp_path)

    run_dir = train_from_config(train_config)

    train_rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_rows = [json.loads(line) for line in (run_dir / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
    train_row = train_rows[-1]
    eval_row = eval_rows[-1]
    mandatory_train_numeric_keys = [
        "route_entropy",
        "block_load_entropy",
        "active_block_evals_per_token",
        "average_route_steps",
        "p_output_mean",
        "skip_ratio",
        "recur_ratio",
        "advance_ratio",
        "location_distance_mean",
        "position_norm_mean",
        "cost_loss",
        "balance_loss",
        "location_loss",
        "route_imitation_accuracy",
    ]
    for key in mandatory_train_numeric_keys:
        assert isinstance(train_row[key], (int, float))
    for key in [
        "route_entropy",
        "block_load_entropy",
        "active_block_evals_per_token",
        "average_route_steps",
        "p_output_mean",
        "skip_ratio",
        "recur_ratio",
        "advance_ratio",
        "location_distance_mean",
        "position_norm_mean",
        "route_imitation_accuracy",
    ]:
        assert isinstance(eval_row[key], (int, float))
    assert isinstance(train_row["top1_block_histogram"], dict)
    assert isinstance(train_row["topk_block_histogram"], dict)
    assert isinstance(train_row["exit_step_distribution"], list)
    assert train_row["route_path_examples"]
    assert train_row["route_path_counts"]
    assert train_row["route_transition_counts"]
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
    assert events[-1]["rng_state_loaded"] is True
    assert events[-1]["rank_state_loaded"] is False
    assert events[-1]["rank_state_path"] is None
    assert events[-1]["data_epoch"] == 0
    assert events[-1]["microbatch_in_epoch"] == 1
    assert (run_dir / "checkpoint_latest" / "state.pt").exists()
    assert (run_dir / "checkpoint_latest" / "rank_state_00000.pt").exists()
    payload = torch.load(run_dir / "checkpoint_latest" / "state.pt", map_location="cpu", weights_only=False)
    rank_payload = torch.load(run_dir / "checkpoint_latest" / "rank_state_00000.pt", map_location="cpu", weights_only=False)
    assert payload["data_epoch"] == 1
    assert payload["microbatch_in_epoch"] == 0
    assert "rng_state" in payload
    assert rank_payload["rank"] == 0
    assert rank_payload["data_epoch"] == 1
    assert rank_payload["microbatch_in_epoch"] == 0
    assert "rng_state" in rank_payload
    train_rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train_rows[-1]["step"] == 2


def _write_tiny_train_fixture(
    tmp_path: Path,
    *,
    max_steps: int,
    resume: bool,
    write_routing_report_on_checkpoint: bool = True,
    save_best_checkpoint: bool = True,
    checkpoint_retention: dict[str, object] | None = None,
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
            "save_best_checkpoint": save_best_checkpoint,
            "checkpoint_retention": checkpoint_retention or {"enabled": False},
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
            "routing": {"pseudo_policy": "sequential", "log_path_counts": True},
            "loss_weights": {"route": 1.0, "balance": 0.01, "cost": 0.01, "location": 0.01},
        },
        train_config,
    )
    return train_config
