from brian_sphere_llm.utils.config import load_config


U1_TRAIN = (
    "configs/train/"
    "cpbc_r125_5b_fb_u1_c128_triton_fused_reader_incremental_ddp2_legacyval.yaml"
)
U4_TRAIN = (
    "configs/train/"
    "cpbc_r125_5b_fb_u4_c128_triton_fused_reader_incremental_ddp2_legacyval.yaml"
)


def test_formal_fb_u4_keeps_optimized_u1_training_contract() -> None:
    u1 = load_config(U1_TRAIN)
    u4 = load_config(U4_TRAIN)

    for key in (
        "model_config",
        "max_steps",
        "eval_interval",
        "save_interval",
        "batch_size",
        "gradient_accumulation_steps",
        "expected_world_size",
        "expected_global_batch_size",
        "seed",
        "learning_rate",
        "weight_decay",
        "data_config",
        "eval_split",
        "routing",
        "loss_weights",
    ):
        assert u1[key] == u4[key], key

    assert "triton_fused_reader_incremental" in u4["model_config"]
    assert u4["distributed_timeout_seconds"] == 1_800
    assert u4["resume"] is True
    assert u1["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 128,
        "detach_interval_chunks": 1,
        "gradient_sync_bucket_mb": 64,
    }
    assert u4["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 128,
        "detach_interval_chunks": 4,
        "gradient_sync_bucket_mb": 64,
    }


def test_formal_fb_u4_keeps_checkpoint_benchmark_contract() -> None:
    u1 = load_config(U1_TRAIN)
    u4 = load_config(U4_TRAIN)

    assert u1["checkpoint_benchmarks"]["interval"] == 15_000
    assert u1["checkpoint_benchmarks"]["reasoning"] == u4["checkpoint_benchmarks"]["reasoning"]
    assert u1["checkpoint_benchmarks"]["public"] == u4["checkpoint_benchmarks"]["public"]
    assert u4["checkpoint_benchmarks"]["reasoning"]["config"].endswith(
        "reasoning_eval_s600_incremental.yaml"
    )
    assert u4["post_train_benchmarks"]["reasoning"]["config"].endswith(
        "reasoning_eval_s600_incremental.yaml"
    )
    assert u1["checkpoint_retention"] == u4["checkpoint_retention"]
