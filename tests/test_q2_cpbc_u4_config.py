from brian_sphere_llm.utils.config import load_config


U1_TRAIN = "configs/train/q1_cpbc_r125_250m_fb_u1_c128_triton_ddp2_legacyval.yaml"
U4_TRAIN = "configs/train/q2_cpbc_r125_250m_fb_u4_c128_triton_ddp2_legacyval.yaml"


def test_q2_u4_keeps_q1_fb_forward_and_training_contract() -> None:
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
        "resume",
        "learning_rate",
        "weight_decay",
        "data_config",
        "eval_split",
        "routing",
        "loss_weights",
    ):
        assert u1[key] == u4[key], key

    assert u4["max_steps"] * u4["expected_global_batch_size"] * 2048 == 250_019_840
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


def test_q2_u4_keeps_q1_checkpoint_benchmark_contract() -> None:
    u1 = load_config(U1_TRAIN)
    u4 = load_config(U4_TRAIN)

    assert u1["checkpoint_benchmarks"]["interval"] == u4["checkpoint_benchmarks"]["interval"] == 1272
    assert u1["checkpoint_benchmarks"]["reasoning"] == u4["checkpoint_benchmarks"]["reasoning"]
    assert u1["checkpoint_benchmarks"]["public"] == u4["checkpoint_benchmarks"]["public"]
    assert u4["checkpoint_retention"] == u1["checkpoint_retention"]
