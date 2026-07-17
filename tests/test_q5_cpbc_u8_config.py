from brian_sphere_llm.utils.config import load_config


U1_TRAIN = "configs/train/q1_cpbc_r125_250m_fb_u1_c128_triton_ddp2_legacyval.yaml"
U8_TRAIN = "configs/train/q5_cpbc_r125_250m_fb_u8_c128_triton_ddp2_legacyval.yaml"


def test_q5_u8_keeps_q1_fb_forward_and_training_contract() -> None:
    u1 = load_config(U1_TRAIN)
    u8 = load_config(U8_TRAIN)

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
        assert u1[key] == u8[key], key

    assert u8["max_steps"] * u8["expected_global_batch_size"] * 2048 == 250_019_840
    assert u1["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 128,
        "detach_interval_chunks": 1,
        "gradient_sync_bucket_mb": 64,
    }
    assert u8["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 128,
        "detach_interval_chunks": 8,
        "gradient_sync_bucket_mb": 64,
    }


def test_q5_u8_keeps_q1_checkpoint_benchmark_contract() -> None:
    u1 = load_config(U1_TRAIN)
    u8 = load_config(U8_TRAIN)

    assert u1["checkpoint_benchmarks"]["interval"] == u8["checkpoint_benchmarks"]["interval"] == 1272
    assert u1["checkpoint_benchmarks"]["reasoning"] == u8["checkpoint_benchmarks"]["reasoning"]
    assert u1["checkpoint_benchmarks"]["public"] == u8["checkpoint_benchmarks"]["public"]
    assert u8["checkpoint_retention"] == u1["checkpoint_retention"]
