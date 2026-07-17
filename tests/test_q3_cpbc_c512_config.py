from copy import deepcopy

from brian_sphere_llm.utils.config import load_config


C128_TRAIN = "configs/train/q1_cpbc_r125_250m_fb_u1_c128_triton_ddp2_legacyval.yaml"
C512_TRAIN = "configs/train/q3_cpbc_r125_250m_fb_u1_c512_triton_ddp2_legacyval.yaml"
C128_MODEL = "configs/model/brian_r125_bdre_cpbc_fb_c128_triton_fused_reader_incremental.yaml"
C512_MODEL = "configs/model/brian_r125_bdre_cpbc_fb_c512_triton_fused_reader_incremental.yaml"


def test_q3_c512_keeps_q1_fb_training_contract() -> None:
    c128 = load_config(C128_TRAIN)
    c512 = load_config(C512_TRAIN)

    for key in (
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
        assert c128[key] == c512[key], key

    assert c512["max_steps"] * c512["expected_global_batch_size"] * 2048 == 250_019_840
    assert c128["stateful_tbptt"]["chunk_size"] == 128
    assert c512["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 512,
        "detach_interval_chunks": 1,
        "gradient_sync_bucket_mb": 64,
    }
    assert c512["checkpoint_benchmarks"]["interval"] == 1272
    assert c512["checkpoint_benchmarks"]["reasoning"]["config"].endswith(
        "reasoning_eval_s600_incremental.yaml"
    )
    assert c512["checkpoint_benchmarks"]["public"]["config"].endswith(
        "public_benchmark_s600.yaml"
    )


def test_q3_c512_model_diff_is_limited_to_chunk_size() -> None:
    c128 = _normalized_model_config(load_config(C128_MODEL))
    c512 = _normalized_model_config(load_config(C512_MODEL))

    assert c128 == c512


def test_q3_c512_full_bank_boundaries_match_training_chunks() -> None:
    train = load_config(C512_TRAIN)
    model = load_config(C512_MODEL)

    assert model["execution"]["depth_visibility_policy"] == "full_bank"
    assert model["execution"]["full_bank_compile"] == "vectorized"
    assert model["execution"]["reader_kernel"] == "triton_fused"
    assert model["execution"]["chunk_size"] == train["stateful_tbptt"]["chunk_size"] == 512


def _normalized_model_config(config):
    normalized = deepcopy(config)
    normalized.pop("model_name")
    normalized["execution"].pop("chunk_size")
    return normalized
