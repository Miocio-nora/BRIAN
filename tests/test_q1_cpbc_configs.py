from copy import deepcopy

from brian_sphere_llm.utils.config import load_config


DP_TRAIN = "configs/train/q1_cpbc_r125_250m_dp_u1_c128_triton_ddp2_legacyval.yaml"
FB_TRAIN = "configs/train/q1_cpbc_r125_250m_fb_u1_c128_triton_ddp2_legacyval.yaml"
DP_MODEL = "configs/model/brian_r125_bdre_cpbc_dp_c128_triton_fused_reader_incremental.yaml"
FB_MODEL = "configs/model/brian_r125_bdre_cpbc_fb_c128_triton_fused_reader_incremental.yaml"


def test_q1_visibility_train_contract_is_matched() -> None:
    dp = load_config(DP_TRAIN)
    fb = load_config(FB_TRAIN)

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
        "stateful_tbptt",
    ):
        assert dp[key] == fb[key], key

    assert dp["max_steps"] == 3815
    assert dp["resume"] is True
    assert dp["max_steps"] * dp["expected_global_batch_size"] * 2048 == 250_019_840
    assert dp["stateful_tbptt"] == {
        "enabled": True,
        "chunk_size": 128,
        "detach_interval_chunks": 1,
        "gradient_sync_bucket_mb": 64,
    }
    assert dp["checkpoint_benchmarks"]["interval"] == 1272
    assert fb["checkpoint_benchmarks"]["interval"] == 1272
    assert dp["checkpoint_benchmarks"]["reasoning"]["config"].endswith(
        "reasoning_eval_s600_incremental.yaml"
    )
    assert fb["checkpoint_benchmarks"]["public"]["config"].endswith(
        "public_benchmark_s600.yaml"
    )


def test_q1_visibility_model_contract_diff_is_limited_to_visibility() -> None:
    dp = _normalized_model_config(load_config(DP_MODEL))
    fb = _normalized_model_config(load_config(FB_MODEL))

    assert dp == fb


def _normalized_model_config(config):
    normalized = deepcopy(config)
    normalized.pop("model_name")
    execution = normalized["execution"]
    execution.pop("depth_visibility_policy")
    execution.pop("full_bank_compile", None)
    return normalized
