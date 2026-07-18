from pathlib import Path

import pytest

from brian_sphere_llm.model.bdre_model import BDREConfig
from brian_sphere_llm.utils.config import load_config


CONFIG_ROOT = Path("configs")
Q2_TRAIN = CONFIG_ROOT / "train/q2_cpbc_r125_250m_fb_u4_c128_triton_ddp2_legacyval.yaml"


@pytest.mark.parametrize("cache_dim", [16, 32, 64])
def test_q7_per_head_models_keep_fb_u4_execution_contract(cache_dim: int) -> None:
    path = CONFIG_ROOT / (
        "model/brian_r125_bdre_cpbc_fb_c128_"
        f"per_head_flex_incremental_d{cache_dim}.yaml"
    )
    raw = load_config(path)
    config = BDREConfig.from_dict(raw, config_dir=path.parent)

    assert config.cache_layout == "per_head"
    assert config.key_dim == config.value_dim == cache_dim
    assert config.execution_mode == "synchronous_prefix"
    assert config.chunk_size == 128
    assert config.depth_visibility_policy == "full_bank"
    assert config.full_bank_compile_mode == "vectorized"
    assert config.dispatch_mode == "grouped_mm_gpu"
    assert config.synchronous_attention_backend == "shared_padded_flex_blockmask"
    assert config.writer_projection_mode == "precomposed"
    assert config.prefix_compile_mode == "incremental_exact"
    assert config.reader_kernel_mode == "flex"


@pytest.mark.parametrize("cache_dim", [16, 32, 64])
def test_q7_optimized_models_use_equivalent_triton_recompute_path(cache_dim: int) -> None:
    path = CONFIG_ROOT / (
        "model/brian_r125_bdre_cpbc_fb_c128_"
        f"per_head_triton_recompute_d{cache_dim}.yaml"
    )
    raw = load_config(path)
    config = BDREConfig.from_dict(raw, config_dir=path.parent)

    assert config.cache_layout == "per_head"
    assert config.key_dim == config.value_dim == cache_dim
    assert config.depth_visibility_policy == "full_bank"
    assert config.chunk_size == 128
    assert config.dispatch_mode == "grouped_mm_gpu"
    assert config.synchronous_attention_backend == "shared_padded_flex_blockmask"
    assert config.writer_projection_mode == "precomposed"
    assert config.prefix_compile_mode == "recompute"
    assert config.reader_kernel_mode == "triton_fused"


@pytest.mark.parametrize("cache_dim", [16, 32, 64])
def test_q7_per_head_training_changes_only_architecture_identity(cache_dim: int) -> None:
    reference = load_config(Q2_TRAIN)
    path = CONFIG_ROOT / (
        "train/q7_cpbc_r125_250m_fb_u4_c128_"
        f"per_head_d{cache_dim}_ddp2_legacyval.yaml"
    )
    candidate = load_config(path)

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
        assert candidate[key] == reference[key], key

    assert f"per_head_d{cache_dim}" in candidate["run_name"]
    assert f"per_head_triton_recompute_d{cache_dim}.yaml" in candidate["model_config"]
    assert candidate["max_steps"] * candidate["expected_global_batch_size"] * 2048 == 250_019_840
    assert candidate["checkpoint_benchmarks"]["reasoning"] == reference["checkpoint_benchmarks"]["reasoning"]
    assert candidate["checkpoint_benchmarks"]["public"] == reference["checkpoint_benchmarks"]["public"]


@pytest.mark.parametrize("cache_dim", [16, 32, 64])
def test_q7_per_head_smoke_configs_disable_external_side_effects(cache_dim: int) -> None:
    path = CONFIG_ROOT / (
        "train/smoke_q7_cpbc_r125_250m_fb_u4_c128_"
        f"per_head_d{cache_dim}_ddp2.yaml"
    )
    config = load_config(path)

    assert config["max_steps"] == 20
    assert config["checkpoint_benchmarks"]["enabled"] is False
    assert config["checkpoint_retention"]["enabled"] is False
    assert config["wandb"]["enabled"] is False
