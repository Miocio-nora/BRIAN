from pathlib import Path

from brian_sphere_llm.model.bdre_model import BDREConfig
from brian_sphere_llm.utils.config import load_config


CONFIG_ROOT = Path("configs")
MODEL = CONFIG_ROOT / "model/brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d32.yaml"
TRAIN = CONFIG_ROOT / "train/q8_cpbc_r125_250m_dp_u1_c2048_per_head_d32_ddp2_legacyval.yaml"
SMOKE = CONFIG_ROOT / "train/smoke_q8_cpbc_r125_250m_dp_u1_c2048_per_head_d32_ddp2.yaml"
Q9_TRAIN = CONFIG_ROOT / "train/q9_cpbc_r125_5b_dp_u1_c2048_per_head_d32_ddp2_legacyval.yaml"
Q10_MODEL = CONFIG_ROOT / "model/brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d64.yaml"
Q10_TRAIN = CONFIG_ROOT / "train/q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2_legacyval.yaml"
Q10_SMOKE = CONFIG_ROOT / "train/smoke_q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2.yaml"


def test_q8_model_combines_strict_per_head_cache_with_dp_c2048() -> None:
    raw = load_config(MODEL)
    config = BDREConfig.from_dict(raw, config_dir=MODEL.parent)

    assert config.cache_layout == "per_head"
    assert config.key_dim == config.value_dim == 32
    assert config.depth_visibility_policy == "depth_prefix"
    assert config.chunk_size == 2048
    assert config.dispatch_mode == "grouped_mm_gpu"
    assert config.synchronous_attention_backend == "shared_padded_flex_blockmask"
    assert config.writer_projection_mode == "precomposed"
    assert config.prefix_compile_mode == "recompute"
    assert config.reader_kernel_mode == "triton_fused"


def test_q8_train_preserves_the_250m_ddp2_contract() -> None:
    config = load_config(TRAIN)

    assert config["max_steps"] == 3815
    assert config["batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 1
    assert config["expected_world_size"] == 2
    assert config["expected_global_batch_size"] == 32
    assert config["stateful_tbptt"]["chunk_size"] == 2048
    assert config["stateful_tbptt"]["detach_interval_chunks"] == 1
    assert config["max_steps"] * config["expected_global_batch_size"] * 2048 == 250_019_840
    assert config["checkpoint_benchmarks"]["enabled"] is False
    assert config["checkpoint_retention"]["keep_last"] == 4


def test_q8_smoke_disables_external_side_effects() -> None:
    config = load_config(SMOKE)

    assert config["max_steps"] == 20
    assert config["eval_max_batches"] == 1
    assert config["checkpoint_retention"]["enabled"] is False
    assert config["checkpoint_benchmarks"]["enabled"] is False
    assert config["wandb"]["enabled"] is False


def test_q9_uses_the_matched_5b_dp_contract_without_inline_benchmarks() -> None:
    config = load_config(Q9_TRAIN)

    assert config["max_steps"] == 76294
    assert config["batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 1
    assert config["expected_world_size"] == 2
    assert config["expected_global_batch_size"] == 32
    assert config["stateful_tbptt"]["chunk_size"] == 2048
    assert config["stateful_tbptt"]["detach_interval_chunks"] == 1
    assert config["model_config"].endswith(
        "brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d32.yaml"
    )
    assert config["max_steps"] * config["expected_global_batch_size"] * 2048 == 5_000_003_584
    assert config["checkpoint_benchmarks"]["enabled"] is False
    assert config["post_train_benchmarks"]["enabled"] is False
    assert config["checkpoint_retention"]["interval"] == 15000
    assert config["checkpoint_retention"]["keep_last"] == 6


def test_q10_changes_only_the_per_head_cache_dimension() -> None:
    d32 = BDREConfig.from_dict(load_config(MODEL), config_dir=MODEL.parent)
    d64 = BDREConfig.from_dict(load_config(Q10_MODEL), config_dir=Q10_MODEL.parent)

    assert d64.cache_layout == "per_head"
    assert d64.key_dim == d64.value_dim == 64
    assert d64.route.base.d_model // d64.route.base.n_heads == 64
    for field in (
        "depth_visibility_policy",
        "chunk_size",
        "dispatch_mode",
        "synchronous_attention_backend",
        "writer_projection_mode",
        "prefix_compile_mode",
        "reader_kernel_mode",
    ):
        assert getattr(d64, field) == getattr(d32, field)


def test_q10_preserves_the_q9_5b_training_contract() -> None:
    q9 = load_config(Q9_TRAIN)
    q10 = load_config(Q10_TRAIN)

    for key in (
        "data_config",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "expected_world_size",
        "expected_global_batch_size",
        "stateful_tbptt",
        "checkpoint_retention",
    ):
        assert q10[key] == q9[key]
    assert q10["model_config"].endswith(
        "brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d64.yaml"
    )
    assert q10["checkpoint_benchmarks"]["enabled"] is False
    assert q10["post_train_benchmarks"]["enabled"] is False


def test_q10_smoke_disables_external_side_effects() -> None:
    config = load_config(Q10_SMOKE)

    assert config["max_steps"] == 20
    assert config["eval_max_batches"] == 1
    assert config["checkpoint_retention"]["enabled"] is False
    assert config["checkpoint_benchmarks"]["enabled"] is False
    assert config["wandb"]["enabled"] is False
