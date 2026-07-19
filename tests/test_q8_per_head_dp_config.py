from pathlib import Path

from brian_sphere_llm.model.bdre_model import BDREConfig
from brian_sphere_llm.utils.config import load_config


CONFIG_ROOT = Path("configs")
MODEL = CONFIG_ROOT / "model/brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d32.yaml"
TRAIN = CONFIG_ROOT / "train/q8_cpbc_r125_250m_dp_u1_c2048_per_head_d32_ddp2_legacyval.yaml"
SMOKE = CONFIG_ROOT / "train/smoke_q8_cpbc_r125_250m_dp_u1_c2048_per_head_d32_ddp2.yaml"


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
