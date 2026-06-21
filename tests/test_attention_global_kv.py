import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.memory import CanonicalAttentionGlobalKVCache
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.model.llama_backbone import BackboneConfig, CausalSelfAttention


def test_attention_global_cache_keeps_sink_window_and_valid_mask() -> None:
    cache = CanonicalAttentionGlobalKVCache(sink_slots=1, window_slots=2)
    state = cache.empty(batch_size=2, n_heads=1, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    for value in range(5):
        key = torch.full((2, 1, 2), float(value))
        valid = torch.tensor([True, value % 2 == 0])
        state = cache.write(state, key, key + 0.5, valid)

    assert state.keys.shape == (2, 1, 3, 2)
    assert state.values.shape == (2, 1, 3, 2)
    assert state.valid.shape == (2, 3)
    assert state.keys[0, 0, :, 0].tolist() == [0.0, 3.0, 4.0]
    assert state.valid[0].tolist() == [True, True, True]
    assert state.valid[1].tolist() == [True, False, True]


def test_causal_attention_reads_attention_global_kv_prefix() -> None:
    cfg = BackboneConfig(
        vocab_size=32,
        context_length=8,
        layers=1,
        d_model=16,
        n_heads=4,
        attention_global_logit_bias_init=-2.0,
        attention_global_sink_slots=1,
    )
    attention = CausalSelfAttention(cfg)
    cache = CanonicalAttentionGlobalKVCache(sink_slots=1, window_slots=2)
    state = cache.empty(batch_size=2, n_heads=4, head_dim=4, device=torch.device("cpu"), dtype=torch.float32)
    state = cache.write(
        state,
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 4),
        torch.tensor([True, False]),
    )
    x = torch.randn(2, 5, 16)
    output, key_summary, value_summary, metrics = attention(x, state, return_attention_kv=True)

    assert output.shape == x.shape
    assert key_summary.shape == (2, 4, 4)
    assert value_summary.shape == (2, 4, 4)
    assert "attention_global_kv_last_token_mass" in metrics
    assert "attention_global_kv_sink_last_token_mass" in metrics
    assert metrics["attention_global_kv_logit_bias"].item() == pytest.approx(-2.0)


def test_causal_attention_token_compressed_writes_all_token_kv_slots() -> None:
    cfg = BackboneConfig(
        vocab_size=32,
        context_length=8,
        layers=1,
        d_model=16,
        n_heads=4,
        attention_global_logit_bias_init=-2.0,
        attention_global_sink_slots=1,
        attention_global_kv_mode="token_compressed",
        attention_global_code_dim=2,
    )
    attention = CausalSelfAttention(cfg)
    cache = CanonicalAttentionGlobalKVCache(sink_slots=1, window_slots=2)
    state = cache.empty(batch_size=2, n_heads=4, head_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 5, 16)

    output, key_write, value_write, metrics = attention(x, state, return_attention_kv=True)
    state = cache.write(state, key_write, value_write, torch.ones(2, 5, dtype=torch.bool))
    output_with_global, _, _, global_metrics = attention(x, state, return_attention_kv=True)

    assert output.shape == x.shape
    assert output_with_global.shape == x.shape
    assert key_write.shape == (2, 4, 5, 2)
    assert value_write.shape == (2, 4, 5, 2)
    assert state.keys.shape == (2, 4, 3, 2)
    assert state.valid.shape == (2, 3)
    assert attention.global_key_write is not None
    assert attention.global_key_read is not None
    assert "attention_global_kv_last_token_mass" in metrics
    assert global_metrics["attention_global_kv_last_token_mass"].item() >= 0.0


def test_brian_attention_global_kv_is_route_only_and_reports_metrics() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=3,
        top_k=1,
        later_top_k=2,
        hard_exit=True,
        attention_global_kv=True,
        attention_global_sink_slots=1,
        attention_global_window_slots=2,
    )
    model = BrianRouteCore(cfg)
    assert model.pre_blocks[0].attn.global_logit_bias is None
    assert model.post_blocks[0].attn.global_logit_bias is None
    assert model.route_blocks[0].block.attn.global_logit_bias is not None

    input_ids = torch.randint(0, 64, (2, 8))
    output = model(
        input_ids,
        targets=input_ids,
        route_mode="scheduled",
        pseudo_policy="balanced_coverage",
        router_probability=0.0,
        hard_exit=True,
    )
    summary = output["routing_summary"]
    assert summary["attention_global_kv_slots_mean"] >= 1.0
    assert summary["attention_global_kv_slots_max"] <= 3.0
    assert summary["attention_global_kv_write_count_mean"] == pytest.approx(2.0)
    assert "attention_global_kv_last_token_mass" in summary
    assert model.model_stats()["attention_global_kv"] == "True"


def test_brian_attention_global_kv_token_compressed_reports_token_writes() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        top_k=1,
        hard_exit=True,
        attention_global_kv=True,
        attention_global_kv_mode="token_compressed",
        attention_global_code_dim=2,
        attention_global_sink_slots=1,
        attention_global_window_slots=2,
    )
    model = BrianRouteCore(cfg)
    assert model.route_blocks[0].block.attn.global_key_write is not None
    assert model.route_blocks[0].block.attn.global_key_read is not None
    assert model.model_stats()["attention_global_kv_mode"] == "token_compressed"
    assert model.model_stats()["attention_global_code_dim"] == "2"

    input_ids = torch.randint(0, 64, (2, 8))
    output = model(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
        hard_exit=True,
    )
    summary = output["routing_summary"]
    assert summary["attention_global_kv_slots_max"] <= 3.0
    assert summary["attention_global_kv_write_count_mean"] > 2.0
    assert "attention_global_kv_last_token_mass" in summary


def test_attention_global_kv_rejects_parallel_passing() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        attention_global_kv=True,
        parallel_passing=True,
    )
    with pytest.raises(ValueError, match="parallel_passing"):
        BrianRouteCore(cfg)
