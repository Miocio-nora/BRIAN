import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.memory import CanonicalGlobalCache
from brian_sphere_llm.memory.read_adapter import GlobalReadAdapter
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def test_global_cache_keeps_sink_and_window() -> None:
    cache = CanonicalGlobalCache(sink_slots=1, window_slots=2)
    state = cache.empty(batch_size=1, code_dim=2, device=torch.device("cpu"), dtype=torch.float32)
    for value in range(5):
        code = torch.tensor([[float(value), float(value)]])
        state = cache.write(state, code)
    assert state.codes.shape == (1, 3, 2)
    assert state.codes[0, :, 0].tolist() == [0.0, 3.0, 4.0]


def test_brian_global_kv_reports_metrics() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        top_k=2,
        hard_exit=True,
        global_kv=True,
        global_code_dim=8,
        global_sink_slots=1,
        global_window_slots=2,
    )
    model = BrianRouteCore(cfg)
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="scheduled", router_probability=1.0, hard_exit=True)
    summary = output["routing_summary"]
    assert "global_attention_mass" in summary
    assert "global_sink_attention_mass" in summary
    assert "global_window_attention_mass" in summary
    assert "global_read_gate_mean" in summary
    assert "global_cache_slots_mean" in summary
    assert 0.0 <= summary["global_sink_attention_mass"] <= 1.0
    assert 0.0 <= summary["global_window_attention_mass"] <= 1.0


def test_global_read_adapter_reports_sink_and_window_attention() -> None:
    adapter = GlobalReadAdapter(d_model=4, code_dim=2)
    hidden = torch.ones(2, 3, 4)
    codes = torch.randn(2, 4, 2)
    _, metrics = adapter(hidden, codes, sink_slots=1)

    assert metrics["global_attention_mass"].item() == pytest.approx(1.0)
    assert metrics["global_sink_attention_mass"].item() >= 0.0
    assert metrics["global_window_attention_mass"].item() >= 0.0
    assert (
        metrics["global_sink_attention_mass"].item() + metrics["global_window_attention_mass"].item()
    ) == pytest.approx(metrics["global_attention_mass"].item())


def test_global_read_adapter_reports_zero_sink_for_no_sink_cache() -> None:
    adapter = GlobalReadAdapter(d_model=4, code_dim=2)
    hidden = torch.ones(1, 3, 4)
    codes = torch.randn(1, 2, 2)
    _, metrics = adapter(hidden, codes, sink_slots=0)

    assert metrics["global_sink_attention_mass"].item() == pytest.approx(0.0)
    assert metrics["global_window_attention_mass"].item() == pytest.approx(1.0)
