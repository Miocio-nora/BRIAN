import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.train.stage_runner import build_model_from_config, train_mode_for_stage


def test_parallel_route_forward_reports_branch_metrics() -> None:
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
        parallel_passing=True,
        beam_size=2,
    )
    model = BrianRouteCore(cfg)
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="parallel", hard_exit=True)
    summary = output["routing_summary"]
    assert output["logits"].shape == (2, 8, 64)
    assert summary["parallel_branch_count_mean"] <= 2.0
    assert "parallel_score_margin_mean" in summary
    assert "global_attention_mass" in summary


def test_stage6_config_builds_parallel_model() -> None:
    model = build_model_from_config("configs/model/brian_tiny_parallel.yaml")
    assert model.config.parallel_passing is True
    assert model.config.beam_size == 2
    assert train_mode_for_stage("stage6_parallel_passing") == "parallel"
