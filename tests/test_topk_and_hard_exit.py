import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def test_top2_weighted_fusion_is_reported() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        top_k=2,
    )
    model = BrianRouteCore(cfg)
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free")
    assert output["logits"].shape == (2, 8, 64)
    assert "topk_block_histogram" in output["routing_summary"]
    assert "weighted_fusion_ratio" in output["routing_summary"]


def test_later_top_k_enables_weighted_fusion_after_first_step() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        top_k=1,
        later_top_k=2,
    )
    model = BrianRouteCore(cfg)
    with torch.no_grad():
        for param in model.router.parameters():
            param.zero_()
        model.router.net[-1].bias[0] = 10.0
        model.router.net[-1].bias[1] = 9.0
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free")

    assert output["route_info"]["topk_actions"][0].shape[-1] == 1
    assert output["route_info"]["topk_actions"][1].shape[-1] == 2
    assert output["routing_summary"]["weighted_fusion_ratio"] == 0.5
    assert output["routing_summary"]["topk_block_histogram"]["1"] == 2


def test_hard_exit_stops_when_router_prefers_out() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=4,
        hard_exit=True,
    )
    model = BrianRouteCore(cfg)
    with torch.no_grad():
        for param in model.router.parameters():
            param.zero_()
        model.router.net[-1].bias[cfg.route_pool_blocks] = 10.0
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free", hard_exit=True)
    assert len(output["route_info"]["selected_actions"]) == 1
    assert output["routing_summary"]["first_exit_step_histogram"] == {"1": 2}
