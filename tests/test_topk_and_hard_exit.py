import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def _set_router_bias(model: BrianRouteCore, bias_by_action: dict[int, float]) -> None:
    with torch.no_grad():
        for param in model.router.parameters():
            param.zero_()
        for action, value in bias_by_action.items():
            model.router.net[-1].bias[action] = value


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


def test_free_route_does_not_record_pseudo_targets() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
    )
    model = BrianRouteCore(cfg)
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free")

    assert output["route_info"]["route_targets"] == []
    assert "route_imitation_accuracy" not in output["routing_summary"]


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
    _set_router_bias(model, {0: 10.0, 1: 9.0})
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free")

    assert output["route_info"]["topk_actions"][0].shape[-1] == 1
    assert output["route_info"]["topk_actions"][1].shape[-1] == 2
    assert output["routing_summary"]["weighted_fusion_ratio"] == 0.5
    assert output["routing_summary"]["topk_block_histogram"]["1"] == 2


def test_pseudo_route_targets_control_forward_and_supervise_out_even_when_router_prefers_out() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        hard_exit=False,
    )
    model = BrianRouteCore(cfg)
    _set_router_bias(model, {cfg.route_pool_blocks: 10.0, 0: 0.0, 1: 0.0})
    input_ids = torch.randint(0, 64, (2, 8))

    output = model(
        input_ids,
        targets=input_ids,
        route_mode="pseudo",
        pseudo_policy="sequential",
        hard_exit=False,
    )

    selected = [actions.tolist() for actions in output["route_info"]["selected_actions"]]
    targets = [actions.tolist() for actions in output["route_info"]["route_targets"]]
    assert selected == targets
    assert selected == [[0, 0], [1, 1], [2, 2]]
    assert output["routing_summary"]["first_exit_step_histogram"] == {"3": 2}


def test_mixed_skip_recur_targets_follow_content_not_batch_slot() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=3,
        hard_exit=False,
    )
    model = BrianRouteCore(cfg)
    low = torch.full((8,), 2, dtype=torch.long)
    medium = torch.full((8,), 24, dtype=torch.long)
    high = torch.full((8,), 48, dtype=torch.long)

    first = model(
        torch.stack([low, medium, high]),
        route_mode="pseudo",
        pseudo_policy="mixed_skip_recur",
    )
    second = model(
        torch.stack([high, low, medium]),
        route_mode="pseudo",
        pseudo_policy="mixed_skip_recur",
    )

    assert first["route_info"]["route_targets"][1].tolist() == [2, 1, 0]
    assert second["route_info"]["route_targets"][1].tolist() == [0, 2, 1]


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
    _set_router_bias(model, {cfg.route_pool_blocks: 10.0})
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="free", hard_exit=True)
    assert len(output["route_info"]["selected_actions"]) == 1
    assert output["routing_summary"]["first_exit_step_histogram"] == {"1": 2}
    assert output["routing_summary"]["forced_max_step_exit_count"] == 0
    assert output["routing_summary"]["forced_max_step_exit_fraction"] == 0.0
    assert output["routing_summary"]["max_route_steps"] == cfg.max_route_steps


def test_hard_exit_ignores_out_when_out_is_only_in_topk() -> None:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=3,
        top_k=2,
        hard_exit=True,
    )
    model = BrianRouteCore(cfg)
    _set_router_bias(model, {0: 10.0, cfg.route_pool_blocks: 9.0, 1: 0.0})
    input_ids = torch.randint(0, 64, (2, 8))

    output = model(input_ids, targets=input_ids, route_mode="free", hard_exit=True)

    assert len(output["route_info"]["selected_actions"]) == cfg.max_route_steps
    assert output["routing_summary"]["first_exit_step_histogram"] == {"0": 2}
    assert output["routing_summary"]["forced_max_step_exit_count"] == 2
    assert output["routing_summary"]["forced_max_step_exit_fraction"] == 1.0
    assert output["routing_summary"]["max_route_steps"] == cfg.max_route_steps
    for topk_actions in output["route_info"]["topk_actions"]:
        assert torch.all(topk_actions[:, 0] == 0)
        assert torch.all(topk_actions[:, 1] == cfg.route_pool_blocks)


def test_route_constraints_suppress_early_out_and_force_final_exit() -> None:
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
    _set_router_bias(model, {cfg.route_pool_blocks: 10.0, 0: 0.0, 1: 0.0})
    input_ids = torch.randint(0, 64, (2, 8))

    output = model(
        input_ids,
        targets=input_ids,
        route_mode="free",
        hard_exit=True,
        routing_constraints={
            "min_exit_step": 5,
            "early_exit_logit_penalty": 20.0,
            "force_final_exit": True,
        },
    )

    selected = output["route_info"]["selected_actions"]
    assert len(selected) == cfg.max_route_steps
    for actions in selected[:-1]:
        assert torch.all(actions != cfg.route_pool_blocks)
    assert torch.all(selected[-1] == cfg.route_pool_blocks)
    assert output["routing_summary"]["first_exit_step_histogram"] == {"4": 2}
