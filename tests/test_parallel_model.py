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
    assert model.config.branch_score_decay == 0.99
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="parallel", hard_exit=True)
    summary = output["routing_summary"]
    assert output["logits"].shape == (2, 8, 64)
    assert summary["parallel_branch_count_mean"] <= 2.0
    assert "parallel_score_margin_mean" in summary
    assert "parallel_delta_cache_slots_mean" in summary
    assert summary["parallel_delta_cache_slots_max"] <= 2.0
    assert "global_attention_mass" in summary
    assert "global_sink_attention_mass" in summary
    assert "global_window_attention_mass" in summary


def test_stage6_config_builds_parallel_model() -> None:
    model = build_model_from_config("configs/model/brian_tiny_parallel.yaml")
    assert model.config.parallel_passing is True
    assert model.config.beam_size == 2
    assert model.config.branch_score_decay == 0.99
    assert train_mode_for_stage("stage6_parallel_passing") == "parallel"


def test_parallel_ablation_configs_build_models() -> None:
    beam4 = build_model_from_config("configs/model/brian_tiny_parallel_beam4.yaml")
    no_cost = build_model_from_config("configs/model/brian_tiny_parallel_no_branch_cost.yaml")
    top1_exit = build_model_from_config("configs/model/brian_tiny_parallel_top1_exit.yaml")
    any_topk_exit = build_model_from_config("configs/model/brian_tiny_parallel_any_topk_exit.yaml")
    assert beam4.config.parallel_passing is True
    assert beam4.config.beam_size == 4
    assert beam4.config.branch_cost == 0.01
    assert beam4.config.branch_score_decay == 0.99
    assert no_cost.config.parallel_passing is True
    assert no_cost.config.beam_size == 2
    assert no_cost.config.branch_cost == 0.0
    assert top1_exit.config.parallel_exit_policy == "top1"
    assert any_topk_exit.config.parallel_exit_policy == "any_topk"


def test_parallel_top1_exit_policy_stops_when_out_is_top1() -> None:
    cfg = _parallel_cfg("top1")
    model = BrianRouteCore(cfg)
    _set_router_bias(model, {0: 9.0, cfg.route_pool_blocks: 10.0})
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="parallel", hard_exit=True)
    assert len(output["route_info"]["selected_actions"]) == 1
    assert output["routing_summary"]["first_exit_step_histogram"] == {"1": 2}


def test_parallel_any_topk_exit_policy_stops_when_out_is_in_topk() -> None:
    any_topk_model = BrianRouteCore(_parallel_cfg("any_topk"))
    top1_model = BrianRouteCore(_parallel_cfg("top1"))
    for model in (any_topk_model, top1_model):
        _set_router_bias(model, {0: 10.0, 1: 0.0, model.config.route_pool_blocks: 9.0})
    input_ids = torch.randint(0, 64, (2, 8))
    any_topk = any_topk_model(input_ids, targets=input_ids, route_mode="parallel", hard_exit=True)
    top1 = top1_model(input_ids, targets=input_ids, route_mode="parallel", hard_exit=True)
    assert len(any_topk["route_info"]["selected_actions"]) == 1
    assert any_topk["routing_summary"]["first_exit_step_histogram"] == {"1": 2}
    assert len(top1["route_info"]["selected_actions"]) == top1_model.config.max_route_steps
    assert top1["routing_summary"]["first_exit_step_histogram"] == {"0": 2}


def _parallel_cfg(policy: str) -> BrianRouteConfig:
    return BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=3,
        top_k=2,
        hard_exit=True,
        parallel_passing=True,
        beam_size=2,
        branch_cost=0.01,
        parallel_exit_policy=policy,
    )


def _set_router_bias(model: BrianRouteCore, bias_by_action: dict[int, float]) -> None:
    with torch.no_grad():
        for param in model.router.parameters():
            param.zero_()
        for action, value in bias_by_action.items():
            model.router.net[-1].bias[action] = value
