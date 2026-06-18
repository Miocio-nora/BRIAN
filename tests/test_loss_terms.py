import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.losses.balance_loss import block_balance_loss
from brian_sphere_llm.losses.coverage_floor_loss import block_coverage_floor_loss
from brian_sphere_llm.losses.cost_loss import route_cost_loss
from brian_sphere_llm.losses.exit_boundary_loss import exit_boundary_loss
from brian_sphere_llm.losses.location_loss import location_loss
from brian_sphere_llm.losses.route_loss import route_imitation_loss
from brian_sphere_llm.losses.selected_balance_loss import selected_block_balance_loss
from brian_sphere_llm.losses.transition_diversity_loss import transition_diversity_loss
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def test_loss_terms_are_scalar() -> None:
    logits = [torch.randn(2, 4)]
    targets = [torch.tensor([1, 2])]
    probs = [torch.softmax(logits[0], dim=-1)]
    assert route_imitation_loss(logits, targets).ndim == 0
    assert block_balance_loss(probs, num_internal_blocks=3).ndim == 0
    assert route_cost_loss(probs, num_internal_blocks=3).ndim == 0
    assert location_loss([torch.tensor(1.0)]).ndim == 0
    assert block_coverage_floor_loss(probs, [torch.tensor([1, 2])], num_internal_blocks=3, floor=0.05).ndim == 0
    assert selected_block_balance_loss(probs, [torch.tensor([1, 2])], num_internal_blocks=3).ndim == 0
    assert transition_diversity_loss(probs + probs, [torch.tensor([1, 2]), torch.tensor([2, 1])], 3).ndim == 0
    assert exit_boundary_loss(probs, num_internal_blocks=3, constraints={"max_route_steps": 1}).ndim == 0


def test_balance_and_cost_losses_ignore_out_action_probability() -> None:
    high_out = [torch.tensor([[0.25, 0.25, 0.50], [0.25, 0.25, 0.50]])]
    low_out = [torch.tensor([[0.25, 0.25, 0.00], [0.25, 0.25, 0.00]])]

    assert block_balance_loss(high_out, num_internal_blocks=2) == block_balance_loss(
        low_out,
        num_internal_blocks=2,
    )
    assert route_cost_loss(high_out, num_internal_blocks=2) == route_cost_loss(
        low_out,
        num_internal_blocks=2,
    )


def test_selected_balance_loss_penalizes_collapsed_selected_blocks() -> None:
    probs = [torch.full((4, 3), 1.0 / 3.0)]
    balanced = [torch.tensor([0, 1, 0, 1])]
    collapsed = [torch.tensor([0, 0, 0, 0])]

    assert selected_block_balance_loss(probs, balanced, num_internal_blocks=2) < selected_block_balance_loss(
        probs,
        collapsed,
        num_internal_blocks=2,
    )


def test_coverage_floor_loss_penalizes_dead_blocks() -> None:
    probs = [torch.full((4, 3), 1.0 / 3.0)]
    covered = [torch.tensor([0, 1, 0, 1])]
    dead = [torch.tensor([0, 0, 0, 0])]

    assert block_coverage_floor_loss(probs, covered, num_internal_blocks=2, floor=0.25) < block_coverage_floor_loss(
        probs,
        dead,
        num_internal_blocks=2,
        floor=0.25,
    )


def test_transition_diversity_loss_penalizes_single_transition() -> None:
    probs = [torch.full((2, 3), 1.0 / 3.0) for _ in range(3)]
    diverse = [torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 1])]
    collapsed = [torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0])]

    assert transition_diversity_loss(probs, diverse, num_internal_blocks=2) < transition_diversity_loss(
        probs,
        collapsed,
        num_internal_blocks=2,
    )


def test_exit_boundary_loss_prefers_late_out_probability() -> None:
    early_out = [torch.tensor([[0.05, 0.05, 0.90]]), torch.tensor([[0.05, 0.05, 0.90]])]
    late_out = [torch.tensor([[0.45, 0.45, 0.10]]), torch.tensor([[0.05, 0.05, 0.90]])]
    constraints = {"max_route_steps": 2, "min_exit_step": 2, "exit_ramp_start": 2}

    assert exit_boundary_loss(late_out, 2, constraints) < exit_boundary_loss(early_out, 2, constraints)


def _tiny_brian_route_core() -> BrianRouteCore:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
    )
    return BrianRouteCore(cfg)


def test_brian_forward_rejects_boolean_loss_weights() -> None:
    model = _tiny_brian_route_core()
    input_ids = torch.randint(0, 64, (2, 8))

    with pytest.raises(ValueError, match="loss_weights.route"):
        model(input_ids, targets=input_ids, route_mode="fixed", loss_weights={"route": True})


def test_brian_forward_rejects_non_mapping_loss_weights() -> None:
    model = _tiny_brian_route_core()
    input_ids = torch.randint(0, 64, (2, 8))

    with pytest.raises(ValueError, match="loss_weights must be a mapping"):
        model(
            input_ids,
            targets=input_ids,
            route_mode="fixed",
            loss_weights=[("route", 1.0)],  # type: ignore[arg-type]
        )
