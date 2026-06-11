import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.losses.balance_loss import block_balance_loss
from brian_sphere_llm.losses.cost_loss import route_cost_loss
from brian_sphere_llm.losses.location_loss import location_loss
from brian_sphere_llm.losses.route_loss import route_imitation_loss


def test_loss_terms_are_scalar() -> None:
    logits = [torch.randn(2, 4)]
    targets = [torch.tensor([1, 2])]
    probs = [torch.softmax(logits[0], dim=-1)]
    assert route_imitation_loss(logits, targets).ndim == 0
    assert block_balance_loss(probs, num_internal_blocks=3).ndim == 0
    assert route_cost_loss(probs, num_internal_blocks=3).ndim == 0
    assert location_loss([torch.tensor(1.0)]).ndim == 0
