import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.model.baseline import BaselineConfig


def test_brian_fixed_route_outputs_route_summary() -> None:
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
    output = model(input_ids, targets=input_ids, route_mode="fixed", pseudo_policy="sequential")
    assert output["logits"].shape == (2, 8, 64)
    assert "route_entropy" in output["routing_summary"]
    assert output["routing_summary"]["route_imitation_accuracy"] == 1.0
