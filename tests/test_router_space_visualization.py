import json

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.eval.router_space_visualization import make_router_space_visualization_from_payload
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def _model() -> BrianRouteCore:
    return BrianRouteCore(
        BrianRouteConfig(
            base=BaselineConfig(vocab_size=64, context_length=8, layers=4, d_model=32, n_heads=4),
            pre_blocks=1,
            route_pool_blocks=2,
            post_blocks=1,
            block_position_dim=8,
            max_route_steps=2,
        )
    )


def test_brian_forward_collects_router_space_payload() -> None:
    model = _model()
    input_ids = torch.randint(0, 64, (3, 8))

    output = model(input_ids, route_mode="free", collect_router_space=True)

    payload = output["router_space"]
    assert payload["out_action"] == 2
    assert len(payload["records"]) == 2
    assert payload["records"][0]["embedding"].shape == (3, 32)
    assert payload["records"][0]["selected_actions"].shape == (3,)


def test_router_space_visualization_reports_domination(tmp_path) -> None:
    model = _model()
    with torch.no_grad():
        for param in model.router.parameters():
            param.zero_()
        model.router.net[-1].bias[1] = 10.0
    embedding = torch.zeros(4, 32)
    raw_logits = torch.tensor([[0.0, 10.0, 0.0]]).expand(4, -1)
    probs = torch.softmax(raw_logits, dim=-1)
    payload = {
        "out_action": 2,
        "records": [
            {
                "step": 0,
                "embedding": embedding,
                "raw_logits": raw_logits,
                "effective_logits": raw_logits,
                "probs": probs,
                "selected_actions": torch.ones(4, dtype=torch.long),
            }
        ],
    }

    html_path = make_router_space_visualization_from_payload(
        payload,
        model,
        output_path=tmp_path / "router_space.html",
        step=7,
    )

    sidecar = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["overall_status"] == "warn"
    assert sidecar["metrics"]["selected_domination_fraction"] == 1.0
    assert sidecar["metrics"]["raw_top_domination_fraction"] == 1.0
    assert sidecar["checks"]["not_single_selected_action"] is False
