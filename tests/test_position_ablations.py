from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.utils.config import load_config


def _config(**overrides) -> BrianRouteConfig:
    data = {
        "base": {
            "vocab_size": 64,
            "context_length": 8,
            "layers": 4,
            "d_model": 32,
            "n_heads": 4,
        },
        "pre_blocks": 1,
        "route_pool_blocks": 2,
        "post_blocks": 1,
        "block_position_dim": 8,
        "max_route_steps": 3,
    }
    data.update(overrides)
    return BrianRouteConfig.from_dict(data)


def test_no_position_ablation_forward_uses_zero_position_state() -> None:
    model = BrianRouteCore(
        _config(
            model_name="brian_no_position_unit",
            block_position_mode="none",
            position_to_router=False,
            position_to_blocks=False,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="fixed")
    assert output["logits"].shape == (2, 8, 64)
    assert output["routing_summary"]["position_norm_mean"] == 0.0
    assert model.model_stats()["model_name"] == "brian_no_position_unit"
    assert model.model_stats()["block_position_mode"] == "none"
    assert model.model_stats()["parameter_count"] == BrianRouteCore(_config()).model_stats()["parameter_count"]


def test_position_router_only_ablation_keeps_state_but_masks_blocks() -> None:
    model = BrianRouteCore(_config(position_to_router=True, position_to_blocks=False))
    position = torch.randn(2, 8)
    assert torch.count_nonzero(model._router_position(position)) > 0
    assert torch.count_nonzero(model._block_position(position)) == 0


def test_direct_position_addition_uses_hidden_dim_position_state() -> None:
    model = BrianRouteCore(_config(block_position_dim=32, block_position_injection="direct_add"))
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids, route_mode="fixed")

    assert output["logits"].shape == (2, 8, 64)
    assert model.model_stats()["block_position_injection"] == "direct_add"


def test_direct_position_addition_requires_matching_hidden_dim() -> None:
    with pytest.raises(ValueError, match="direct_add position_dim must equal d_model"):
        BrianRouteCore(_config(block_position_injection="direct_add"))


def test_location_bias_penalizes_distant_action_logits() -> None:
    model = BrianRouteCore(_config(location_bias_weight=0.5))
    position = model.position_table.by_action(torch.tensor([0, 1]))
    logits = torch.zeros(2, model.config.route_pool_blocks + 1)
    biased = model._apply_location_bias(logits, position)
    expected = -0.5 * model.position_table.action_distances(position)

    assert torch.allclose(biased, expected)
    assert model.model_stats()["location_bias_weight"] == "0.5"


def test_position_ablation_configs_resolve() -> None:
    for path in [
        "configs/train/stage3_no_position_tiny_debug.yaml",
        "configs/train/stage3_position_router_only_tiny_debug.yaml",
        "configs/train/stage3_position_circular_tiny_debug.yaml",
        "configs/train/stage3_position_random_tiny_debug.yaml",
        "configs/train/stage3_position_no_location_bias_tiny_debug.yaml",
        "configs/train/stage3_position_no_location_loss_tiny_debug.yaml",
        "configs/train/stage3_position_direct_add_tiny_debug.yaml",
        "configs/train/stage3_position_separate_state_tiny_debug.yaml",
        "configs/train/ablation_p0_no_position.yaml",
        "configs/train/ablation_p1_position_random.yaml",
        "configs/train/ablation_p3_position_circular.yaml",
        "configs/train/ablation_p4_position_router_only.yaml",
        "configs/train/ablation_p5_position_router_and_blocks.yaml",
        "configs/train/ablation_p6_no_location_bias.yaml",
        "configs/train/ablation_p7_no_location_loss.yaml",
        "configs/train/ablation_p8_direct_position_add.yaml",
        "configs/train/ablation_p9_separate_position_state.yaml",
    ]:
        cfg = load_config(path)
        model_config_path = (Path(path).parent / cfg["model_config"]).resolve()
        model_cfg = load_config(model_config_path)
        assert cfg["stage"] == "stage3_scheduled_free_routing"
        assert model_cfg["architecture"] == "brian_route_core"
