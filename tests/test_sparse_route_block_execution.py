from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore


def _paired_models() -> tuple[BrianRouteCore, BrianRouteCore]:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=6, layers=5, d_model=24, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
    )
    full = BrianRouteCore(cfg)
    sparse = BrianRouteCore(replace(cfg, route_block_execution="sparse"))
    sparse.load_state_dict(full.state_dict())
    full.eval()
    sparse.eval()
    return full, sparse


def _position(model: BrianRouteCore, batch_size: int, seq_len: int) -> torch.Tensor:
    base = model.position_table.initial(batch_size, torch.device("cpu"))
    return base.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


def test_sparse_route_block_execution_matches_full_sequence_top1() -> None:
    torch.manual_seed(11)
    full, sparse = _paired_models()
    hidden = torch.randn(2, 6, 24)
    position = _position(full, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, full.out_action, 0, 1],
            [2, 0, full.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
    actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_sparse_route_block_execution_matches_full_sequence_weighted_fusion() -> None:
    torch.manual_seed(17)
    full, sparse = _paired_models()
    hidden = torch.randn(2, 6, 24)
    position = _position(full, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    top_actions = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 0], [0, full.out_action], [1, 0], [2, 1]],
            [[2, 1], [0, 2], [1, full.out_action], [2, 0], [0, 1], [1, 2]],
        ],
        dtype=torch.long,
    )
    top_weights = torch.tensor(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [1.0, 0.0], [0.8, 0.2], [0.55, 0.45]],
            [[0.65, 0.35], [0.9, 0.1], [1.0, 0.0], [0.75, 0.25], [0.6, 0.4], [0.5, 0.5]],
        ],
        dtype=hidden.dtype,
    )
    use_weighted_fusion = torch.tensor(
        [
            [True, False, True, True, False, True],
            [True, True, True, False, True, False],
        ],
        dtype=torch.bool,
    )

    expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
    actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_sparse_route_block_execution_supports_bf16_autocast() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA autocast coverage requires CUDA.")
    torch.manual_seed(23)
    _, sparse = _paired_models()
    sparse = sparse.cuda()
    hidden = torch.randn(2, 6, 24, device="cuda")
    position = _position(sparse, 2, 6).cuda()
    selected = torch.tensor(
        [
            [0, 1, 2, sparse.out_action, 0, 1],
            [2, 0, sparse.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
        device="cuda",
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1, device="cuda")
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert actual.shape == hidden.shape
    assert torch.isfinite(actual).all()


def test_sparse_route_block_execution_config_stats_and_validation() -> None:
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 24, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "route_block_execution": "sparse",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.route_block_execution == "sparse"
    assert model.model_stats()["route_block_execution"] == "sparse"
    with pytest.raises(ValueError, match="route_block_execution"):
        BrianRouteConfig.from_dict(
            {
                "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 24, "n_heads": 4},
                "pre_blocks": 1,
                "route_pool_blocks": 2,
                "post_blocks": 1,
                "block_position_dim": 8,
                "max_route_steps": 2,
                "route_block_execution": "fast",
            }
        )
