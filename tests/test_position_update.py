import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.routing.block_position import BlockPositionTable


def test_position_update_norm() -> None:
    table = BlockPositionTable(num_internal_blocks=3, position_dim=8)
    action = torch.tensor([0, 1, 3])
    pos = table.by_action(action)
    assert pos.shape == (3, 8)
    assert torch.allclose(pos.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_position_none_mode_returns_zero_state() -> None:
    table = BlockPositionTable(num_internal_blocks=3, position_dim=8, mode="none")
    action = torch.tensor([0, 1, 3])
    pos = table.by_action(action)
    probs = torch.full((3, 4), 0.25)
    assert torch.count_nonzero(table.initial(batch_size=2, device=torch.device("cpu"))) == 0
    assert torch.count_nonzero(pos) == 0
    assert table.location_distance(pos, probs).item() == 0.0


def test_position_circular_places_out_near_initial_position() -> None:
    table = BlockPositionTable(num_internal_blocks=3, position_dim=8, mode="circular")
    start = table.by_action(torch.tensor([0]))
    out = table.by_action(torch.tensor([3]))
    assert torch.allclose(start, out, atol=1e-5)
