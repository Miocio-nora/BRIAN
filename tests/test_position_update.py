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


def test_independent_input_position_is_trainable_and_not_an_action() -> None:
    table = BlockPositionTable(
        num_internal_blocks=3,
        position_dim=8,
        independent_input_position=True,
    )
    assert table.input_position is not None
    assert table.input_position.requires_grad
    assert table.embeddings.shape == (4, 8)

    with torch.no_grad():
        table.input_position.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        table.embeddings[0].copy_(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    start = table.initial(batch_size=2, device=torch.device("cpu"))
    block_zero = table.by_action(torch.tensor([0, 0]))
    assert start.requires_grad
    assert not torch.allclose(start, block_zero)
    assert torch.allclose(start.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_position_circular_places_out_near_initial_position() -> None:
    table = BlockPositionTable(num_internal_blocks=3, position_dim=8, mode="circular")
    start = table.by_action(torch.tensor([0]))
    out = table.by_action(torch.tensor([3]))
    assert torch.allclose(start, out, atol=1e-5)
