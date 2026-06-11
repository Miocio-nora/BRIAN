import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.routing.block_position import BlockPositionTable


def test_position_update_norm() -> None:
    table = BlockPositionTable(num_internal_blocks=3, position_dim=8)
    action = torch.tensor([0, 1, 3])
    pos = table.by_action(action)
    assert pos.shape == (3, 8)
    assert torch.allclose(pos.norm(dim=-1), torch.ones(3), atol=1e-5)
