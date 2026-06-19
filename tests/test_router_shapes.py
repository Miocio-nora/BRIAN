import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.routing.router import LatentRouter


def test_router_shape() -> None:
    router = LatentRouter(d_model=16, position_dim=4, num_actions=3)
    hidden = torch.randn(2, 5, 16)
    position = torch.randn(2, 4)
    logits = router(hidden, position)
    assert logits.shape == (2, 3)


def test_router_embedding_and_expert_vectors_match_logits() -> None:
    router = LatentRouter(d_model=16, position_dim=4, num_actions=3)
    hidden = torch.randn(2, 5, 16)
    position = torch.randn(2, 4)

    embedding = router.embedding(hidden, position)
    logits = router.logits_from_embedding(embedding)

    assert embedding.shape == (2, 16)
    assert router.expert_vectors().shape == (3, 16)
    assert torch.allclose(logits, router(hidden, position))
