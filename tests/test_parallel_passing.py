import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.routing.parallel_passing import prune_branches, score_branch_actions


def test_branch_scoring_and_pruning() -> None:
    route_log_probs = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.5]]), dim=-1)
    actions = torch.tensor([[0, 1, 2]])
    scored = score_branch_actions(
        parent_score=torch.tensor([0.0]),
        route_log_probs=route_log_probs,
        actions=actions,
        out_action=2,
        cost_penalty=0.2,
    )
    pruned = prune_branches(scored.actions, scored.scores, beam_size=2)
    assert pruned.actions.shape == (1, 2)
    assert pruned.scores.shape == (1, 2)
    assert pruned.scores[0, 0] >= pruned.scores[0, 1]
