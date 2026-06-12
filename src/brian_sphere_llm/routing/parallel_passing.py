from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class BranchScores:
    actions: "torch.Tensor"
    scores: "torch.Tensor"


def score_branch_actions(
    *,
    parent_score: "torch.Tensor",
    route_log_probs: "torch.Tensor",
    actions: "torch.Tensor",
    out_action: int,
    cost_penalty: float,
    branch_score_decay: float = 1.0,
) -> BranchScores:
    """Score candidate branch actions as latent beam-search transitions."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for parallel passing.")
    gathered = route_log_probs.gather(dim=-1, index=actions)
    internal = actions != out_action
    scores = (
        parent_score.unsqueeze(-1) * float(branch_score_decay)
        + gathered
        - float(cost_penalty) * internal.float()
    )
    return BranchScores(actions=actions, scores=scores)


def prune_branches(actions: "torch.Tensor", scores: "torch.Tensor", beam_size: int) -> BranchScores:
    """Keep the top scoring branch actions per sample."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for parallel passing.")
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    keep = min(beam_size, actions.size(-1))
    top_scores, indices = scores.topk(keep, dim=-1)
    top_actions = actions.gather(dim=-1, index=indices)
    return BranchScores(actions=top_actions, scores=top_scores)
