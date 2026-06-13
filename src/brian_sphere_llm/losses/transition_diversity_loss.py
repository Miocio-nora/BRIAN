from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def transition_diversity_loss(
    route_probs: list["torch.Tensor"],
    selected_actions: list["torch.Tensor"],
    num_internal_blocks: int,
) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for transition diversity loss.")
    if len(route_probs) < 2 or len(selected_actions) < 2 or num_internal_blocks <= 1:
        return torch.tensor(0.0)

    steps = min(len(route_probs), len(selected_actions))
    probs = torch.stack([item[:, :num_internal_blocks] for item in route_probs[:steps]], dim=0)
    selected = torch.stack(selected_actions[:steps], dim=0).clamp(min=0, max=num_internal_blocks)
    source = selected[:-1]
    target = selected[1:]
    valid = (source < num_internal_blocks) & (target < num_internal_blocks)
    if not bool(valid.any().detach().cpu()):
        return probs.sum() * 0.0

    soft = probs[:-1].unsqueeze(-1) * probs[1:].unsqueeze(-2)
    hard_source = F.one_hot(source.clamp(max=num_internal_blocks - 1), num_classes=num_internal_blocks).to(probs.dtype)
    hard_target = F.one_hot(target.clamp(max=num_internal_blocks - 1), num_classes=num_internal_blocks).to(probs.dtype)
    hard = hard_source.unsqueeze(-1) * hard_target.unsqueeze(-2)
    hard = hard * valid.unsqueeze(-1).unsqueeze(-1).to(probs.dtype)
    straight_through = hard - soft.detach() + soft
    transition_load = straight_through.mean(dim=(0, 1))
    total = transition_load.sum().detach().clamp_min(1e-9)
    distribution = (transition_load / total).clamp_min(1e-9)
    entropy = -(distribution * distribution.log()).sum()
    max_entropy = math.log(float(num_internal_blocks * num_internal_blocks))
    return 1.0 - entropy / max_entropy
