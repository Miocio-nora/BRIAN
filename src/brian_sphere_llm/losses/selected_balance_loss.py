from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def selected_block_balance_loss(
    route_probs: list["torch.Tensor"],
    selected_actions: list["torch.Tensor"],
    num_internal_blocks: int,
) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for selected balance loss.")
    if not route_probs or not selected_actions:
        return torch.tensor(0.0)

    steps = min(len(route_probs), len(selected_actions))
    probs = torch.stack([item[:, :num_internal_blocks] for item in route_probs[:steps]], dim=0)
    selected = torch.stack(selected_actions[:steps], dim=0).clamp(min=0, max=num_internal_blocks)
    internal = selected < num_internal_blocks
    if not bool(internal.any().detach().cpu()):
        return probs.sum() * 0.0

    hard = F.one_hot(selected.clamp(max=num_internal_blocks - 1), num_classes=num_internal_blocks).to(probs.dtype)
    hard = hard * internal.unsqueeze(-1).to(probs.dtype)
    straight_through = hard - probs.detach() + probs
    load = straight_through.mean(dim=(0, 1))
    target = torch.full_like(load, load.sum().detach() / float(num_internal_blocks))
    return (load - target).pow(2).mean()
