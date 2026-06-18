from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def block_coverage_floor_loss(
    route_probs: list["torch.Tensor"],
    selected_actions: list["torch.Tensor"],
    num_internal_blocks: int,
    *,
    floor: float,
) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for coverage floor loss.")
    if floor <= 0.0 or not route_probs or not selected_actions:
        reference = route_probs[0] if route_probs else None
        return reference.sum() * 0.0 if reference is not None else torch.tensor(0.0)

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
    floor_tensor = torch.full_like(load, float(floor))
    return (floor_tensor - load).clamp_min(0.0).sum()
