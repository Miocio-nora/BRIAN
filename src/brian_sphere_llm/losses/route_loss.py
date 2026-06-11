from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def route_imitation_loss(route_logits: list["torch.Tensor"], route_targets: list["torch.Tensor"]) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for route loss.")
    if not route_logits or not route_targets:
        return torch.tensor(0.0)
    losses = []
    for logits, target in zip(route_logits, route_targets):
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()
