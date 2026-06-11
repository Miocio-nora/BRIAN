from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def route_cost_loss(route_probs: list["torch.Tensor"], num_internal_blocks: int) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for cost loss.")
    if not route_probs:
        return torch.tensor(0.0)
    internal_mass = [probs[:, :num_internal_blocks].sum(dim=-1).mean() for probs in route_probs]
    return torch.stack(internal_mass).sum()
