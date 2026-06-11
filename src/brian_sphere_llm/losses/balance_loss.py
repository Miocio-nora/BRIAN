from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def block_balance_loss(route_probs: list["torch.Tensor"], num_internal_blocks: int) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for balance loss.")
    if not route_probs:
        return torch.tensor(0.0)
    internal = torch.stack([probs[:, :num_internal_blocks] for probs in route_probs], dim=0)
    load = internal.mean(dim=(0, 1))
    target = torch.full_like(load, 1.0 / num_internal_blocks)
    return (load - target).pow(2).mean()
