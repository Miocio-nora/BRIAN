from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def location_loss(location_distances: list["torch.Tensor"]) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for location loss.")
    if not location_distances:
        return torch.tensor(0.0)
    return torch.stack(location_distances).mean()
