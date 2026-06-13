from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    F = None


def exit_boundary_loss(
    route_probs: list["torch.Tensor"],
    num_internal_blocks: int,
    constraints: Mapping[str, Any] | None = None,
) -> "torch.Tensor":
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for exit boundary loss.")
    if not route_probs:
        return torch.tensor(0.0)
    constraints = constraints or {}
    max_steps = _int_value(constraints.get("max_route_steps", len(route_probs)), "routing.constraints.max_route_steps")
    min_exit_step = _int_value(constraints.get("min_exit_step", 1), "routing.constraints.min_exit_step")
    ramp_start = _int_value(
        constraints.get("exit_ramp_start", max(min_exit_step, max_steps)),
        "routing.constraints.exit_ramp_start",
    )
    losses = []
    for index, probs in enumerate(route_probs):
        step = index + 1
        p_out = probs[:, num_internal_blocks].float().clamp(min=1e-6, max=1.0 - 1e-6)
        target = torch.full_like(p_out, _exit_target(step, min_exit_step, ramp_start, max_steps))
        losses.append(-(target * p_out.log() + (1.0 - target) * (1.0 - p_out).log()).mean())
    return torch.stack(losses).mean()


def _exit_target(step: int, min_exit_step: int, ramp_start: int, max_steps: int) -> float:
    if step < min_exit_step:
        return 0.0
    if step >= max_steps:
        return 1.0
    if step < ramp_start:
        return 0.0
    denom = max(1, max_steps - ramp_start)
    return min(1.0, max(0.0, float(step - ramp_start + 1) / float(denom + 1)))


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value
