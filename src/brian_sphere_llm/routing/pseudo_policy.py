from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def sequential_actions(num_internal_blocks: int, batch_size: int, device: "torch.device") -> list["torch.Tensor"]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for pseudo policies.")
    actions = [
        torch.full((batch_size,), index, dtype=torch.long, device=device)
        for index in range(num_internal_blocks)
    ]
    actions.append(torch.full((batch_size,), num_internal_blocks, dtype=torch.long, device=device))
    return actions


def mixed_skip_recur_actions(
    num_internal_blocks: int,
    max_route_steps: int,
    batch_size: int,
    device: "torch.device",
    difficulty: "torch.Tensor | None" = None,
) -> list["torch.Tensor"]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for pseudo policies.")
    if difficulty is None:
        difficulty = torch.arange(batch_size, device=device) % 3
    current = torch.zeros(batch_size, dtype=torch.long, device=device)
    actions: list[torch.Tensor] = []
    out = torch.full((batch_size,), num_internal_blocks, dtype=torch.long, device=device)
    for step in range(max_route_steps):
        action = current.clone()
        easy = difficulty == 0
        medium = difficulty == 1
        hard = difficulty == 2
        if step >= 1:
            action = torch.where(easy, out, action)
        if step >= max_route_steps - 1:
            action = torch.where(medium | hard, out, action)
        actions.append(action)
        next_current = current + 1
        next_current = torch.where(easy, current + 2, next_current)
        next_current = torch.where(hard & (step % 2 == 0), current, next_current)
        current = torch.clamp(next_current, max=num_internal_blocks - 1)
    if not torch.all(actions[-1] == out):
        actions.append(out)
    return actions


def actions_for_policy(
    policy: str,
    *,
    num_internal_blocks: int,
    max_route_steps: int,
    batch_size: int,
    device: "torch.device",
    difficulty: "torch.Tensor | None" = None,
) -> list["torch.Tensor"]:
    if policy == "sequential":
        return sequential_actions(num_internal_blocks, batch_size, device)
    if policy == "mixed_skip_recur":
        return mixed_skip_recur_actions(num_internal_blocks, max_route_steps, batch_size, device, difficulty)
    raise ValueError(f"Unknown pseudo policy: {policy}")
