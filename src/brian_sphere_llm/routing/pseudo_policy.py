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
    actions: list[torch.Tensor] = []
    out = torch.full((batch_size,), num_internal_blocks, dtype=torch.long, device=device)
    for step in range(max_route_steps):
        sequential = torch.full(
            (batch_size,),
            min(step, max(0, num_internal_blocks - 1)),
            dtype=torch.long,
            device=device,
        )
        easy_action = torch.full(
            (batch_size,),
            min(2, max(0, num_internal_blocks - 1)),
            dtype=torch.long,
            device=device,
        )
        hard_action = torch.full(
            (batch_size,),
            min(max(0, step - 1), max(0, num_internal_blocks - 1)),
            dtype=torch.long,
            device=device,
        )
        action = sequential
        easy = difficulty == 0
        medium = difficulty == 1
        hard = difficulty == 2
        if num_internal_blocks > 2:
            action = torch.where(easy & (step == 1), easy_action, action)
            action = torch.where(easy & (step >= 2), out, action)
        else:
            action = torch.where(easy & (step >= 1), out, action)
        action = torch.where(hard, hard_action, action)
        action = torch.where((medium | hard) & (step >= max_route_steps - 1), out, action)
        actions.append(action)
    if not torch.all(actions[-1] == out):
        actions.append(out)
    return actions


def balanced_coverage_actions(
    num_internal_blocks: int,
    max_route_steps: int,
    batch_size: int,
    device: "torch.device",
) -> list["torch.Tensor"]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for pseudo policies.")
    batch_offsets = torch.arange(batch_size, device=device, dtype=torch.long) % num_internal_blocks
    return [(batch_offsets + step) % num_internal_blocks for step in range(max_route_steps)]


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
    if policy == "balanced_coverage":
        return balanced_coverage_actions(num_internal_blocks, max_route_steps, batch_size, device)
    raise ValueError(f"Unknown pseudo policy: {policy}")
