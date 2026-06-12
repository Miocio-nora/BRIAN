from __future__ import annotations

import os
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def world_size() -> int:
    return _int_env("WORLD_SIZE", default=1, minimum=1)


def rank() -> int:
    return _int_env("RANK", default=0, minimum=0)


def local_rank() -> int:
    return _int_env("LOCAL_RANK", default=0, minimum=0)


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    return rank() == 0


def init_distributed(device: Any | None = None) -> bool:
    if not is_distributed():
        return False
    if torch is None or not torch.distributed.is_available():
        raise ModuleNotFoundError("PyTorch distributed support is required for WORLD_SIZE > 1.")
    if torch.distributed.is_initialized():
        return True
    device_id = None
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.set_device(local_rank())
        device_id = torch.device("cuda", local_rank())
    torch.distributed.init_process_group(
        backend=_backend_for_device(device),
        init_method="env://",
        device_id=device_id,
    )
    return True


def is_initialized() -> bool:
    return bool(torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized())


def barrier() -> None:
    if is_initialized():
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[local_rank()])
            return
        torch.distributed.barrier()


def destroy_distributed() -> None:
    if is_initialized():
        torch.distributed.destroy_process_group()


def mean_scalar(value: float, device: Any | None = None) -> float:
    if not is_initialized():
        return float(value)
    tensor_device = device if device is not None else torch.device("cpu")
    tensor = torch.tensor(float(value), dtype=torch.float64, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return float(tensor.item())


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _backend_for_device(device: Any | None) -> str:
    if device is not None and getattr(device, "type", None) == "cuda":
        return "nccl"
    return "gloo"


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value
