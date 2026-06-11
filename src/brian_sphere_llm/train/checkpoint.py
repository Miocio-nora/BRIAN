from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import random
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    best_eval_loss: float | None = None,
    extra: Mapping[str, Any] | None = None,
    include_rng_state: bool = True,
) -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_eval_loss": best_eval_loss,
    }
    if extra:
        payload.update(dict(extra))
    if include_rng_state:
        payload["rng_state"] = capture_rng_state()
    torch.save(payload, path / "state.pt")


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    restore_rng_state: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    payload = _load_state(Path(path) / "state.pt")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng_state and "rng_state" in payload:
        restore_rng(payload["rng_state"])
    return payload


def save_rank_state(
    path: str | Path,
    *,
    rank: int,
    step: int,
    data_epoch: int,
    microbatch_in_epoch: int,
    best_eval_loss: float | None = None,
) -> Path:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state_path = path / _rank_state_name(rank)
    torch.save(
        {
            "rank": int(rank),
            "step": int(step),
            "data_epoch": int(data_epoch),
            "microbatch_in_epoch": int(microbatch_in_epoch),
            "best_eval_loss": best_eval_loss,
            "rng_state": capture_rng_state(),
        },
        state_path,
    )
    return state_path


def load_rank_state(path: str | Path, *, rank: int, restore_rng_state: bool = False) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    payload = _load_state(Path(path) / _rank_state_name(rank))
    if int(payload.get("rank", rank)) != int(rank):
        raise ValueError(f"Rank state rank mismatch: expected {rank}, found {payload.get('rank')}.")
    if restore_rng_state and "rng_state" in payload:
        restore_rng(payload["rng_state"])
    return payload


def rank_state_path(path: str | Path, *, rank: int) -> Path:
    return Path(path) / _rank_state_name(rank)


def capture_rng_state() -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    return {
        "python": random.getstate(),
        "numpy": _pack_numpy_state(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(_unpack_numpy_state(state["numpy"]))
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(cuda_state))


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        return torch.load(path, map_location="cpu")


def _rank_state_name(rank: int) -> str:
    return f"rank_state_{int(rank):05d}.pt"


def _pack_numpy_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    name, keys, pos, has_gauss, cached_gaussian = state
    return name, keys.tolist(), pos, has_gauss, cached_gaussian


def _unpack_numpy_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    name, keys, pos, has_gauss, cached_gaussian = state
    return name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian
