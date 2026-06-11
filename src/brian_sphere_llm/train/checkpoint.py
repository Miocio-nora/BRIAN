from __future__ import annotations

from pathlib import Path
from typing import Any

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
) -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_eval_loss": best_eval_loss,
        },
        path / "state.pt",
    )


def load_checkpoint(path: str | Path, *, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for checkpointing.")
    payload = torch.load(Path(path) / "state.pt", map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
