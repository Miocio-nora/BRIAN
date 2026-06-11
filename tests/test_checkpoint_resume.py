from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.checkpoint import (
    load_checkpoint,
    load_rank_state,
    rank_state_path,
    save_checkpoint,
    save_rank_state,
)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=1, d_model=16, n_heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "ckpt", model=model, optimizer=optimizer, step=3, best_eval_loss=1.2)
    payload = load_checkpoint(tmp_path / "ckpt", model=model, optimizer=optimizer)
    assert payload["step"] == 3
    assert payload["best_eval_loss"] == 1.2
    assert "rng_state" in payload


def test_checkpoint_restores_rng_state_and_extra_training_state(tmp_path: Path) -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=1, d_model=16, n_heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        optimizer=optimizer,
        step=3,
        best_eval_loss=1.2,
        extra={"data_epoch": 2, "microbatch_in_epoch": 5},
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(()))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    payload = load_checkpoint(tmp_path / "ckpt", model=model, optimizer=optimizer, restore_rng_state=True)

    assert payload["data_epoch"] == 2
    assert payload["microbatch_in_epoch"] == 5
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(())) == expected_torch


def test_rank_state_roundtrip_restores_rank_local_rng_and_loader_offset(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    path = save_rank_state(
        tmp_path / "ckpt",
        rank=3,
        step=7,
        data_epoch=2,
        microbatch_in_epoch=5,
        best_eval_loss=1.5,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(()))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    payload = load_rank_state(tmp_path / "ckpt", rank=3, restore_rng_state=True)

    assert path == rank_state_path(tmp_path / "ckpt", rank=3)
    assert payload["rank"] == 3
    assert payload["step"] == 7
    assert payload["data_epoch"] == 2
    assert payload["microbatch_in_epoch"] == 5
    assert payload["best_eval_loss"] == 1.5
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(())) == expected_torch
