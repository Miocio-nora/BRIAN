from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=1, d_model=16, n_heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "ckpt", model=model, optimizer=optimizer, step=3, best_eval_loss=1.2)
    payload = load_checkpoint(tmp_path / "ckpt", model=model, optimizer=optimizer)
    assert payload["step"] == 3
    assert payload["best_eval_loss"] == 1.2
