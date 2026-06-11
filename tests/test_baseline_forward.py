import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM


def test_baseline_forward_shapes() -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=2, d_model=32, n_heads=4))
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids)
    assert output["logits"].shape == (2, 8, 64)
    assert output["loss"].ndim == 0
