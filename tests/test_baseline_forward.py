import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM


def test_baseline_forward_shapes() -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=2, d_model=32, n_heads=4))
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids)
    assert output["logits"].shape == (2, 8, 64)
    assert output["loss"].ndim == 0


def test_baseline_activation_checkpointing_backward() -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=8, layers=2, d_model=32, n_heads=4))
    model.activation_checkpointing = True
    model.train()
    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids, targets=input_ids)

    output["loss"].backward()

    assert model.token_embedding.weight.grad is not None


def test_baseline_model_stats_preserve_config_name() -> None:
    config = BaselineConfig.from_dict(
        {
            "model_name": "baseline_unit",
            "vocab_size": 64,
            "context_length": 8,
            "layers": 2,
            "d_model": 32,
            "n_heads": 4,
        }
    )
    model = BaselineLM(config)

    assert model.model_stats()["model_name"] == "baseline_unit"


def test_baseline_config_rejects_boolean_numeric_fields() -> None:
    data = {
        "model_name": "baseline_unit",
        "vocab_size": 64,
        "context_length": 8,
        "layers": True,
        "d_model": 32,
        "n_heads": 4,
    }

    with pytest.raises(ValueError, match="layers"):
        BaselineConfig.from_dict(data)


def test_baseline_config_rejects_boolean_dropout() -> None:
    data = {
        "model_name": "baseline_unit",
        "vocab_size": 64,
        "context_length": 8,
        "layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "dropout": False,
    }

    with pytest.raises(ValueError, match="dropout"):
        BaselineConfig.from_dict(data)
