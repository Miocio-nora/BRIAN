import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig, BaselineLM
from brian_sphere_llm.train.trainer import evaluate


def test_evaluate_reports_inference_timing_metrics() -> None:
    model = BaselineLM(BaselineConfig(vocab_size=64, context_length=4, layers=1, d_model=16, n_heads=4))
    val_loader = [
        torch.randint(0, 64, (2, 4)),
        torch.randint(0, 64, (2, 4)),
    ]

    row = evaluate(
        model,
        val_loader,
        config={"stage": "stage0_baseline"},
        device=torch.device("cpu"),
        route_mode="baseline",
        global_step=1,
    )

    assert row["eval_batch_count"] == 2
    assert row["eval_token_count"] == 16
    assert row["inference_time_seconds"] > 0.0
    assert row["inference_tokens_per_second"] > 0.0
    assert row["inference_latency_ms_per_token"] > 0.0
    assert row["validation_loss"] >= 0.0
