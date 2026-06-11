import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.eval.difficulty import summarize_difficulty_samples
from brian_sphere_llm.eval.difficulty_report import (
    causal_lm_sample_losses,
    output_probability_per_sample,
    route_steps_per_sample,
)


def test_causal_lm_sample_losses_shape_and_order() -> None:
    logits = torch.zeros(2, 4, 5)
    targets = torch.tensor([[0, 1, 2, 3], [0, 2, 2, 2]])
    logits[0, :-1, 1:] = 10.0
    losses = causal_lm_sample_losses(logits, targets)
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()


def test_route_steps_and_output_probability_per_sample() -> None:
    route_info = {
        "selected_actions": [
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 2, 2]),
            torch.tensor([2, 2, 2]),
        ],
        "exit_flags": [
            torch.tensor([False, False, True]),
            torch.tensor([True, False, True]),
            torch.tensor([True, True, True]),
        ],
        "route_probs": [
            torch.tensor([[0.8, 0.1, 0.1], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]]),
            torch.tensor([[0.1, 0.1, 0.8], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]]),
        ],
    }
    assert route_steps_per_sample(route_info, batch_size=3) == [2, 3, 1]
    assert route_steps_per_sample(route_info, batch_size=3, out_action=2) == [1, 1, 0]
    assert output_probability_per_sample(route_info, out_action=2, batch_size=3) == pytest.approx([0.45, 0.25, 0.75])


def test_summarize_difficulty_samples() -> None:
    summary = summarize_difficulty_samples(
        [
            {"baseline_cross_entropy": 1.0, "routed_cross_entropy": 1.2, "route_steps": 1},
            {"baseline_cross_entropy": 2.0, "routed_cross_entropy": 2.1, "route_steps": 2},
            {"baseline_cross_entropy": 3.0, "routed_cross_entropy": 3.2, "route_steps": 3},
        ]
    )
    assert summary["sample_count"] == 3
    assert summary["difficulty_step_correlation"] == pytest.approx(1.0)
