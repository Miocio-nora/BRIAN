import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.eval.difficulty import summarize_difficulty_samples
from brian_sphere_llm.eval.difficulty_report import (
    _assign_difficulty_bins,
    _forward_routed_for_eval,
    _mapping_config,
    _summarize_baseline_difficulty_samples,
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


def test_summarize_difficulty_samples_rejects_boolean_numeric_metrics() -> None:
    summary = summarize_difficulty_samples(
        [
            {"baseline_cross_entropy": True, "routed_cross_entropy": 1.2, "route_steps": 1},
            {"baseline_cross_entropy": 2.0, "routed_cross_entropy": False, "route_steps": 2},
            {"baseline_cross_entropy": 3.0, "routed_cross_entropy": 3.2, "route_steps": True},
        ]
    )

    assert summary["sample_count"] == 0
    assert summary["mean_baseline_cross_entropy"] is None
    assert summary["mean_routed_cross_entropy"] is None
    assert summary["mean_route_steps"] is None
    assert summary["difficulty_step_correlation"] is None


def test_difficulty_eval_forward_parses_string_false_hard_exit() -> None:
    class CaptureModel:
        hard_exit = None

        def __call__(self, *args, **kwargs):
            self.hard_exit = kwargs["hard_exit"]
            return {"logits": torch.zeros(1, 4, 8)}

    model = CaptureModel()
    batch = torch.randint(0, 8, (1, 4))

    _forward_routed_for_eval(
        model,
        batch,
        config={"stage": "stage4_output_action", "routing": {"hard_exit": "false"}},
        route_mode="free",
        global_step=1,
    )

    assert model.hard_exit is False


def test_difficulty_eval_forward_rejects_non_mapping_routing_config() -> None:
    with pytest.raises(ValueError, match="routing"):
        _mapping_config({"routing": True}, "routing")


def test_baseline_difficulty_bins_are_ce_ordered() -> None:
    samples = [
        {"sample_id": 0, "baseline_cross_entropy": 3.0},
        {"sample_id": 1, "baseline_cross_entropy": 1.0},
        {"sample_id": 2, "baseline_cross_entropy": 2.0},
        {"sample_id": 3, "baseline_cross_entropy": 6.0},
        {"sample_id": 4, "baseline_cross_entropy": 5.0},
        {"sample_id": 5, "baseline_cross_entropy": 4.0},
    ]
    bins = ["easy", "medium", "hard"]

    _assign_difficulty_bins(samples, bins)
    summary = _summarize_baseline_difficulty_samples(samples, bins)

    assert samples[1]["difficulty_bin"] == "easy"
    assert samples[2]["difficulty_bin"] == "easy"
    assert samples[0]["difficulty_bin"] == "medium"
    assert samples[5]["difficulty_bin"] == "medium"
    assert samples[4]["difficulty_bin"] == "hard"
    assert samples[3]["difficulty_bin"] == "hard"
    assert summary["sample_count"] == 6
    assert summary["difficulty_bin_count"] == 3
    assert summary["by_difficulty"]["easy"]["mean_baseline_cross_entropy"] == pytest.approx(1.5)
    assert summary["by_difficulty"]["medium"]["mean_baseline_cross_entropy"] == pytest.approx(3.5)
    assert summary["by_difficulty"]["hard"]["mean_baseline_cross_entropy"] == pytest.approx(5.5)
