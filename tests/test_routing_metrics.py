import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.routing.metrics import block_load_entropy_from_counts, summarize_routes


def test_block_load_entropy_from_counts() -> None:
    collapsed, collapsed_norm = block_load_entropy_from_counts({0: 4, 1: 0}, num_internal_blocks=2)
    balanced, balanced_norm = block_load_entropy_from_counts({0: 2, 1: 2}, num_internal_blocks=2)
    assert collapsed == pytest.approx(0.0)
    assert collapsed_norm == pytest.approx(0.0)
    assert balanced == pytest.approx(0.693147, rel=1e-5)
    assert balanced_norm == pytest.approx(1.0)


def test_summarize_routes_reports_load_entropy_and_path_diversity() -> None:
    route_info = {
        "route_probs": [
            torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.9, 0.05]]),
            torch.tensor([[0.05, 0.9, 0.05], [0.05, 0.9, 0.05], [0.9, 0.05, 0.05]]),
        ],
        "selected_actions": [
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 1, 1]),
        ],
        "exit_flags": [
            torch.tensor([False, False, False]),
            torch.tensor([False, False, False]),
        ],
    }
    summary = summarize_routes(route_info, num_internal_blocks=2)
    assert summary["block_load_entropy"] > 0.0
    assert summary["block_load_entropy_normalized"] > 0.0
    assert summary["route_path_count"] == 2
    assert summary["route_path_diversity"] == pytest.approx(2 / 3)


def test_summarize_routes_reports_path_examples_and_position_trajectories() -> None:
    route_info = {
        "route_probs": [
            torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]),
            torch.tensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
        ],
        "selected_actions": [
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
        ],
        "position_norms": [
            torch.tensor(1.0),
            torch.tensor(0.5),
        ],
        "location_distance": [
            torch.tensor(0.25),
            torch.tensor(0.75),
        ],
    }
    summary = summarize_routes(route_info, num_internal_blocks=2)

    assert summary["route_path_examples"] == [
        {"sample_index": 0, "actions": [0, 1]},
        {"sample_index": 1, "actions": [1, 2]},
    ]
    assert summary["position_norm_trajectory"] == [1.0, 0.5]
    assert summary["position_norm_mean"] == pytest.approx(0.75)
    assert summary["location_distance_trajectory"] == [0.25, 0.75]
    assert summary["location_distance_mean"] == pytest.approx(0.5)


def test_summarize_routes_derives_global_read_ratios() -> None:
    route_info = {
        "global_read_gate": [
            torch.tensor(0.25),
            torch.tensor(0.75),
        ],
    }
    summary = summarize_routes(route_info, num_internal_blocks=2)

    assert summary["global_read_gate_mean"] == pytest.approx(0.5)
    assert summary["local_read_fraction_mean"] == pytest.approx(0.5)
    assert summary["global_to_local_read_ratio"] == pytest.approx(1.0)
    assert summary["local_to_global_read_ratio"] == pytest.approx(1.0)


def test_summarize_routes_reports_forced_max_step_exit_fallback() -> None:
    route_info = {
        "hard_exit_enabled": True,
        "max_route_steps": 3,
        "route_probs": [
            torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]),
            torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]),
            torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]),
        ],
        "selected_actions": [
            torch.tensor([0, 0]),
            torch.tensor([0, 2]),
            torch.tensor([0, 2]),
        ],
        "exit_flags": [
            torch.tensor([False, False]),
            torch.tensor([False, True]),
            torch.tensor([False, True]),
        ],
    }
    summary = summarize_routes(route_info, num_internal_blocks=2)

    assert summary["first_exit_step_histogram"] == {"0": 1, "2": 1}
    assert summary["forced_max_step_exit_count"] == 1
    assert summary["forced_max_step_exit_fraction"] == 0.5
    assert summary["max_route_steps"] == 3
