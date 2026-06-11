import pytest

from brian_sphere_llm.routing.schedule import scheduled_value


def test_scheduled_value_uses_step_boundaries_and_default() -> None:
    schedule = [
        {"max_step": 2, "router_probability": 0.1},
        {"max_step": 4, "router_probability": 0.5},
    ]

    assert scheduled_value(schedule, 1, "router_probability", 0.0) == 0.1
    assert scheduled_value(schedule, 3, "router_probability", 0.0) == 0.5
    assert scheduled_value(schedule, 5, "router_probability", 0.0) == 0.5
    assert scheduled_value([], 5, "router_probability", 0.25) == 0.25
    assert scheduled_value(schedule, 1, "lambda_route", 1.0) == 1.0


def test_scheduled_value_rejects_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="max_step"):
        scheduled_value([{"max_step": True, "router_probability": 0.1}], 1, "router_probability", 0.0)
    with pytest.raises(ValueError, match="router_probability"):
        scheduled_value([{"max_step": 1, "router_probability": False}], 1, "router_probability", 0.0)
    with pytest.raises(ValueError, match="router_probability"):
        scheduled_value([], 1, "router_probability", True)


def test_scheduled_value_rejects_nonfinite_fields() -> None:
    with pytest.raises(ValueError, match="router_probability"):
        scheduled_value([{"max_step": 1, "router_probability": float("nan")}], 1, "router_probability", 0.0)
    with pytest.raises(ValueError, match="max_step"):
        scheduled_value([{"max_step": float("inf"), "router_probability": 0.1}], 1, "router_probability", 0.0)


def test_scheduled_value_rejects_non_mapping_items() -> None:
    with pytest.raises(ValueError, match="mappings"):
        scheduled_value([["not", "a", "mapping"]], 1, "router_probability", 0.0)
