from __future__ import annotations


def scheduled_value(schedule: list[dict], step: int, key: str, default: float) -> float:
    for item in schedule:
        if step <= int(item["max_step"]):
            return float(item.get(key, default))
    return float(schedule[-1].get(key, default)) if schedule else default
