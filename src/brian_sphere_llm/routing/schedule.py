from __future__ import annotations

import math
from typing import Any


def scheduled_value(schedule: list[dict], step: int, key: str, default: Any) -> float:
    for item in schedule:
        if not isinstance(item, dict):
            raise ValueError("Schedule items must be mappings.")
        max_step = _schedule_number(item.get("max_step"), key="max_step")
        if step <= int(max_step):
            return _schedule_number(item.get(key, default), key=key)
    return _schedule_number(schedule[-1].get(key, default), key=key) if schedule else _schedule_number(default, key=key)


def _schedule_number(value: Any, *, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Schedule field {key!r} must be a finite numeric value.")
    return float(value)
