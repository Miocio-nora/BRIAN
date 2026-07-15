from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping


@dataclass
class DashboardState:
    run_name: str = "BRIAN"
    model_name: str = ""
    status: str = "waiting"
    step: int = 0
    max_steps: int = 1
    num_blocks: int = 8
    world_size: int = 1
    device_name: str = ""
    device_memory_mb: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    loss_history: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=720))
    eval_loss_history: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=240))
    lr_history: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=720))
    throughput_history: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=240))
    route: list[int] = field(default_factory=list)
    route_revision: int = 0
    token_index: int = 0
    positions: list[list[float]] | None = None
    position_revision: int = 0
    updated_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "DashboardState":
        state = cls()
        state.update_metadata(metadata or {})
        return state

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        if "run_name" in metadata:
            self.run_name = str(metadata["run_name"])
        if "model_name" in metadata:
            self.model_name = str(metadata["model_name"])
        if "start_step" in metadata:
            self.step = max(self.step, _safe_int(metadata.get("start_step"), self.step))
        if "max_steps" in metadata:
            self.max_steps = max(1, _safe_int(metadata.get("max_steps"), self.max_steps))
        if "num_blocks" in metadata:
            self.num_blocks = max(1, _safe_int(metadata.get("num_blocks"), self.num_blocks))
        if "world_size" in metadata:
            self.world_size = max(1, _safe_int(metadata.get("world_size"), self.world_size))
        if "device_name" in metadata:
            self.device_name = str(metadata["device_name"])
        memory = metadata.get("device_memory_mb")
        if isinstance(memory, (int, float)) and memory > 0:
            self.device_memory_mb = float(memory)

    def apply(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("kind", ""))
        self.updated_at = time.monotonic()
        if kind == "status":
            self.status = str(event.get("status", self.status))
            self.step = max(self.step, _safe_int(event.get("step"), self.step))
            return
        if kind not in {"train", "eval"}:
            return
        step = _safe_int(event.get("step"), self.step)
        self.step = max(self.step, step)
        self.max_steps = max(1, _safe_int(event.get("max_steps"), self.max_steps))
        numeric = _numeric_mapping(event.get("metrics"))
        if kind == "eval":
            self.eval_metrics.update(numeric)
            value = numeric.get("validation_loss")
            if value is not None:
                _append_step_value(self.eval_loss_history, step, value)
            return

        self.status = "running"
        self.metrics.update(numeric)
        if "loss" in numeric:
            _append_step_value(self.loss_history, step, numeric["loss"])
        if "learning_rate" in numeric:
            _append_step_value(self.lr_history, step, numeric["learning_rate"])
        if "tokens_per_second" in numeric:
            _append_step_value(self.throughput_history, step, numeric["tokens_per_second"])
        route = event.get("route")
        if isinstance(route, list):
            clean_route = [int(value) for value in route if isinstance(value, int) and 0 <= value < self.num_blocks]
            if clean_route:
                self.route = clean_route
                self.route_revision += 1
        self.token_index = max(0, _safe_int(event.get("token_index"), self.token_index))
        positions = event.get("positions")
        if _valid_positions(positions, self.num_blocks):
            self.positions = [[float(value) for value in row] for row in positions]
            self.position_revision += 1

    @property
    def progress(self) -> float:
        return min(1.0, max(0.0, self.step / max(1, self.max_steps)))

    @property
    def eta_seconds(self) -> float | None:
        step_time = self.metrics.get("train_step_time_seconds")
        if step_time is None or step_time <= 0 or self.step >= self.max_steps:
            return None
        return (self.max_steps - self.step) * step_time

    @property
    def stale(self) -> bool:
        return self.status == "running" and time.monotonic() - self.updated_at > 30.0


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compact_number(value: float | None, *, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.{decimals}f}k"
    if 0 < magnitude < 0.001:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def _append_step_value(history: deque[tuple[int, float]], step: int, value: float) -> None:
    if history and history[-1][0] == step:
        history[-1] = (step, value)
    else:
        history.append((step, value))


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            result[str(key)] = float(item)
    return result


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def _valid_positions(value: Any, count: int) -> bool:
    if not isinstance(value, list) or len(value) < count:
        return False
    width: int | None = None
    for row in value[:count]:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        if len(row) != width or not all(isinstance(item, (int, float)) for item in row):
            return False
    return True
