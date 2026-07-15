from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Protocol


class EventSource(Protocol):
    metadata: Mapping[str, Any]

    def poll(self) -> list[dict[str, Any]]: ...


class JsonlTail:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.inode: int | None = None

    def poll(self, *, limit: int = 2000) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        stat = self.path.stat()
        if self.inode is None or self.inode != stat.st_ino or stat.st_size < self.offset:
            self.inode = stat.st_ino
            self.offset = 0
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            while len(rows) < limit:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(start)
                    break
                self.offset = handle.tell()
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows


class TelemetrySource:
    def __init__(self, run_dir: str | Path, *, output_dir: str = "terminal_dashboard") -> None:
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / output_dir
        self.metadata = _load_json(self.directory / "metadata.json")
        self.tail = JsonlTail(self.directory / "events.jsonl")

    def poll(self) -> list[dict[str, Any]]:
        if not self.metadata:
            self.metadata = _load_json(self.directory / "metadata.json")
        return self.tail.poll()


class ReplaySource:
    def __init__(self, path: str | Path, *, speed: float = 1.0) -> None:
        self.path = Path(path)
        self.speed = max(0.05, float(speed))
        self.metadata = _load_json(self.path.parent / "metadata.json")
        self.events = _load_jsonl(self.path)
        self.index = 0
        self.started = time.monotonic()
        self.base_created = _event_timestamp(self.events[0]) if self.events else 0.0

    def poll(self) -> list[dict[str, Any]]:
        if not self.events:
            return []
        elapsed = (time.monotonic() - self.started) * self.speed
        rows: list[dict[str, Any]] = []
        while self.index < len(self.events):
            event = self.events[self.index]
            event_time = _event_timestamp(event) - self.base_created
            if event_time > elapsed and rows:
                break
            if event_time > elapsed:
                break
            rows.append(event)
            self.index += 1
        return rows


class DemoSource:
    def __init__(self, *, num_blocks: int = 8, max_steps: int = 1200, interval: float = 0.7) -> None:
        self.metadata = {
            "run_name": "BRIAN / BDRE LIVE",
            "model_name": "BRIAN-R125-BDRE-RCKV-v1",
            "max_steps": max_steps,
            "start_step": 0,
            "num_blocks": num_blocks,
            "world_size": 4,
            "device_name": "NVIDIA B200",
            "device_memory_mb": 183500.0,
        }
        self.num_blocks = num_blocks
        self.max_steps = max_steps
        self.interval = interval
        self.random = random.Random(1729)
        self.step = 0
        self.last_emit = 0.0

    def poll(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if now - self.last_emit < self.interval:
            return []
        self.last_emit = now
        self.step = min(self.max_steps, self.step + 1)
        progress = self.step / self.max_steps
        loss = 1.94 + 2.55 * math.exp(-5.2 * progress) + self.random.uniform(-0.035, 0.035)
        warmup = min(1.0, self.step / 80.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * max(0.0, (self.step - 80) / (self.max_steps - 80))))
        learning_rate = 3e-4 * warmup * decay
        route = self._route()
        event = {
            "kind": "train",
            "step": self.step,
            "max_steps": self.max_steps,
            "token_index": (self.step * 97) % 2048,
            "route": route,
            "metrics": {
                "loss": loss,
                "lm_loss": loss - 0.015,
                "learning_rate": learning_rate,
                "tokens_per_second": 18300 + self.random.uniform(-900, 1100),
                "train_step_time_seconds": 2.85 + self.random.uniform(-0.12, 0.12),
                "cuda_memory_allocated_mb": 62100 + self.random.uniform(-800, 1200),
                "cuda_max_memory_allocated_mb": 66800 + self.random.uniform(-400, 800),
                "block_load_entropy_normalized": 0.82 + self.random.uniform(-0.045, 0.045),
                "route_entropy": 1.58 + self.random.uniform(-0.08, 0.08),
                "average_route_steps": float(len(route)),
                "bdre_key_weight_entropy": 1.32 + self.random.uniform(-0.08, 0.08),
            },
        }
        rows = [event]
        if self.step % 120 == 0:
            rows.append(
                {
                    "kind": "eval",
                    "step": self.step,
                    "metrics": {"validation_loss": loss + self.random.uniform(0.06, 0.14)},
                }
            )
        if self.step == self.max_steps:
            rows.append({"kind": "status", "step": self.step, "status": "complete"})
        return rows

    def _route(self) -> list[int]:
        length = self.random.randint(5, 10)
        current = self.random.randrange(self.num_blocks)
        route = [current]
        for _ in range(length - 1):
            if self.random.random() < 0.16:
                route.append(current)
                continue
            choices = [index for index in range(self.num_blocks) if index != current]
            current = self.random.choice(choices)
            route.append(current)
        return route


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _event_timestamp(event: Mapping[str, Any]) -> float:
    value = event.get("created_at")
    if not isinstance(value, str):
        return float(event.get("step", 0))
    try:
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float(event.get("step", 0))
