from __future__ import annotations

from itertools import combinations
import math
import time
from typing import Iterable

import numpy as np
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static

from tools.route_sphere_tui.canvas import BrailleCanvas, RGB
from tools.route_sphere_tui.state import DashboardState, compact_number, format_duration

INK = (226, 226, 226)
MUTED = (124, 124, 124)
DIM = (42, 42, 42)
BRIGHT = (244, 244, 244)
MID = (170, 170, 170)
SOFT = (96, 96, 96)
WHITE = (255, 255, 255)


class StateWidget(Static):
    @property
    def dashboard(self) -> DashboardState:
        return self.app.dashboard  # type: ignore[attr-defined]


class TrainingHeader(StateWidget):
    def render(self) -> Text:
        state = self.dashboard
        status = "STALE" if state.stale else state.status.upper()
        status_color = MUTED if state.stale else BRIGHT if state.status == "running" else MID
        title = Text()
        title.append(state.run_name, Style(color=_color(BRIGHT), bold=True))
        title.append("\n")
        title.append("● ", Style(color=_color(status_color), bold=True))
        title.append(status, Style(color=_color(status_color), bold=True))
        if state.model_name:
            title.append("    ")
            title.append(state.model_name, Style(color=_color(MUTED)))
        return title


class TrainingProgress(StateWidget):
    def render(self) -> Text:
        state = self.dashboard
        width = max(12, self.size.width - 2)
        bar_width = max(8, width - 2)
        filled = min(bar_width, int(round(bar_width * state.progress)))
        output = Text()
        output.append(f"STEP  {state.step:,} / {state.max_steps:,}", Style(color=_color(INK), bold=True))
        output.append(f"   {state.progress * 100:5.1f}%", Style(color=_color(BRIGHT), bold=True))
        output.append("\n")
        output.append("━" * filled, Style(color=_color(BRIGHT), bold=True))
        output.append("━" * (bar_width - filled), Style(color=_color(DIM)))
        output.append("\n")
        output.append("ETA  ", Style(color=_color(MUTED)))
        output.append(format_duration(state.eta_seconds), Style(color=_color(INK)))
        output.append("    STEP  ", Style(color=_color(MUTED)))
        output.append(
            f"{state.metrics.get('train_step_time_seconds', 0.0):.2f}s"
            if state.metrics.get("train_step_time_seconds") is not None
            else "--",
            Style(color=_color(INK)),
        )
        return output


class MetricGrid(StateWidget):
    def render(self) -> Table:
        state = self.dashboard
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        loss = state.metrics.get("loss")
        validation = state.eval_metrics.get("validation_loss")
        lr = state.metrics.get("learning_rate")
        throughput = state.metrics.get("tokens_per_second")
        memory = state.metrics.get("cuda_max_memory_allocated_mb")
        table.add_row(
            _metric("LOSS", compact_number(loss, decimals=4), BRIGHT),
            _metric("VAL", compact_number(validation, decimals=4), MID),
            _metric("LR", compact_number(lr), INK),
        )
        table.add_row(
            _metric("TOK / S", compact_number(throughput, decimals=1), INK),
            _metric("GPU PEAK", _memory(memory), INK),
            _metric("WORLD", str(state.world_size), MUTED),
        )
        return table


class HistoryChart(StateWidget):
    def __init__(self, metric: str, title: str, color: RGB, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.chart_title = title
        self.chart_color = color

    def render(self) -> Text:
        state = self.dashboard
        if self.metric == "loss":
            values = list(state.loss_history)
            secondary = list(state.eval_loss_history)
            latest = state.metrics.get("loss")
        else:
            values = list(state.lr_history)
            secondary = []
            latest = state.metrics.get("learning_rate")

        output = Text()
        output.append(self.chart_title, Style(color=_color(MUTED), bold=True))
        output.append("  ")
        output.append(compact_number(latest, decimals=5 if self.metric == "lr" else 4), Style(color=_color(self.chart_color), bold=True))
        output.append("\n")
        chart_height = max(2, self.size.height - 2)
        canvas = BrailleCanvas(max(4, self.size.width), chart_height)
        for fraction in (0.25, 0.5, 0.75):
            y = (chart_height - 1) * fraction
            canvas.line(0, y, self.size.width - 1, y, color=DIM, intensity=0.11, dashed=True)
        _draw_history(canvas, values, color=self.chart_color, secondary=secondary)
        output.append_text(canvas.text())
        return output


class RouteHealth(StateWidget):
    def render(self) -> Text:
        state = self.dashboard
        width = max(8, self.size.width - 24)
        load = state.metrics.get("block_load_entropy_normalized")
        route_entropy = state.metrics.get("route_entropy")
        depth = state.metrics.get("average_route_steps")
        memory = state.metrics.get("cuda_max_memory_allocated_mb")
        memory_ratio = None
        if memory is not None and state.device_memory_mb:
            memory_ratio = memory / state.device_memory_mb
        output = Text()
        output.append_text(_gauge("BLOCK BALANCE", load, width, BRIGHT))
        output.append("\n")
        output.append_text(_gauge("GPU MEMORY", memory_ratio, width, MID))
        output.append("\n")
        output.append("ROUTE ENTROPY  ", Style(color=_color(MUTED)))
        output.append(compact_number(route_entropy, decimals=3), Style(color=_color(INK), bold=True))
        output.append("      DEPTH  ", Style(color=_color(MUTED)))
        output.append(compact_number(depth, decimals=2), Style(color=_color(MID), bold=True))
        return output


class RouteSphereWidget(Widget):
    can_focus = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rotation_enabled = True
        self.learned_layout_enabled = False
        self._route_revision = -1
        self._position_revision = -1
        self._route_started = time.monotonic()
        self._route: list[int] = []
        self._learned_nodes: np.ndarray | None = None
        self._phase_origin = time.monotonic()

    @property
    def dashboard(self) -> DashboardState:
        return self.app.dashboard  # type: ignore[attr-defined]

    def render(self) -> Text:
        width = max(4, self.size.width)
        height = max(4, self.size.height)
        canvas = BrailleCanvas(width, height)
        state = self.dashboard
        if state.route_revision != self._route_revision:
            self._route_revision = state.route_revision
            self._route = list(state.route)
            self._route_started = time.monotonic()

        phase = time.monotonic() - self._phase_origin
        yaw = 0.38 + (phase * 0.045 if self.rotation_enabled else 0.0)
        pitch = -0.22 + 0.025 * math.sin(phase * 0.14)
        roll = 0.018 * math.sin(phase * 0.09)
        rotation = _rotation_matrix(yaw, pitch, roll)
        raw_nodes = self._layout_nodes(state)
        rotated = raw_nodes @ rotation.T
        projected = _project(rotated, width, height)

        self._draw_shell(canvas, rotation, width, height)
        self._draw_connections(canvas, projected, rotated)
        active_nodes = self._draw_route(canvas, projected)
        self._draw_nodes(canvas, projected, rotated, active_nodes)
        return canvas.text()

    def toggle_rotation(self) -> None:
        self.rotation_enabled = not self.rotation_enabled

    def toggle_layout(self) -> None:
        self.learned_layout_enabled = not self.learned_layout_enabled

    def _layout_nodes(self, state: DashboardState) -> np.ndarray:
        if not self.learned_layout_enabled or state.positions is None:
            return _sphere_nodes(state.num_blocks)
        if state.position_revision != self._position_revision:
            self._position_revision = state.position_revision
            reference = self._learned_nodes
            if reference is None or reference.shape[0] != state.num_blocks:
                reference = _sphere_nodes(state.num_blocks)
            self._learned_nodes = _learned_position_layout(
                state.positions[: state.num_blocks],
                reference=reference,
            )
        return self._learned_nodes if self._learned_nodes is not None else _sphere_nodes(state.num_blocks)

    def _draw_shell(self, canvas: BrailleCanvas, rotation: np.ndarray, width: int, height: int) -> None:
        del rotation
        samples = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=True)
        radius_x, radius_y = _projection_radii(width, height)
        points = np.stack(
            [
                width * 0.5 + np.cos(samples) * radius_x,
                height * 0.5 - np.sin(samples) * radius_y,
            ],
            axis=1,
        )
        canvas.polyline(points, color=(48, 48, 48), intensity=0.13)

    def _draw_connections(self, canvas: BrailleCanvas, points: np.ndarray, rotated: np.ndarray) -> None:
        ordered = sorted(
            _connection_pairs(rotated),
            key=lambda pair: float(rotated[pair[0], 2] + rotated[pair[1], 2]),
        )
        for first, second in ordered:
            depth = float((rotated[first, 2] + rotated[second, 2]) * 0.5)
            intensity = 0.052 + 0.05 * (depth + 1.0) * 0.5
            canvas.line(*points[first], *points[second], color=(66, 66, 66), intensity=intensity)

    def _draw_route(self, canvas: BrailleCanvas, points: np.ndarray) -> set[int]:
        route = [value for value in self._route if 0 <= value < len(points)]
        if not route:
            return set()
        if len(route) == 1:
            pulse = 0.5 + 0.5 * math.sin((time.monotonic() - self._route_started) * 5.0)
            canvas.point(*points[route[0]], color=WHITE, intensity=0.38 + 0.42 * pulse, radius=1.5 + pulse)
            return {route[0]}

        segment_duration = 0.58
        hold_duration = 1.15
        segment_count = len(route) - 1
        cycle = segment_count * segment_duration + hold_duration
        elapsed = (time.monotonic() - self._route_started) % max(segment_duration, cycle)
        completed = min(segment_count, int(elapsed / segment_duration))
        active_fraction = (elapsed / segment_duration) - completed if completed < segment_count else 1.0

        for index in range(completed):
            first, second = route[index], route[index + 1]
            if first == second:
                continue
            canvas.line(*points[first], *points[second], color=(174, 174, 174), intensity=0.58)

        active_nodes = set(route[: completed + 1])
        if completed < segment_count:
            first, second = route[completed], route[completed + 1]
            active_nodes.add(second)
            if first == second:
                pulse = math.sin(active_fraction * math.pi)
                canvas.point(*points[first], color=WHITE, intensity=0.75 + 0.25 * pulse, radius=1.0 + 2.0 * pulse)
            else:
                canvas.line(
                    *points[first],
                    *points[second],
                    color=(190, 190, 190),
                    intensity=0.55,
                    end=active_fraction,
                )
                tail_start = max(0.0, active_fraction - 0.24)
                canvas.line(
                    *points[first],
                    *points[second],
                    color=WHITE,
                    intensity=0.98,
                    start=tail_start,
                    end=active_fraction,
                )
                x = points[first, 0] + (points[second, 0] - points[first, 0]) * active_fraction
                y = points[first, 1] + (points[second, 1] - points[first, 1]) * active_fraction
                canvas.point(x, y, color=WHITE, intensity=1.0, radius=0.8)
        else:
            active_nodes.update(route)
            for first, second in zip(route[:-1], route[1:]):
                if first != second:
                    canvas.line(*points[first], *points[second], color=(174, 174, 174), intensity=0.58)
        return active_nodes

    def _draw_nodes(
        self,
        canvas: BrailleCanvas,
        points: np.ndarray,
        rotated: np.ndarray,
        active_nodes: set[int],
    ) -> None:
        for index in np.argsort(rotated[:, 2]):
            depth = float((rotated[index, 2] + 1.0) * 0.5)
            active = int(index) in active_nodes
            glow_color = BRIGHT if active else SOFT
            canvas.point(
                *points[index],
                color=glow_color,
                intensity=(0.28 + 0.2 * depth) if active else (0.1 + 0.08 * depth),
                radius=1.5 if active else 0.8,
            )
            level = int(132 + 62 * depth)
            color = WHITE if active else (level, level, level)
            canvas.glyph(*points[index], "*", color=color, bold=active, priority=2.0 + depth)


def _draw_history(
    canvas: BrailleCanvas,
    values: list[tuple[int, float]],
    *,
    color: RGB,
    secondary: list[tuple[int, float]],
) -> None:
    combined = values + secondary
    if not combined:
        return
    x_min = min(step for step, _ in combined)
    x_max = max(step for step, _ in combined)
    raw_values = [value for _, value in combined]
    y_min = min(raw_values)
    y_max = max(raw_values)
    padding = max(1e-9, (y_max - y_min) * 0.12)
    y_min -= padding
    y_max += padding

    def point(item: tuple[int, float]) -> tuple[float, float]:
        step, value = item
        x = (step - x_min) / max(1, x_max - x_min) * (canvas.width - 1)
        y = (1.0 - (value - y_min) / max(1e-12, y_max - y_min)) * (canvas.height - 1)
        return x, y

    points = [point(item) for item in values]
    canvas.polyline(points, color=color, intensity=0.92)
    if secondary:
        secondary_points = [point(item) for item in secondary]
        for x, y in secondary_points:
            canvas.point(x, y, color=MID, intensity=1.0, radius=0.4)


def _sphere_nodes(count: int) -> np.ndarray:
    count = max(1, int(count))
    if count == 8:
        scale = 1.0 / math.sqrt(3.0)
        rows = [
            (x * scale, y * scale, z * scale)
            for z in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for x in (-1.0, 1.0)
        ]
        return np.asarray(rows, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rows = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden_angle * index
        rows.append((radius * math.cos(angle), y, radius * math.sin(angle)))
    return np.asarray(rows, dtype=np.float64)


def _connection_pairs(nodes: np.ndarray) -> list[tuple[int, int]]:
    count = len(nodes)
    if count == 8:
        return [pair for pair in combinations(range(8), 2) if pair[0] ^ pair[1] in (1, 2, 4)]
    if count < 2:
        return []
    neighbors = min(3, count - 1)
    distances = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    pairs: set[tuple[int, int]] = set()
    for index in range(count):
        nearest = np.argsort(distances[index])[1 : neighbors + 1]
        pairs.update((min(index, int(other)), max(index, int(other))) for other in nearest)
    return sorted(pairs)


def _learned_position_layout(positions: list[list[float]], *, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return reference
    values = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(values, full_matrices=False)
    dimensions = min(3, vh.shape[0])
    projected = values @ vh[:dimensions].T
    if dimensions < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - dimensions)))
    radius = float(np.linalg.norm(projected, axis=1).max())
    if radius <= 1e-12:
        return reference
    projected = projected / radius
    if reference.shape == projected.shape:
        left, _, right = np.linalg.svd(projected.T @ reference, full_matrices=False)
        projected = projected @ (left @ right)
    return projected


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rotate_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotate_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rotate_z = np.asarray([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rotate_z @ rotate_x @ rotate_y


def _project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    radius_x, radius_y = _projection_radii(width, height)
    x = width * 0.5 + points[:, 0] * radius_x
    y = height * 0.5 - points[:, 1] * radius_y
    return np.stack([x, y], axis=1)


def _projection_radii(width: int, height: int) -> tuple[float, float]:
    return max(2.0, width * 0.35), max(2.0, height * 0.41)


def _gauge(label: str, value: float | None, width: int, color: RGB) -> Text:
    ratio = min(1.0, max(0.0, value if value is not None else 0.0))
    filled = int(round(width * ratio))
    output = Text()
    output.append(f"{label:<14}", Style(color=_color(MUTED)))
    output.append("━" * filled, Style(color=_color(color), bold=True))
    output.append("━" * (width - filled), Style(color=_color(DIM)))
    output.append(f" {ratio * 100:5.1f}%" if value is not None else "    -- ", Style(color=_color(INK)))
    return output


def _metric(label: str, value: str, color: RGB) -> Text:
    output = Text()
    output.append(label, Style(color=_color(MUTED)))
    output.append("\n")
    output.append(value, Style(color=_color(color), bold=True))
    return output


def _memory(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value / 1024.0:.1f} GiB"


def _color(value: RGB) -> str:
    return f"rgb({value[0]},{value[1]},{value[2]})"
