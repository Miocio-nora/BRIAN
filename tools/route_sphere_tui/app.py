from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical

from tools.route_sphere_tui.source import DemoSource, EventSource, ReplaySource, TelemetrySource
from tools.route_sphere_tui.state import DashboardState
from tools.route_sphere_tui.widgets import (
    HistoryChart,
    MetricGrid,
    RouteHealth,
    RouteSphereWidget,
    TrainingHeader,
    TrainingProgress,
)


class RouteSphereApp(App[None]):
    CSS = """
    Screen {
        background: #111111;
        color: #e2e2e2;
    }

    #body {
        width: 100%;
        height: 100%;
    }

    #left {
        width: 39%;
        min-width: 38;
        height: 100%;
        padding: 1 2;
        background: #181818;
        border-right: solid #343434;
    }

    #header {
        height: 3;
        margin-bottom: 1;
    }

    #progress {
        height: 4;
        margin-bottom: 1;
    }

    #metrics {
        height: 5;
        margin-bottom: 1;
        padding: 0 1;
        background: #181818;
    }

    .chart {
        height: 1fr;
        min-height: 6;
        margin-top: 1;
        background: #181818;
    }

    #health {
        height: 4;
        margin-top: 1;
    }

    #sphere {
        width: 1fr;
        height: 100%;
        background: #111111;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
        ("r", "toggle_rotation", "Rotation"),
        ("p", "toggle_layout", "Position layout"),
    ]

    def __init__(self, source: EventSource, *, fps: float = 24.0) -> None:
        super().__init__()
        self.source = source
        self.dashboard = DashboardState.from_metadata(source.metadata)
        self.fps = min(60.0, max(8.0, float(fps)))

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield TrainingHeader(id="header")
                yield TrainingProgress(id="progress")
                yield MetricGrid(id="metrics")
                yield HistoryChart("loss", "TRAIN / VALIDATION LOSS", (238, 238, 238), id="loss", classes="chart")
                yield HistoryChart("lr", "LEARNING RATE", (176, 176, 176), id="lr", classes="chart")
                yield RouteHealth(id="health")
            yield RouteSphereWidget(id="sphere")

    def on_mount(self) -> None:
        self._poll_source()
        self.set_interval(0.12, self._poll_source)
        self.set_interval(1.0 / self.fps, self._refresh_frame)

    def action_toggle_rotation(self) -> None:
        self.query_one(RouteSphereWidget).toggle_rotation()

    def action_toggle_layout(self) -> None:
        self.query_one(RouteSphereWidget).toggle_layout()

    def _poll_source(self) -> None:
        self.dashboard.update_metadata(self.source.metadata)
        for event in self.source.poll():
            self.dashboard.apply(event)
        self._refresh_widgets()

    def _refresh_frame(self) -> None:
        self.query_one(RouteSphereWidget).refresh()
        self.query_one(TrainingHeader).refresh()

    def _refresh_widgets(self) -> None:
        for widget in self.query("#left > *"):
            widget.refresh()
        self.query_one(RouteSphereWidget).refresh()


def build_source(args: argparse.Namespace) -> EventSource:
    if args.demo:
        return DemoSource(num_blocks=args.blocks)
    if args.replay is not None:
        return ReplaySource(args.replay, speed=args.speed)
    if args.run_dir is None:
        raise ValueError("--run-dir, --replay, or --demo is required.")
    return TelemetrySource(args.run_dir, output_dir=args.output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BRIAN training and route-sphere terminal dashboard.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Training run directory to follow.")
    source.add_argument("--replay", type=Path, help="Recorded terminal_dashboard/events.jsonl file.")
    source.add_argument("--demo", action="store_true", help="Run the self-contained animated demo.")
    parser.add_argument("--output-dir", default="terminal_dashboard", help="Telemetry directory below run-dir.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--fps", type=float, default=24.0, help="Animation frame rate.")
    parser.add_argument("--blocks", type=int, default=8, help="Demo route block count.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = RouteSphereApp(build_source(args), fps=args.fps)
    app.run()


if __name__ == "__main__":
    main()
