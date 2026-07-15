from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from brian_sphere_llm.train.live_telemetry import (
    LiveTelemetryWriter,
    TerminalDashboardConfig,
    extract_route_trace,
)
from brian_sphere_llm.utils.config import load_config
from tools.route_sphere_tui.app import RouteSphereApp
from tools.route_sphere_tui.canvas import BrailleCanvas
from tools.route_sphere_tui.source import DemoSource, JsonlTail, ReplaySource
from tools.route_sphere_tui.state import DashboardState
from tools.route_sphere_tui.widgets import RouteSphereWidget, _learned_position_layout, _sphere_nodes


def test_terminal_dashboard_config_defaults_off_and_validates() -> None:
    assert not TerminalDashboardConfig.from_train_config({}).enabled
    config = TerminalDashboardConfig.from_train_config(
        {
            "terminal_dashboard": {
                "enabled": True,
                "interval": 3,
                "position_interval": 20,
                "sample_index": 1,
                "output_dir": "live",
            }
        }
    )
    assert config.train_due(3, 100)
    assert not config.train_due(4, 100)
    assert config.positions_due(100, 100)
    with pytest.raises(ValueError, match="interval"):
        TerminalDashboardConfig.from_train_config({"terminal_dashboard": {"interval": 0}})


def test_bdre_train_configs_keep_dashboard_separate_from_loss_weights() -> None:
    for path in [
        "configs/train/stage5_bdre_tiny_debug.yaml",
        "configs/train/bdre_rckv_r125_5b_ddp2_legacyval.yaml",
    ]:
        config = load_config(path)
        dashboard = TerminalDashboardConfig.from_train_config(config)
        assert dashboard.enabled
        assert "selected_balance" not in config["terminal_dashboard"]
        assert config["loss_weights"]["selected_balance"] > 0.0


def test_extract_route_trace_filters_out_and_uses_one_sample() -> None:
    outputs = {
        "route_info": {
            "selected_actions": [
                torch.tensor([1, 5]),
                torch.tensor([3, 4]),
                torch.tensor([8, 2]),
                torch.tensor([7, 1]),
            ]
        }
    }
    assert extract_route_trace(outputs, num_internal_blocks=8, sample_index=0) == [1, 3]
    assert extract_route_trace(outputs, num_internal_blocks=8, sample_index=1) == [5, 4, 2, 1]


def test_live_telemetry_writer_records_compact_events(tmp_path: Path) -> None:
    config = TerminalDashboardConfig(enabled=True, interval=1, position_interval=2)
    writer = LiveTelemetryWriter(
        tmp_path,
        config,
        run_name="test_run",
        max_steps=10,
        start_step=0,
        num_blocks=8,
        model_name="test_model",
        world_size=2,
        device_name="B200",
        device_memory_mb=183500.0,
    )
    writer.write_train(
        {"loss": 2.5, "learning_rate": 1e-4, "ignored": {"large": "mapping"}, "unused_numeric": 9.0},
        step=1,
        max_steps=10,
        route=[1, 2, 2, 7],
        token_index=31,
        positions=[[0.0, 1.0] for _ in range(8)],
    )
    writer.write_eval({"validation_loss": 2.7}, step=1)
    writer.write_status("complete", step=10)

    metadata = json.loads((tmp_path / "terminal_dashboard" / "metadata.json").read_text())
    assert metadata["num_blocks"] == 8
    events = [json.loads(line) for line in (tmp_path / "terminal_dashboard" / "events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["status", "train", "eval", "status"]
    assert events[1]["route"] == [1, 2, 2, 7]
    assert "ignored" not in events[1]["metrics"]
    assert "unused_numeric" not in events[1]["metrics"]
    assert writer.healthy


def test_jsonl_tail_waits_for_complete_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind":"train","step":1}\n{"kind":"train"', encoding="utf-8")
    tail = JsonlTail(path)
    assert [row["step"] for row in tail.poll()] == [1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(',"step":2}\n')
    assert [row["step"] for row in tail.poll()] == [2]


def test_replay_source_replays_recorded_events(tmp_path: Path) -> None:
    metadata = {"run_name": "replay", "num_blocks": 8, "max_steps": 2}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    path = tmp_path / "events.jsonl"
    rows = [
        {"created_at": "2026-07-16T00:00:00+00:00", "kind": "train", "step": 1},
        {"created_at": "2026-07-16T00:00:00+00:00", "kind": "train", "step": 2},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source = ReplaySource(path, speed=1000.0)
    assert source.metadata["run_name"] == "replay"
    assert [row["step"] for row in source.poll()] == [1, 2]


def test_dashboard_state_accumulates_metrics_and_route() -> None:
    state = DashboardState.from_metadata({"max_steps": 100, "num_blocks": 8})
    state.apply(
        {
            "kind": "train",
            "step": 25,
            "max_steps": 100,
            "token_index": 77,
            "route": [0, 3, 3, 6],
            "metrics": {"loss": 2.1, "learning_rate": 3e-4, "tokens_per_second": 1234},
        }
    )
    state.apply({"kind": "eval", "step": 25, "metrics": {"validation_loss": 2.3}})
    assert state.progress == pytest.approx(0.25)
    assert state.route == [0, 3, 3, 6]
    assert state.route_revision == 1
    assert list(state.loss_history) == [(25, 2.1)]
    assert list(state.eval_loss_history) == [(25, 2.3)]


def test_sphere_layout_is_unit_norm_and_not_a_ring() -> None:
    nodes = _sphere_nodes(8)
    assert nodes.shape == (8, 3)
    assert np.linalg.norm(nodes, axis=1) == pytest.approx(np.ones(8))
    assert len({round(float(value), 3) for value in nodes[:, 2]}) == 2


def test_learned_position_layout_is_finite_and_aligned() -> None:
    reference = _sphere_nodes(8)
    positions = np.eye(8, dtype=np.float64).tolist()
    layout = _learned_position_layout(positions, reference=reference)
    assert layout.shape == (8, 3)
    assert np.isfinite(layout).all()
    assert np.linalg.norm(layout, axis=1).max() == pytest.approx(1.0)


def test_braille_canvas_preserves_ascii_node_overlay() -> None:
    canvas = BrailleCanvas(20, 8)
    canvas.line(1, 1, 18, 6, color=(60, 80, 100), intensity=0.2)
    canvas.glyph(10, 4, "*", color=(255, 255, 255), bold=True)
    rendered = canvas.text().plain
    assert rendered.count("*") == 1
    assert any(0x2800 <= ord(char) <= 0x28FF for char in rendered)


def test_textual_dashboard_mounts_and_sphere_is_text_free() -> None:
    async def run() -> None:
        app = RouteSphereApp(DemoSource(interval=0.0), fps=12)
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause(0.25)
            sphere = app.query_one(RouteSphereWidget)
            text = sphere.render().plain
            assert text.count("*") == 8
            assert app.dashboard.route_revision > 0
            assert all(char in {" ", "\n", "*"} or 0x2800 <= ord(char) <= 0x28FF for char in text)

    asyncio.run(run())


def test_textual_dashboard_keeps_all_nodes_in_compact_terminal() -> None:
    async def run() -> None:
        app = RouteSphereApp(DemoSource(interval=0.0), fps=8)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.2)
            assert app.query_one(RouteSphereWidget).render().plain.count("*") == 8

    asyncio.run(run())
