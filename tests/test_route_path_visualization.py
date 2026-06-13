from collections import Counter
from pathlib import Path
import json

import numpy as np
import pytest

from brian_sphere_llm.eval.route_path_visualization import (
    _aggregate_from_counts,
    _aggregate_without_input_for_display,
    _path_counts_from_train_row,
    _project_nodes,
    _write_html,
    make_route_path_visualization_from_train_log,
)


def test_project_nodes_uses_pca_and_labels_out_action() -> None:
    positions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    nodes, report = _project_nodes(positions)

    assert [node["label"] for node in nodes] == ["B0", "B1", "B2", "OUT"]
    assert all({"x", "y", "z"} <= set(node) for node in nodes)
    assert report["method"] == "pca"
    assert len(report["explained_variance_ratio"]) == 3


def test_project_nodes_can_include_independent_input_position() -> None:
    positions = np.eye(3, 4)
    input_position = np.array([0.5, 0.5, 0.0, 0.0])

    nodes, _ = _project_nodes(positions, input_position=input_position)

    assert [node["label"] for node in nodes] == ["B0", "B1", "OUT", "IN"]
    assert nodes[-1]["kind"] == "input"
    assert nodes[-1]["route_action"] is None


def test_train_log_path_counts_are_exact_and_truncated_at_out() -> None:
    counts, exact = _path_counts_from_train_row(
        {
            "route_path_counts": [
                {"actions": [0, 1, 2, 2], "count": 3},
                {"actions": [1, 2], "count": 1},
            ],
            "route_path_examples": [{"actions": [0, 0]}],
        },
        out_action=2,
    )

    assert exact is True
    assert counts == Counter({(0, 1, 2): 3, (1, 2): 1})


def test_train_log_path_counts_can_prepend_input_node() -> None:
    counts, exact = _path_counts_from_train_row(
        {"route_path_counts": [{"actions": [0, 1, 2], "count": 3}]},
        out_action=2,
        input_node=3,
    )

    assert exact is True
    assert counts == Counter({(3, 0, 1, 2): 3})


def test_train_log_path_examples_are_approximate_fallback() -> None:
    counts, exact = _path_counts_from_train_row(
        {"route_path_examples": [{"actions": [0, 0, 2, 2]}, {"actions": [0, 1, 2]}]},
        out_action=2,
    )

    assert exact is False
    assert counts == Counter({(0, 0, 2): 1, (0, 1, 2): 1})


def test_aggregate_from_counts_reports_paths_edges_and_nodes() -> None:
    aggregate = _aggregate_from_counts(
        name="unit",
        path_counts=Counter({(0, 1, 2): 3, (0, 2): 1}),
        exact=True,
        metadata={"sample_count": 4},
    )

    assert aggregate["total_path_count"] == 4
    assert aggregate["unique_path_count"] == 2
    assert {"source": 0, "target": 1, "count": 3} in aggregate["edge_counts"]
    assert {"source": 0, "target": 2, "count": 1} in aggregate["edge_counts"]
    assert {"action": 2, "count": 4} in aggregate["node_counts"]


def test_aggregate_without_input_for_display_strips_input_node() -> None:
    aggregate = _aggregate_from_counts(
        name="unit",
        path_counts=Counter({(3, 0, 1, 2): 3, (3, 1, 2): 1}),
        exact=True,
        metadata={"sample_count": 4},
    )

    stripped = _aggregate_without_input_for_display(aggregate, input_node=3, top_paths=8)

    assert stripped["path_counts"] == [
        {"actions": [0, 1, 2], "count": 3},
        {"actions": [1, 2], "count": 1},
    ]
    assert all(row["source"] != 3 and row["target"] != 3 for row in stripped["edge_counts"])
    assert all(row["action"] != 3 for row in stripped["node_counts"])


def test_write_html_creates_plotly_document(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    nodes = [
        {"action": 0, "label": "B0", "x": 0.0, "y": 0.0, "z": 0.0},
        {"action": 1, "label": "B1", "x": 1.0, "y": 0.0, "z": 0.0},
        {"action": 2, "label": "OUT", "x": 1.0, "y": 1.0, "z": 0.0},
    ]
    aggregate = _aggregate_from_counts(
        name="checkpoint",
        path_counts=Counter({(0, 1, 2): 2}),
        exact=True,
        metadata={"sample_count": 2},
    )
    report = {
        "run_dir": "runs/unit",
        "source": "checkpoint",
        "sidecar_json": str(tmp_path / "paths.json"),
        "projection": {"method": "pca", "explained_variance_ratio": [1.0, 0.0, 0.0]},
        "checks": {"paths_present": True},
        "overall_status": "pass",
        "nodes": nodes,
        "aggregates": [aggregate],
    }
    output_path = tmp_path / "paths.html"

    _write_html(report, output_path, top_paths=8)

    html = output_path.read_text(encoding="utf-8")
    assert "Route Path Visualization" in html
    assert "Plotly" in html or "plotly" in html


def test_make_route_path_visualization_from_train_log(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("plotly")

    class PositionTable:
        embeddings = torch.eye(3, 4)
        input_position = torch.tensor([0.5, 0.5, 0.0, 0.0])

    class Model:
        position_table = PositionTable()

    train_log = tmp_path / "train_log.jsonl"
    train_log.write_text(
        '{"step": 1, "route_path_counts": [{"actions": [0, 1, 2], "count": 2}]}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "paths.html"

    result = make_route_path_visualization_from_train_log(
        tmp_path,
        Model(),
        output_path=output_path,
        step=1,
    )

    assert result == output_path
    assert output_path.exists()
    sidecar = output_path.with_suffix(".json")
    assert sidecar.exists()
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["nodes"][-1]["label"] == "IN"
    assert report["aggregates"][0]["path_counts"][0]["actions"] == [3, 0, 1, 2]
    html = output_path.read_text(encoding="utf-8")
    assert "Show IN" in html
    assert "Hide IN" in html
