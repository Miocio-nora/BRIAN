import json
from pathlib import Path

from brian_sphere_llm.eval.routing_report import make_routing_report


def test_routing_report_preserves_latest_route_examples_and_trajectories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "train_log.jsonl",
        [
            {
                "route_entropy": 0.1,
                "top1_block_histogram": {"0": 1, "1": 1, "2": 0},
                "exit_step_distribution": [0, 1],
                "first_exit_step_histogram": {"2": 1},
                "route_path_examples": [{"sample_index": 0, "actions": [0, 1]}],
                "position_norm_trajectory": [1.0, 0.75],
                "location_distance_trajectory": [0.5, 0.25],
            },
            {
                "route_entropy": 0.3,
                "top1_block_histogram": {"0": 0, "1": 1, "2": 1},
                "exit_step_distribution": [1, 1],
                "first_exit_step_histogram": {"1": 1},
                "route_path_examples": [{"sample_index": 0, "actions": [2]}],
                "position_norm_trajectory": [0.5],
                "location_distance_trajectory": [0.1],
            },
        ],
    )
    _write_jsonl(run_dir / "eval_log.jsonl", [{"validation_loss": 2.0, "perplexity": 7.4}])

    output = make_routing_report(run_dir)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["summary"]["route_entropy"] == 0.2
    assert report["latest_block_histogram"] == {"0": 0, "1": 1, "2": 1}
    assert report["latest_exit_step_distribution"] == [1, 1]
    assert report["latest_first_exit_step_histogram"] == {"1": 1}
    assert report["latest_route_path_examples"] == [{"sample_index": 0, "actions": [2]}]
    assert report["latest_position_norm_trajectory"] == [0.5]
    assert report["latest_location_distance_trajectory"] == [0.1]
    assert report["latest_eval"]["validation_loss"] == 2.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
