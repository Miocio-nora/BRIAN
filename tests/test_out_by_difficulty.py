import json
from pathlib import Path

from brian_sphere_llm.eval.out_by_difficulty import make_out_by_difficulty_report


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_out_by_difficulty_passes_when_hard_uses_more_compute(tmp_path: Path) -> None:
    samples = _write_jsonl(
        tmp_path / "samples.jsonl",
        [
            _sample("easy", route_steps=1.0, active=0.25, p_output=0.8),
            _sample("easy", route_steps=1.0, active=0.25, p_output=0.7),
            _sample("medium", route_steps=2.0, active=0.50, p_output=0.4),
            _sample("hard", route_steps=3.0, active=0.75, p_output=0.2),
            _sample("hard", route_steps=3.0, active=0.75, p_output=0.3),
        ],
    )
    output = make_out_by_difficulty_report(samples_path=samples, output_path=tmp_path / "out.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["by_difficulty"]["easy"]["sample_count"] == 2
    assert report["by_difficulty"]["hard"]["mean_route_steps"] == 3.0
    assert report["deltas"]["hard_minus_easy_route_steps"] == 2.0
    assert report["deltas"]["easy_minus_hard_p_output"] == 0.5
    assert all(report["checks"].values())


def test_out_by_difficulty_reads_samples_from_reasoning_report(tmp_path: Path) -> None:
    samples = _write_jsonl(
        tmp_path / "reasoning_samples.jsonl",
        [
            _sample("easy", route_steps=1.0, active=0.25, p_output=0.9),
            _sample("hard", route_steps=2.0, active=0.50, p_output=0.1),
        ],
    )
    reasoning = _write_json(tmp_path / "reasoning.json", {"samples_path": samples.name, "run_dir": "run-a"})
    output = make_out_by_difficulty_report(reasoning_report_path=reasoning, output_path=tmp_path / "out.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["inputs"]["reasoning_report"] == str(reasoning)
    assert report["inputs"]["reasoning_run_dir"] == "run-a"


def test_out_by_difficulty_fails_when_hard_exits_earlier(tmp_path: Path) -> None:
    samples = _write_jsonl(
        tmp_path / "samples.jsonl",
        [
            _sample("easy", route_steps=3.0, active=0.75, p_output=0.2),
            _sample("hard", route_steps=1.0, active=0.25, p_output=0.8),
        ],
    )
    output = make_out_by_difficulty_report(samples_path=samples, output_path=tmp_path / "out.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["route_steps_non_decreasing_with_difficulty"] is False
    assert report["checks"]["active_compute_non_decreasing_with_difficulty"] is False
    assert report["checks"]["easy_output_probability_at_least_hard"] is False


def test_out_by_difficulty_warns_when_routing_metrics_are_missing(tmp_path: Path) -> None:
    samples = _write_jsonl(
        tmp_path / "samples.jsonl",
        [
            {"difficulty": "easy"},
            {"difficulty": "hard"},
        ],
    )
    output = make_out_by_difficulty_report(samples_path=samples, output_path=tmp_path / "out.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "warn"
    assert report["checks"]["easy_and_hard_present"] is True
    assert report["checks"]["route_steps_non_decreasing_with_difficulty"] is None


def test_out_by_difficulty_rejects_boolean_routing_metrics(tmp_path: Path) -> None:
    samples = _write_jsonl(
        tmp_path / "samples.jsonl",
        [
            _sample("easy", route_steps=True, active=True, p_output=True),
            _sample("hard", route_steps=True, active=True, p_output=True),
        ],
    )
    output = make_out_by_difficulty_report(samples_path=samples, output_path=tmp_path / "out.json")
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "warn"
    assert report["by_difficulty"]["easy"]["mean_route_steps"] is None
    assert report["by_difficulty"]["hard"]["mean_p_output"] is None
    assert report["checks"]["easy_and_hard_present"] is True
    assert report["checks"]["route_steps_non_decreasing_with_difficulty"] is None
    assert report["checks"]["active_compute_non_decreasing_with_difficulty"] is None
    assert report["checks"]["easy_output_probability_at_least_hard"] is None


def _sample(difficulty: str, *, route_steps: float, active: float, p_output: float) -> dict:
    return {
        "difficulty": difficulty,
        "routing_average_route_steps": route_steps,
        "routing_active_block_evals_per_token": active,
        "routing_p_output_mean": p_output,
    }
