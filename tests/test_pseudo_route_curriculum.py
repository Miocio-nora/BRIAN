import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from brian_sphere_llm.eval.pseudo_route_curriculum import make_pseudo_route_curriculum_report
from brian_sphere_llm.routing.pseudo_policy import mixed_skip_recur_actions
from brian_sphere_llm.utils.config import load_config


def test_mixed_skip_recur_policy_conditions_actions_by_difficulty() -> None:
    difficulty = torch.tensor([0, 1, 2])
    actions = mixed_skip_recur_actions(
        num_internal_blocks=5,
        max_route_steps=4,
        batch_size=3,
        device=torch.device("cpu"),
        difficulty=difficulty,
    )
    paths = torch.stack(actions).T.tolist()

    assert paths[0] == [0, 2, 5, 5]
    assert paths[1] == [0, 1, 2, 5]
    assert paths[2] == [0, 0, 1, 5]


def test_pseudo_route_curriculum_report_uses_baseline_difficulty_bins(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3_pseudo"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_pseudo_skip_recur",
                "routing": {"mode": "pseudo", "pseudo_policy": "mixed_skip_recur"},
                "model_config_resolved": {
                    "architecture": "brian_route_core",
                    "route_pool_blocks": 5,
                    "max_route_steps": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "baseline_samples.jsonl"
    _write_jsonl(
        samples,
        [
            {"sample_id": 0, "baseline_cross_entropy": 1.0, "difficulty_bin": "easy"},
            {"sample_id": 1, "baseline_cross_entropy": 2.0, "difficulty_bin": "medium"},
            {"sample_id": 2, "baseline_cross_entropy": 3.0, "difficulty_bin": "hard"},
        ],
    )
    baseline_report = tmp_path / "baseline_difficulty_report.json"
    baseline_report.write_text(json.dumps({"samples_path": str(samples), "sample_count": 3}), encoding="utf-8")

    output = make_pseudo_route_curriculum_report(
        run_dir,
        baseline_difficulty_report_path=baseline_report,
        output_path=tmp_path / "curriculum.json",
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["stage"] == "stage3_pseudo_skip_recur"
    assert report["routing_mode"] == "pseudo"
    assert report["checks"]["stage3_pseudo_skip_recur_stage"] is True
    assert report["checks"]["pseudo_routing_mode"] is True
    assert report["checks"]["baseline_cross_entropy_numeric"] is True
    assert report["checks"]["baseline_cross_entropy_ordered_by_difficulty"] is True
    assert report["checks"]["easy_has_skip_or_small_pool"] is True
    assert report["checks"]["hard_has_recur_transition"] is True
    assert report["checks"]["exit_action_supervised"] is True
    assert report["by_difficulty"]["easy"]["mean_internal_route_steps"] == 2.0
    assert report["by_difficulty"]["hard"]["recur_transition_count"] == 1


def test_pseudo_route_curriculum_report_fails_boolean_baseline_ce(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3_pseudo"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_pseudo_skip_recur",
                "routing": {"mode": "pseudo", "pseudo_policy": "mixed_skip_recur"},
                "model_config_resolved": {
                    "architecture": "brian_route_core",
                    "route_pool_blocks": 5,
                    "max_route_steps": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "baseline_samples.jsonl"
    _write_jsonl(
        samples,
        [
            {"sample_id": 0, "baseline_cross_entropy": True, "difficulty_bin": "easy"},
            {"sample_id": 1, "baseline_cross_entropy": False, "difficulty_bin": "medium"},
            {"sample_id": 2, "baseline_cross_entropy": True, "difficulty_bin": "hard"},
        ],
    )
    baseline_report = tmp_path / "baseline_difficulty_report.json"
    baseline_report.write_text(json.dumps({"samples_path": str(samples), "sample_count": 3}), encoding="utf-8")

    output = make_pseudo_route_curriculum_report(
        run_dir,
        baseline_difficulty_report_path=baseline_report,
        output_path=tmp_path / "curriculum.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["baseline_cross_entropy_numeric"] is False
    assert report["by_difficulty"]["easy"]["mean_baseline_cross_entropy"] is None
    assert report["by_difficulty"]["medium"]["mean_baseline_cross_entropy"] is None
    assert report["by_difficulty"]["hard"]["mean_baseline_cross_entropy"] is None


def test_pseudo_route_curriculum_report_fails_unordered_baseline_ce(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3_pseudo"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_pseudo_skip_recur",
                "routing": {"mode": "pseudo", "pseudo_policy": "mixed_skip_recur"},
                "model_config_resolved": {
                    "architecture": "brian_route_core",
                    "route_pool_blocks": 5,
                    "max_route_steps": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "baseline_samples.jsonl"
    _write_jsonl(
        samples,
        [
            {"sample_id": 0, "baseline_cross_entropy": 3.0, "difficulty_bin": "easy"},
            {"sample_id": 1, "baseline_cross_entropy": 2.0, "difficulty_bin": "medium"},
            {"sample_id": 2, "baseline_cross_entropy": 1.0, "difficulty_bin": "hard"},
        ],
    )
    baseline_report = tmp_path / "baseline_difficulty_report.json"
    baseline_report.write_text(json.dumps({"samples_path": str(samples), "sample_count": 3}), encoding="utf-8")

    output = make_pseudo_route_curriculum_report(
        run_dir,
        baseline_difficulty_report_path=baseline_report,
        output_path=tmp_path / "curriculum.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["baseline_cross_entropy_numeric"] is True
    assert report["checks"]["baseline_cross_entropy_ordered_by_difficulty"] is False


def test_pseudo_route_curriculum_report_fails_non_pseudo_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage3_scheduled"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "stage3_scheduled_free_routing",
                "routing": {"mode": "scheduled", "pseudo_policy": "mixed_skip_recur"},
                "model_config_resolved": {
                    "architecture": "brian_route_core",
                    "route_pool_blocks": 5,
                    "max_route_steps": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    samples = tmp_path / "baseline_samples.jsonl"
    _write_jsonl(
        samples,
        [
            {"sample_id": 0, "baseline_cross_entropy": 1.0, "difficulty_bin": "easy"},
            {"sample_id": 1, "baseline_cross_entropy": 2.0, "difficulty_bin": "medium"},
            {"sample_id": 2, "baseline_cross_entropy": 3.0, "difficulty_bin": "hard"},
        ],
    )
    baseline_report = tmp_path / "baseline_difficulty_report.json"
    baseline_report.write_text(json.dumps({"samples_path": str(samples), "sample_count": 3}), encoding="utf-8")

    output = make_pseudo_route_curriculum_report(
        run_dir,
        baseline_difficulty_report_path=baseline_report,
        output_path=tmp_path / "curriculum.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["pseudo_policy"] == "mixed_skip_recur"
    assert report["checks"]["mixed_skip_recur_policy"] is True
    assert report["checks"]["stage3_pseudo_skip_recur_stage"] is False
    assert report["checks"]["pseudo_routing_mode"] is False


def test_pseudo_route_curriculum_eval_config_resolves() -> None:
    cfg = load_config("configs/eval/pseudo_route_curriculum.yaml")
    assert cfg["eval_name"] == "pseudo_route_curriculum_report"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
