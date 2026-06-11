import json
from pathlib import Path

from brian_sphere_llm.eval.go_no_go_report import make_go_no_go_report


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _passing_stage_gate() -> dict:
    return {
        "gates": {
            "stage1_to_2": {
                "status": "pass",
                "loss_ratio": 1.01,
                "checks": {"loss_within_1_to_3_percent": True},
            },
            "stage2_to_3": {
                "status": "pass",
                "checks": {
                    "route_imitation_accuracy": True,
                    "block_usage_non_degenerate": True,
                    "block_load_entropy_present": True,
                },
            },
            "stage3_to_4": {
                "status": "pass",
                "loss_ratio_vs_stage1": 1.0,
                "checks": {
                    "validation_loss_not_collapsed": True,
                    "block_load_entropy_present": True,
                    "route_path_diversity_present": True,
                },
            },
            "stage4_to_5": {
                "status": "pass",
                "first_exit_step_histogram": {"1": 1, "2": 2},
                "cost_control_status": "pass",
                "cost_control_active_block_evals_range": 0.5,
                "checks": {
                    "cost_control_report_present": True,
                    "cost_control_active_range_present": True,
                    "cost_control_active_not_increasing": True,
                    "cost_control_output_not_decreasing": True,
                    "exit_distribution_present": True,
                    "not_all_immediate_exit": True,
                },
            },
        },
        "runs": [{"stage": "stage3_scheduled_free_routing", "difficulty_step_correlation": 0.25}],
    }


def test_go_no_go_r125_passes_with_required_evidence(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    position = _write_json(tmp_path / "position.json", {"overall_status": "pass", "candidate_count": 3})
    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    phase = report["phases"]["r125_to_r350"]
    assert report["overall_status"] == "pass"
    assert phase["recommendation"] == "proceed"
    assert all(item["status"] == "pass" for item in phase["criteria"])


def test_go_no_go_r125_fails_and_marks_missing_evidence(tmp_path: Path) -> None:
    data = _passing_stage_gate()
    data["gates"]["stage4_to_5"]["checks"]["not_all_immediate_exit"] = False
    stage_gate = _write_json(tmp_path / "stage_gate.json", data)
    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}
    assert report["overall_status"] == "fail"
    assert report["recommendation"] == "stop"
    assert criteria["output_action_not_always_early_or_never_used"]["status"] == "fail"
    assert criteria["block_position_ablation_measurable_difference"]["status"] == "missing"


def test_go_no_go_r350_passes_with_compute_reasoning_and_memory_evidence(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {"run_dir": "baseline"},
                {
                    "run_dir": "routed",
                    "stage": "stage4_output_action",
                    "validation_loss": 9.9,
                    "baseline_comparison": {
                        "same_active_compute_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", {"overall_status": "pass"})
    long_context = _write_json(tmp_path / "long_context.json", {"overall_status": "pass", "candidate_count": 1})
    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        reasoning_baseline_report_path=reasoning_baseline,
        reasoning_candidate_report_paths=[reasoning_candidate],
        out_by_difficulty_report_path=out,
        long_context_compare_report_path=long_context,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    phase = report["phases"]["r350_to_1b"]
    assert report["overall_status"] == "pass"
    assert phase["recommendation"] == "proceed"
    assert all(item["status"] == "pass" for item in phase["criteria"])
