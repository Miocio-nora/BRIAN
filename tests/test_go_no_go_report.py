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
                    "cost_control_stage4_output_action_runs": True,
                    "cost_control_hard_exit_enabled": True,
                    "cost_control_active_range_present": True,
                    "cost_control_active_not_increasing": True,
                    "cost_control_average_steps_not_increasing": True,
                    "cost_control_output_not_decreasing": True,
                    "exit_distribution_present": True,
                    "not_all_immediate_exit": True,
                    "not_never_exit": True,
                },
            },
        },
        "runs": [{"stage": "stage3_scheduled_free_routing", "difficulty_step_correlation": 0.25}],
    }


def _controlled_memory_compare() -> dict:
    return {
        "overall_status": "pass",
        "candidate_count": 1,
        "comparisons": [
            {
                "candidate_report": "candidate_long_context.json",
                "candidate_run_dir": "r1b_candidate",
                "status": "pass",
                "memory_budget": {
                    "candidate": {
                        "estimated_global_cache_capacity_to_local_context_ratio": 0.25,
                    },
                },
                "checks": {
                    "baseline_stage4_output_action": True,
                    "baseline_scheduled_route_mode": True,
                    "baseline_local_kv": True,
                    "candidate_stage5_global_kv": True,
                    "candidate_scheduled_route_mode": True,
                    "candidate_global_kv_enabled": True,
                    "baseline_task_family_coverage": True,
                    "baseline_difficulty_coverage": True,
                    "candidate_task_family_coverage": True,
                    "candidate_difficulty_coverage": True,
                    "global_kv_active": True,
                    "quality_not_worse": True,
                    "memory_budget_present": True,
                    "global_budget_below_local_context": True,
                },
            }
        ],
    }


def _passing_out_by_difficulty() -> dict:
    return {
        "overall_status": "pass",
        "checks": {
            "stage4_output_action_reasoning": True,
            "hard_exit_reasoning": True,
            "easy_and_hard_present": True,
            "route_steps_non_decreasing_with_difficulty": True,
            "active_compute_non_decreasing_with_difficulty": True,
            "easy_output_probability_at_least_hard": True,
        },
        "deltas": {
            "hard_minus_easy_route_steps": 1.0,
            "hard_minus_easy_active_block_evals_per_token": 0.5,
            "easy_minus_hard_p_output": 0.2,
        },
    }


def _passing_position_ablation() -> dict:
    return {
        "overall_status": "pass",
        "candidate_count": 3,
        "checks": {
            "candidate_present": True,
            "any_measurable_difference": True,
            "reference_position_enabled": True,
            "no_position_candidate_present": True,
            "any_valid_no_position_measurable_difference": True,
        },
        "comparisons": [
            {
                "status": "pass",
                "checks": {
                    "reference_position_enabled": True,
                    "candidate_no_position_ablation": True,
                    "measurable_difference": True,
                },
                "validation_loss_delta": 0.01,
            }
        ],
    }


def test_go_no_go_r125_passes_with_required_evidence(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    position = _write_json(tmp_path / "position.json", _passing_position_ablation())
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


def test_go_no_go_r125_requires_average_steps_cost_control(tmp_path: Path) -> None:
    data = _passing_stage_gate()
    data["gates"]["stage4_to_5"]["checks"]["cost_control_average_steps_not_increasing"] = False
    stage_gate = _write_json(tmp_path / "stage_gate.json", data)
    position = _write_json(tmp_path / "position.json", _passing_position_ablation())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        output_path=tmp_path / "go_no_go.json",
        phase="r125_to_r350",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}

    assert report["overall_status"] == "fail"
    assert criteria["route_steps_controlled_by_cost_loss"]["status"] == "fail"
    assert criteria["route_steps_controlled_by_cost_loss"]["evidence"]["checks"][
        "cost_control_average_steps_not_increasing"
    ] is False


def test_go_no_go_r125_requires_stage4_hard_exit_cost_sweep(tmp_path: Path) -> None:
    data = _passing_stage_gate()
    data["gates"]["stage4_to_5"]["checks"]["cost_control_stage4_output_action_runs"] = False
    data["gates"]["stage4_to_5"]["checks"]["cost_control_hard_exit_enabled"] = False
    stage_gate = _write_json(tmp_path / "stage_gate.json", data)
    position = _write_json(tmp_path / "position.json", _passing_position_ablation())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        output_path=tmp_path / "go_no_go.json",
        phase="r125_to_r350",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}

    assert report["overall_status"] == "fail"
    assert criteria["route_steps_controlled_by_cost_loss"]["status"] == "fail"
    assert criteria["route_steps_controlled_by_cost_loss"]["evidence"]["checks"][
        "cost_control_stage4_output_action_runs"
    ] is False
    assert criteria["route_steps_controlled_by_cost_loss"]["evidence"]["checks"][
        "cost_control_hard_exit_enabled"
    ] is False


def test_go_no_go_r125_fails_when_output_action_never_exits(tmp_path: Path) -> None:
    data = _passing_stage_gate()
    data["gates"]["stage4_to_5"]["first_exit_step_histogram"] = {"0": 3}
    data["gates"]["stage4_to_5"]["checks"]["not_never_exit"] = False
    stage_gate = _write_json(tmp_path / "stage_gate.json", data)
    position = _write_json(tmp_path / "position.json", _passing_position_ablation())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}
    output_action = criteria["output_action_not_always_early_or_never_used"]

    assert report["overall_status"] == "fail"
    assert output_action["status"] == "fail"
    assert output_action["evidence"]["checks"]["not_never_exit"] is False


def test_go_no_go_r125_rejects_empty_passing_position_report(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    position = _write_json(tmp_path / "position.json", {"overall_status": "pass", "candidate_count": 3})

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}
    position_criterion = criteria["block_position_ablation_measurable_difference"]

    assert report["overall_status"] == "fail"
    assert position_criterion["status"] == "fail"
    assert position_criterion["evidence"]["checks"] is None


def test_go_no_go_r125_requires_no_position_ablation_role(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    position_data = _passing_position_ablation()
    position_data["checks"]["no_position_candidate_present"] = False
    position_data["checks"]["any_valid_no_position_measurable_difference"] = False
    position_data["comparisons"][0]["checks"]["candidate_no_position_ablation"] = False
    position = _write_json(tmp_path / "position.json", position_data)

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        position_ablation_report_path=position,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r125_to_r350"]["criteria"]}
    position_criterion = criteria["block_position_ablation_measurable_difference"]

    assert report["overall_status"] == "fail"
    assert position_criterion["status"] == "fail"
    assert position_criterion["passed"] is False
    assert position_criterion["evidence"]["checks"]["no_position_candidate_present"] is False


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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())
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


def test_go_no_go_r350_treats_absent_global_kv_evidence_as_not_tested(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        reasoning_baseline_report_path=reasoning_baseline,
        reasoning_candidate_report_paths=[reasoning_candidate],
        out_by_difficulty_report_path=out,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    global_kv = criteria["global_kv_long_context_benefit_if_tested"]

    assert report["overall_status"] == "pass"
    assert report["recommendation"] == "proceed"
    assert global_kv["status"] == "pass"
    assert global_kv["evidence"]["tested"] is False
    assert global_kv["evidence"]["optional"] is True


def test_go_no_go_r350_requires_all_compute_comparison_views(tmp_path: Path) -> None:
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
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    compute_criterion = criteria["same_active_compute_routed_not_worse_than_baseline"]

    assert report["overall_status"] == "fail"
    assert compute_criterion["status"] == "fail"
    comparison = compute_criterion["evidence"]["comparisons"][0]["baseline_comparison"]
    assert comparison["same_active_compute_view"] is True
    assert "same_parameter_count_view" not in comparison
    assert "similar_training_flops_view" not in comparison


def test_go_no_go_r350_rejects_empty_passing_long_context_report(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    long_context = _write_json(
        tmp_path / "long_context.json",
        {"overall_status": "pass", "candidate_count": 1, "comparisons": []},
    )

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    global_kv = criteria["global_kv_long_context_benefit_if_tested"]

    assert report["overall_status"] == "fail"
    assert global_kv["status"] == "fail"
    assert global_kv["evidence"]["long_context_compare"]["benefit_candidates"] == []


def test_go_no_go_r350_requires_stage5_long_context_roles(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    long_context_data = _controlled_memory_compare()
    long_context_data["comparisons"][0]["checks"]["candidate_stage5_global_kv"] = False
    long_context = _write_json(tmp_path / "long_context.json", long_context_data)

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    global_kv = criteria["global_kv_long_context_benefit_if_tested"]
    candidate = global_kv["evidence"]["long_context_compare"]["benefit_candidates"][0]

    assert report["overall_status"] == "fail"
    assert global_kv["status"] == "fail"
    assert candidate["role_checks"]["candidate_stage5_global_kv"] is False
    assert candidate["role_contract_passed"] is False
    assert candidate["passes_memory_quality_proxy"] is False


def test_go_no_go_r350_requires_long_context_coverage_contract(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    long_context_data = _controlled_memory_compare()
    long_context_data["comparisons"][0]["checks"]["candidate_task_family_coverage"] = False
    long_context = _write_json(tmp_path / "long_context.json", long_context_data)

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    global_kv = criteria["global_kv_long_context_benefit_if_tested"]
    candidate = global_kv["evidence"]["long_context_compare"]["benefit_candidates"][0]

    assert report["overall_status"] == "fail"
    assert global_kv["status"] == "fail"
    assert candidate["coverage_checks"]["candidate_task_family_coverage"] is False
    assert candidate["coverage_contract_passed"] is False
    assert candidate["passes_memory_quality_proxy"] is False


def test_go_no_go_r350_rejects_empty_passing_out_by_difficulty_report(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", {"overall_status": "pass"})
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    out_action = criteria["out_action_reduces_compute_on_easy_samples"]

    assert report["overall_status"] == "fail"
    assert out_action["status"] == "fail"
    assert out_action["evidence"]["checks"] is None


def test_go_no_go_r350_requires_stage4_hard_exit_out_by_difficulty(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out_data = _passing_out_by_difficulty()
    out_data["checks"]["stage4_output_action_reasoning"] = False
    out_data["checks"]["hard_exit_reasoning"] = False
    out = _write_json(tmp_path / "out.json", out_data)
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())

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
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    out_action = criteria["out_action_reduces_compute_on_easy_samples"]

    assert report["overall_status"] == "fail"
    assert out_action["status"] == "fail"
    assert out_action["evidence"]["checks"]["stage4_output_action_reasoning"] is False
    assert out_action["evidence"]["checks"]["hard_exit_reasoning"] is False


def test_go_no_go_r350_accepts_global_kv_ablation_memory_quality_evidence(tmp_path: Path) -> None:
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
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": -0.1,
                    },
                },
            ],
        },
    )
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": 0.2}})
    reasoning_candidate = _write_json(tmp_path / "reasoning_candidate.json", {"overall": {"exact_match_accuracy": 0.3}})
    out = _write_json(tmp_path / "out.json", _passing_out_by_difficulty())
    global_kv_ablation = _write_json(
        tmp_path / "global_kv_ablation.json",
        {
            "overall_status": "pass",
            "checks": {
                "long_context_quality_metrics_present": True,
                "memory_budget_metrics_present": True,
            },
            "comparisons": {
                "local_vs_global": [
                    {
                        "entry_id": "K4",
                        "entry_name": "global_kv_with_sink",
                        "run_dir": "global",
                        "global_cache_capacity_ratio": 0.25,
                        "exact_match_delta_vs_local": 0.0,
                        "teacher_forced_token_accuracy_delta_vs_local": 0.02,
                    }
                ]
            },
        },
    )
    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        reasoning_baseline_report_path=reasoning_baseline,
        reasoning_candidate_report_paths=[reasoning_candidate],
        out_by_difficulty_report_path=out,
        global_kv_ablation_report_path=global_kv_ablation,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    assert report["overall_status"] == "pass"
    assert criteria["global_kv_long_context_benefit_if_tested"]["status"] == "pass"
    assert criteria["global_kv_long_context_benefit_if_tested"]["evidence"]["global_kv_ablation"][
        "benefit_candidates"
    ][0]["passes_memory_quality_proxy"] is True


def test_go_no_go_r350_difficulty_uses_any_positive_correlation(tmp_path: Path) -> None:
    stage_gate_data = _passing_stage_gate()
    stage_gate_data["runs"] = [
        {"run_dir": "stage2", "stage": "stage2_router_imitation", "difficulty_step_correlation": -0.5},
        {"run_dir": "stage3", "stage": "stage3_scheduled_free_routing", "difficulty_step_correlation": 0.25},
    ]
    stage_gate = _write_json(tmp_path / "stage_gate.json", stage_gate_data)

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
        min_difficulty_step_correlation=0.0,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    difficulty = criteria["difficulty_step_correlation_positive"]

    assert difficulty["status"] == "pass"
    assert [row["difficulty_step_correlation"] for row in difficulty["evidence"]["runs"]] == [-0.5, 0.25]
    assert difficulty["evidence"]["runs"][1]["stage"] == "stage3_scheduled_free_routing"


def test_go_no_go_r350_rejects_boolean_difficulty_correlation(tmp_path: Path) -> None:
    stage_gate_data = _passing_stage_gate()
    stage_gate_data["runs"] = [
        {"run_dir": "stage3", "stage": "stage3_scheduled_free_routing", "difficulty_step_correlation": True}
    ]
    stage_gate = _write_json(tmp_path / "stage_gate.json", stage_gate_data)

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    difficulty = criteria["difficulty_step_correlation_positive"]

    assert difficulty["status"] == "missing"
    assert difficulty["evidence"]["runs"] == []


def test_go_no_go_r350_rejects_boolean_compute_loss_delta(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {"run_dir": "baseline", "validation_loss": 10.0},
                {
                    "run_dir": "candidate",
                    "validation_loss": 10.0,
                    "baseline_comparison": {
                        "same_parameter_count_view": True,
                        "same_active_compute_view": True,
                        "similar_training_flops_view": True,
                        "validation_loss_delta": False,
                    },
                },
            ],
        },
    )

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}

    assert criteria["same_active_compute_routed_not_worse_than_baseline"]["status"] == "fail"


def test_go_no_go_r350_rejects_boolean_reasoning_scores(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    reasoning_baseline = _write_json(tmp_path / "reasoning_base.json", {"overall": {"exact_match_accuracy": False}})
    reasoning_candidate = _write_json(
        tmp_path / "reasoning_candidate.json",
        {"overall": {"exact_match_accuracy": True}},
    )

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        reasoning_baseline_report_path=reasoning_baseline,
        reasoning_candidate_report_paths=[reasoning_candidate],
        phase="r350_to_1b",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r350_to_1b"]["criteria"]}
    reasoning = criteria["reasoning_or_synthetic_multistep_improves"]

    assert reasoning["status"] == "missing"
    assert reasoning["evidence"]["baseline_score"] is None
    assert reasoning["evidence"]["candidate_scores"] == [None]


def test_go_no_go_includes_parallel_compare_as_optional_evidence(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    parallel = _write_json(
        tmp_path / "parallel.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "comparisons": [
                {
                    "candidate_run": "parallel",
                    "status": "pass",
                    "checks": {"parallel_branch_benefit_proxy": True},
                    "parallel": {"parallel_branch_count_mean": 2.0},
                    "baseline_comparison": {"validation_loss_delta": -0.1},
                }
            ],
        },
    )

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        parallel_compare_report_path=parallel,
        phase="r125_to_r350",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    evidence = report["optional_evidence"]["parallel_compare"]
    assert evidence["overall_status"] == "pass"
    assert evidence["candidate_count"] == 1
    assert evidence["comparisons"][0]["checks"]["parallel_branch_benefit_proxy"] is True


def test_go_no_go_r1b_success_passes_with_compute_adjusted_advantage(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {
                    "run_dir": "baseline",
                    "validation_loss": 10.0,
                    "inference_latency_ms_per_token_latest": 1.0,
                },
                {
                    "run_dir": "r1b_candidate",
                    "stage": "stage6_parallel_passing",
                    "validation_loss": 10.2,
                    "inference_latency_ms_per_token_latest": 1.5,
                    "baseline_comparison": {
                        "estimated_flops_per_token_ratio": 0.9,
                        "inference_latency_ms_per_token_ratio": 1.5,
                        "validation_loss_delta": 0.2,
                    },
                },
            ],
        },
    )
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        long_context_compare_report_path=long_context,
        phase="r1b_success",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    phase = report["phases"]["r1b_success"]
    criteria = {item["name"]: item for item in phase["criteria"]}

    assert report["overall_status"] == "pass"
    assert criteria["routing_does_not_collapse"]["status"] == "pass"
    assert criteria["compute_adjusted_eval_present"]["status"] == "pass"
    assert criteria["kv_memory_remains_controlled"]["status"] == "pass"
    assert criteria["inference_latency_remains_acceptable"]["status"] == "pass"
    assert criteria["at_least_one_core_advantage_stable"]["status"] == "pass"
    memory = criteria["kv_memory_remains_controlled"]["evidence"]["long_context_compare"]["memory_candidates"]
    latency = criteria["inference_latency_remains_acceptable"]["evidence"]["candidates"]
    adjusted = criteria["at_least_one_core_advantage_stable"]["evidence"]["better_compute_adjusted_perplexity"]
    assert memory[0]["global_cache_capacity_ratio"] == 0.25
    assert latency[0]["inference_latency_ms_per_token_ratio"] == 1.5
    assert adjusted["passed"] is True
    assert adjusted["evidence"]["candidates"][0]["compute_adjusted_loss_delta"] < 0.0


def test_go_no_go_r1b_success_fails_explicit_routing_collapse(tmp_path: Path) -> None:
    stage_gate_data = _passing_stage_gate()
    stage_gate_data["gates"]["stage2_to_3"]["checks"]["block_usage_non_degenerate"] = False
    stage_gate = _write_json(tmp_path / "stage_gate.json", stage_gate_data)
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {
                    "run_dir": "baseline",
                    "validation_loss": 10.0,
                    "inference_latency_ms_per_token_latest": 1.0,
                },
                {
                    "run_dir": "r1b_candidate",
                    "validation_loss": 10.2,
                    "inference_latency_ms_per_token_latest": 1.5,
                    "baseline_comparison": {
                        "estimated_flops_per_token_ratio": 0.9,
                        "inference_latency_ms_per_token_ratio": 1.5,
                    },
                },
            ],
        },
    )
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        long_context_compare_report_path=long_context,
        phase="r1b_success",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r1b_success"]["criteria"]}

    assert report["overall_status"] == "fail"
    assert criteria["routing_does_not_collapse"]["status"] == "fail"


def test_go_no_go_r1b_success_requires_stage5_long_context_memory_roles(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {
                    "run_dir": "baseline",
                    "validation_loss": 10.0,
                    "inference_latency_ms_per_token_latest": 1.0,
                },
                {
                    "run_dir": "r1b_candidate",
                    "stage": "stage6_parallel_passing",
                    "validation_loss": 10.2,
                    "inference_latency_ms_per_token_latest": 1.5,
                    "baseline_comparison": {
                        "estimated_flops_per_token_ratio": 0.9,
                        "inference_latency_ms_per_token_ratio": 1.5,
                        "validation_loss_delta": 0.2,
                    },
                },
            ],
        },
    )
    long_context_data = _controlled_memory_compare()
    long_context_data["comparisons"][0]["checks"]["baseline_local_kv"] = False
    long_context = _write_json(tmp_path / "long_context.json", long_context_data)

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        long_context_compare_report_path=long_context,
        phase="r1b_success",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r1b_success"]["criteria"]}
    memory = criteria["kv_memory_remains_controlled"]["evidence"]["long_context_compare"]["memory_candidates"][0]

    assert report["overall_status"] == "fail"
    assert criteria["kv_memory_remains_controlled"]["status"] == "fail"
    assert memory["role_checks"]["baseline_local_kv"] is False
    assert memory["role_contract_passed"] is False
    assert memory["passes_memory_control_proxy"] is False


def test_go_no_go_r1b_success_requires_long_context_memory_coverage(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {
                    "run_dir": "baseline",
                    "validation_loss": 10.0,
                    "inference_latency_ms_per_token_latest": 1.0,
                },
                {
                    "run_dir": "r1b_candidate",
                    "stage": "stage6_parallel_passing",
                    "validation_loss": 10.2,
                    "inference_latency_ms_per_token_latest": 1.5,
                    "baseline_comparison": {
                        "estimated_flops_per_token_ratio": 0.9,
                        "inference_latency_ms_per_token_ratio": 1.5,
                        "validation_loss_delta": 0.2,
                    },
                },
            ],
        },
    )
    long_context_data = _controlled_memory_compare()
    long_context_data["comparisons"][0]["checks"]["candidate_difficulty_coverage"] = False
    long_context = _write_json(tmp_path / "long_context.json", long_context_data)

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        long_context_compare_report_path=long_context,
        phase="r1b_success",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r1b_success"]["criteria"]}
    memory = criteria["kv_memory_remains_controlled"]["evidence"]["long_context_compare"]["memory_candidates"][0]

    assert report["overall_status"] == "fail"
    assert criteria["kv_memory_remains_controlled"]["status"] == "fail"
    assert memory["coverage_checks"]["candidate_difficulty_coverage"] is False
    assert memory["coverage_contract_passed"] is False
    assert memory["passes_memory_control_proxy"] is False


def test_go_no_go_r1b_success_accepts_less_visible_cot_advantage(tmp_path: Path) -> None:
    stage_gate = _write_json(tmp_path / "stage_gate.json", _passing_stage_gate())
    compute = _write_json(
        tmp_path / "compute.json",
        {
            "run_count": 2,
            "baseline_run": "baseline",
            "runs": [
                {
                    "run_dir": "baseline",
                    "validation_loss": 10.0,
                    "inference_latency_ms_per_token_latest": 1.0,
                },
                {
                    "run_dir": "r1b_candidate",
                    "validation_loss": 10.1,
                    "inference_latency_ms_per_token_latest": 1.1,
                    "baseline_comparison": {
                        "estimated_flops_per_token_ratio": 1.0,
                        "inference_latency_ms_per_token_ratio": 1.1,
                    },
                },
            ],
        },
    )
    long_context = _write_json(tmp_path / "long_context.json", _controlled_memory_compare())
    reasoning_baseline = _write_json(
        tmp_path / "reasoning_base.json",
        {"overall": {"exact_match_accuracy": 0.6, "visible_cot_tokens_mean": 40.0}},
    )
    reasoning_candidate = _write_json(
        tmp_path / "reasoning_candidate.json",
        {"overall": {"exact_match_accuracy": 0.6, "visible_cot_tokens_mean": 20.0}},
    )

    output = make_go_no_go_report(
        stage_gate_report_path=stage_gate,
        compute_report_path=compute,
        long_context_compare_report_path=long_context,
        reasoning_baseline_report_path=reasoning_baseline,
        reasoning_candidate_report_paths=[reasoning_candidate],
        phase="r1b_success",
        output_path=tmp_path / "go.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    criteria = {item["name"]: item for item in report["phases"]["r1b_success"]["criteria"]}
    cot_advantage = criteria["at_least_one_core_advantage_stable"]["evidence"][
        "less_visible_cot_for_similar_reasoning"
    ]

    assert report["overall_status"] == "pass"
    assert criteria["kv_memory_remains_controlled"]["status"] == "pass"
    assert criteria["inference_latency_remains_acceptable"]["status"] == "pass"
    assert cot_advantage["passed"] is True
    assert cot_advantage["evidence"]["candidate_comparisons"][0]["visible_cot_token_delta"] == -20.0
