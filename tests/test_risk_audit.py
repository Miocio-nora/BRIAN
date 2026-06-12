import json
from pathlib import Path

from brian_sphere_llm.eval.risk_audit import make_risk_audit_report
from brian_sphere_llm.utils.config import load_config


def test_risk_audit_passes_with_clear_evidence(tmp_path: Path) -> None:
    routing = _write_json(
        tmp_path / "routing.json",
        {
            "summary": {
                "block_load_entropy_normalized": 0.5,
                "route_path_diversity": 0.5,
                "p_output_mean": 0.25,
                "location_distance_mean": 0.2,
                "global_attention_mass": 0.02,
            },
            "latest_first_exit_step_histogram": {"2": 8, "3": 2},
            "latest_position_norm_trajectory": [0.1, 0.2, 0.35],
            "latest_location_distance_trajectory": [0.05, 0.2, 0.4],
        },
    )
    stage_gate = _write_json(
        tmp_path / "stage_gate.json",
        {
            "gates": {
                "stage1_to_2": {"status": "pass", "checks": {"loss_within_1_to_3_percent": True}},
                "stage2_to_3": {"status": "pass", "checks": {"block_usage_non_degenerate": True}},
                "stage3_to_4": {
                    "status": "pass",
                    "checks": {
                        "validation_loss_not_collapsed": True,
                        "route_path_diversity_present": True,
                    },
                    "loss_ratio_vs_stage1": 1.01,
                },
                "stage4_to_5": {
                    "status": "pass",
                    "checks": {"not_all_immediate_exit": True, "not_never_exit": True},
                    "first_exit_step_histogram": {"2": 8, "3": 2},
                },
                "stage5_to_6": {"status": "pass", "checks": {"global_attention_mass_nonzero": True}},
            }
        },
    )
    position = _write_json(
        tmp_path / "position.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "checks": {
                "candidate_present": True,
                "any_measurable_difference": True,
                "reference_position_enabled": True,
                "no_position_candidate_present": True,
                "any_valid_no_position_measurable_difference": True,
            },
        },
    )
    retention = _write_json(
        tmp_path / "retention.json",
        {
            "overall_status": "pass",
            "metrics": {"global_attention_mass": 0.02},
            "checks": {"global_attention_mass_nonzero": True},
        },
    )
    long_context = _write_json(
        tmp_path / "long_context_compare.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "comparisons": [
                {
                    "candidate_report": "global.json",
                    "candidate_run_dir": "global",
                    "status": "pass",
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
        },
    )
    global_ablation = _write_json(
        tmp_path / "global_ablation.json",
        {
            "comparisons": {
                "local_vs_global": [
                    {
                        "validation_loss_delta_vs_local": -0.1,
                        "exact_match_delta_vs_local": 0.05,
                        "teacher_forced_token_accuracy_delta_vs_local": 0.02,
                    }
                ]
            }
        },
    )
    parallel_passing = _write_json(
        tmp_path / "parallel_passing.json",
        {
            "checks": {
                "branch_count_bounded_by_beam": True,
                "delta_cache_bounded_by_window": True,
                "score_margin_measured": True,
                "branch_score_decay_configured": True,
            },
            "model": {"beam_size": 2},
            "routing": {"parallel_branch_count": {"max": 2}},
        },
    )
    parallel_compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "pass",
            "comparisons": [
                {
                    "candidate_run": "parallel",
                    "status": "pass",
                    "checks": {
                        "baseline_stage5_global_kv": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_global_kv_enabled": True,
                        "baseline_parallel_passing_disabled": True,
                        "baseline_topk_weighted_fusion": True,
                        "candidate_parallel_stage": True,
                        "candidate_parallel_route_mode": True,
                        "candidate_parallel_passing_enabled": True,
                        "candidate_global_kv_enabled": True,
                        "parallel_branch_benefit_proxy": True,
                    },
                }
            ],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        stage_gate_report_path=stage_gate,
        routing_report_path=routing,
        position_ablation_report_path=position,
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
        global_kv_ablation_report_path=global_ablation,
        parallel_passing_report_path=parallel_passing,
        parallel_compare_report_path=parallel_compare,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert all(risk["status"] == "pass" for risk in report["risks"].values())
    assert report["risks"]["router_collapse"]["symptoms"][0]["status"] == "clear"


def test_risk_audit_flags_triggered_symptoms(tmp_path: Path) -> None:
    routing = _write_json(
        tmp_path / "routing.json",
        {
            "summary": {
                "block_load_entropy_normalized": 0.0,
                "route_path_diversity": 0.0,
                "p_output_mean": 0.5,
                "location_distance_mean": 0.0,
                "global_attention_mass": 0.0,
            },
            "latest_first_exit_step_histogram": {"1": 10},
            "latest_position_norm_trajectory": [0.1, 0.1, 0.1],
            "latest_location_distance_trajectory": [0.0, 0.0, 0.0],
        },
    )
    stage_gate = _write_json(
        tmp_path / "stage_gate.json",
        {
            "gates": {
                "stage1_to_2": {"status": "pass", "checks": {"loss_within_1_to_3_percent": True}},
                "stage3_to_4": {
                    "status": "fail",
                    "checks": {"validation_loss_not_collapsed": False},
                    "loss_ratio_vs_stage1": 1.5,
                },
                "stage4_to_5": {
                    "status": "fail",
                    "checks": {"not_all_immediate_exit": False, "not_never_exit": True},
                    "first_exit_step_histogram": {"1": 10},
                },
                "stage5_to_6": {"status": "fail", "checks": {"global_attention_mass_nonzero": False}},
            }
        },
    )
    position = _write_json(
        tmp_path / "position.json",
        {
            "overall_status": "fail",
            "candidate_count": 1,
            "checks": {
                "candidate_present": True,
                "any_measurable_difference": False,
                "reference_position_enabled": True,
                "no_position_candidate_present": True,
                "any_valid_no_position_measurable_difference": False,
            },
        },
    )
    retention = _write_json(
        tmp_path / "retention.json",
        {
            "overall_status": "fail",
            "metrics": {"global_attention_mass": 0.0},
            "checks": {"global_attention_mass_nonzero": False},
        },
    )
    long_context = _write_json(
        tmp_path / "long_context_compare.json",
        {"overall_status": "fail", "candidate_count": 1, "comparisons": [{"status": "fail"}]},
    )
    global_ablation = _write_json(
        tmp_path / "global_ablation.json",
        {
            "comparisons": {
                "local_vs_global": [
                    {
                        "validation_loss_delta_vs_local": 0.1,
                        "exact_match_delta_vs_local": 0.0,
                        "teacher_forced_token_accuracy_delta_vs_local": 0.0,
                    }
                ]
            }
        },
    )
    parallel_passing = _write_json(
        tmp_path / "parallel_passing.json",
        {
            "overall_status": "fail",
            "checks": {
                "branch_count_bounded_by_beam": False,
                "delta_cache_bounded_by_window": False,
                "score_margin_measured": False,
                "branch_score_decay_configured": True,
            },
        },
    )
    parallel_compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "fail",
            "comparisons": [{"status": "fail", "checks": {"parallel_branch_benefit_proxy": False}}],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        stage_gate_report_path=stage_gate,
        routing_report_path=routing,
        position_ablation_report_path=position,
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
        global_kv_ablation_report_path=global_ablation,
        parallel_passing_report_path=parallel_passing,
        parallel_compare_report_path=parallel_compare,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall_status"] == "fail"
    assert report["risks"]["router_collapse"]["status"] == "fail"
    assert _symptom(report, "router_collapse", "always_selects_same_block")["triggered"] is True
    assert _symptom(report, "block_position_state_no_effect", "no_position_ablation_equals_main_model")[
        "triggered"
    ] is True
    assert _symptom(report, "free_routing_lm_loss_degrades", "fixed_route_works_but_free_route_validation_loss_spikes")[
        "triggered"
    ] is True
    assert _symptom(report, "global_kv_noise", "global_cache_worsens_loss")["triggered"] is True
    assert _symptom(report, "parallel_passing_cost_explosion", "branch_count_grows")["triggered"] is True


def test_risk_audit_flags_never_exit_from_stage_gate_check(tmp_path: Path) -> None:
    routing = _write_json(
        tmp_path / "routing.json",
        {
            "summary": {
                "block_load_entropy_normalized": 0.5,
                "route_path_diversity": 0.5,
                "p_output_mean": 0.5,
            }
        },
    )
    stage_gate = _write_json(
        tmp_path / "stage_gate.json",
        {
            "gates": {
                "stage4_to_5": {
                    "status": "fail",
                    "checks": {"not_all_immediate_exit": True, "not_never_exit": False},
                }
            }
        },
    )
    position = _write_json(
        tmp_path / "position.json",
        {
            "overall_status": "pass",
            "checks": {
                "candidate_present": True,
                "any_measurable_difference": True,
                "reference_position_enabled": True,
                "no_position_candidate_present": True,
                "any_valid_no_position_measurable_difference": True,
            },
        },
    )
    retention = _write_json(tmp_path / "retention.json", {"overall_status": "pass", "checks": {}})
    long_context = _write_json(tmp_path / "long_context.json", {"overall_status": "pass", "comparisons": []})
    global_ablation = _write_json(tmp_path / "global_ablation.json", {"comparisons": {"local_vs_global": []}})
    parallel_passing = _write_json(tmp_path / "parallel_passing.json", {"checks": {}})
    parallel_compare = _write_json(tmp_path / "parallel_compare.json", {"comparisons": []})

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        stage_gate_report_path=stage_gate,
        routing_report_path=routing,
        position_ablation_report_path=position,
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
        global_kv_ablation_report_path=global_ablation,
        parallel_passing_report_path=parallel_passing,
        parallel_compare_report_path=parallel_compare,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    never_exits = _symptom(report, "router_collapse", "never_exits")

    assert never_exits["triggered"] is True
    assert never_exits["evidence"]["stage4_not_never_exit"] is False


def test_risk_audit_treats_wrong_position_ablation_role_as_missing(tmp_path: Path) -> None:
    position = _write_json(
        tmp_path / "position.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "checks": {
                "candidate_present": True,
                "any_measurable_difference": True,
                "reference_position_enabled": True,
                "no_position_candidate_present": False,
                "any_valid_no_position_measurable_difference": False,
            },
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        position_ablation_report_path=position,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    symptom = _symptom(report, "block_position_state_no_effect", "no_position_ablation_equals_main_model")

    assert symptom["status"] == "missing"
    assert symptom["triggered"] is None


def test_risk_audit_treats_legacy_position_ablation_report_as_missing(tmp_path: Path) -> None:
    position = _write_json(
        tmp_path / "position.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "checks": {"candidate_present": True, "any_measurable_difference": True},
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        position_ablation_report_path=position,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    symptom = _symptom(report, "block_position_state_no_effect", "no_position_ablation_equals_main_model")

    assert symptom["status"] == "missing"
    assert symptom["triggered"] is None


def test_risk_audit_rejects_boolean_numeric_evidence(tmp_path: Path) -> None:
    routing = _write_json(
        tmp_path / "routing.json",
        {
            "summary": {
                "block_load_entropy_normalized": True,
                "route_path_diversity": True,
                "p_output_mean": False,
                "location_distance_mean": True,
            },
            "latest_position_norm_trajectory": [True, False],
            "latest_location_distance_trajectory": [True, False],
        },
    )

    output = make_risk_audit_report(output_path=tmp_path / "risk.json", routing_report_path=routing)
    report = json.loads(output.read_text(encoding="utf-8"))

    same_block = _symptom(report, "router_collapse", "always_selects_same_block")
    never_exits = _symptom(report, "router_collapse", "never_exits")
    constant_position = _symptom(report, "block_position_state_no_effect", "position_state_becomes_constant")
    location_structure = _symptom(report, "block_position_state_no_effect", "location_distance_has_no_structure")

    assert same_block["status"] == "missing"
    assert same_block["evidence"]["block_load_entropy_normalized"] is None
    assert never_exits["status"] == "missing"
    assert never_exits["evidence"]["p_output_mean"] is None
    assert constant_position["status"] == "missing"
    assert constant_position["evidence"]["position_norm_trajectory_count"] == 0
    assert location_structure["status"] == "missing"
    assert location_structure["evidence"]["location_distance_mean"] is None


def test_risk_audit_requires_parallel_compare_stage_roles(tmp_path: Path) -> None:
    passing = _write_json(
        tmp_path / "parallel_passing.json",
        {
            "checks": {
                "branch_count_bounded_by_beam": True,
                "delta_cache_bounded_by_window": True,
                "score_margin_measured": True,
                "branch_score_decay_configured": True,
            },
        },
    )
    compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "pass",
            "comparisons": [
                {
                    "candidate_run": "not_parallel",
                    "status": "pass",
                    "checks": {
                        "baseline_stage5_global_kv": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_global_kv_enabled": True,
                        "baseline_parallel_passing_disabled": True,
                        "baseline_topk_weighted_fusion": True,
                        "candidate_parallel_stage": False,
                        "candidate_parallel_route_mode": True,
                        "candidate_parallel_passing_enabled": True,
                        "candidate_global_kv_enabled": True,
                        "parallel_branch_benefit_proxy": True,
                    },
                }
            ],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        parallel_passing_report_path=passing,
        parallel_compare_report_path=compare,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    unstable = _symptom(report, "parallel_passing_cost_explosion", "branch_credit_assignment_unstable")
    candidate = unstable["evidence"]["parallel_compare_candidates"][0]

    assert report["risks"]["parallel_passing_cost_explosion"]["status"] == "fail"
    assert unstable["triggered"] is True
    assert candidate["role_checks"]["candidate_parallel_stage"] is False
    assert candidate["role_contract_passed"] is False
    assert candidate["passes_parallel_compare_contract"] is False


def test_risk_audit_requires_topk_weighted_parallel_baseline(tmp_path: Path) -> None:
    passing = _write_json(
        tmp_path / "parallel_passing.json",
        {
            "checks": {
                "branch_count_bounded_by_beam": True,
                "delta_cache_bounded_by_window": True,
                "score_margin_measured": True,
                "branch_score_decay_configured": True,
            },
        },
    )
    compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "pass",
            "comparisons": [
                {
                    "candidate_run": "parallel",
                    "status": "pass",
                    "checks": {
                        "baseline_stage5_global_kv": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_global_kv_enabled": True,
                        "baseline_parallel_passing_disabled": True,
                        "baseline_topk_weighted_fusion": False,
                        "candidate_parallel_stage": True,
                        "candidate_parallel_route_mode": True,
                        "candidate_parallel_passing_enabled": True,
                        "candidate_global_kv_enabled": True,
                        "parallel_branch_benefit_proxy": True,
                    },
                }
            ],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        parallel_passing_report_path=passing,
        parallel_compare_report_path=compare,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    unstable = _symptom(report, "parallel_passing_cost_explosion", "branch_credit_assignment_unstable")
    candidate = unstable["evidence"]["parallel_compare_candidates"][0]

    assert report["risks"]["parallel_passing_cost_explosion"]["status"] == "fail"
    assert unstable["triggered"] is True
    assert candidate["role_checks"]["baseline_topk_weighted_fusion"] is False
    assert candidate["role_contract_passed"] is False
    assert candidate["passes_parallel_compare_contract"] is False


def test_risk_audit_requires_parallel_branch_score_decay(tmp_path: Path) -> None:
    passing = _write_json(
        tmp_path / "parallel_passing.json",
        {
            "checks": {
                "branch_count_bounded_by_beam": True,
                "delta_cache_bounded_by_window": True,
                "score_margin_measured": True,
                "branch_score_decay_configured": False,
            },
        },
    )
    compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "pass",
            "comparisons": [
                {
                    "candidate_run": "parallel",
                    "status": "pass",
                    "checks": {
                        "baseline_stage5_global_kv": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_global_kv_enabled": True,
                        "baseline_parallel_passing_disabled": True,
                        "baseline_topk_weighted_fusion": True,
                        "candidate_parallel_stage": True,
                        "candidate_parallel_route_mode": True,
                        "candidate_parallel_passing_enabled": True,
                        "candidate_global_kv_enabled": True,
                        "parallel_branch_benefit_proxy": True,
                    },
                }
            ],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        parallel_passing_report_path=passing,
        parallel_compare_report_path=compare,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    unstable = _symptom(report, "parallel_passing_cost_explosion", "branch_credit_assignment_unstable")

    assert report["risks"]["parallel_passing_cost_explosion"]["status"] == "fail"
    assert unstable["triggered"] is True
    assert unstable["evidence"]["branch_score_decay_configured"] is False


def test_risk_audit_requires_stage5_long_context_roles(tmp_path: Path) -> None:
    retention = _write_json(
        tmp_path / "retention.json",
        {
            "overall_status": "pass",
            "metrics": {"global_attention_mass": 0.02},
            "checks": {"global_attention_mass_nonzero": True},
        },
    )
    long_context = _write_json(
        tmp_path / "long_context_compare.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "comparisons": [
                {
                    "candidate_report": "wrong.json",
                    "candidate_run_dir": "wrong",
                    "status": "pass",
                    "checks": {
                        "baseline_stage4_output_action": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_local_kv": True,
                        "candidate_stage5_global_kv": False,
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
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    no_difference = _symptom(report, "global_kv_noise", "global_on_off_no_difference")
    candidate = no_difference["evidence"]["comparison_candidates"][0]

    assert report["risks"]["global_kv_noise"]["status"] == "fail"
    assert no_difference["triggered"] is True
    assert candidate["role_checks"]["candidate_stage5_global_kv"] is False
    assert candidate["role_contract_passed"] is False
    assert candidate["passes_stage5_long_context_contract"] is False


def test_risk_audit_requires_stage5_long_context_coverage(tmp_path: Path) -> None:
    retention = _write_json(
        tmp_path / "retention.json",
        {
            "overall_status": "pass",
            "metrics": {"global_attention_mass": 0.02},
            "checks": {"global_attention_mass_nonzero": True},
        },
    )
    long_context = _write_json(
        tmp_path / "long_context_compare.json",
        {
            "overall_status": "pass",
            "candidate_count": 1,
            "comparisons": [
                {
                    "candidate_report": "global.json",
                    "candidate_run_dir": "global",
                    "status": "pass",
                    "checks": {
                        "baseline_stage4_output_action": True,
                        "baseline_scheduled_route_mode": True,
                        "baseline_local_kv": True,
                        "candidate_stage5_global_kv": True,
                        "candidate_scheduled_route_mode": True,
                        "candidate_global_kv_enabled": True,
                        "baseline_task_family_coverage": True,
                        "baseline_difficulty_coverage": True,
                        "candidate_task_family_coverage": False,
                        "candidate_difficulty_coverage": True,
                        "global_kv_active": True,
                        "quality_not_worse": True,
                        "memory_budget_present": True,
                        "global_budget_below_local_context": True,
                    },
                }
            ],
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    no_difference = _symptom(report, "global_kv_noise", "global_on_off_no_difference")
    candidate = no_difference["evidence"]["comparison_candidates"][0]

    assert report["risks"]["global_kv_noise"]["status"] == "fail"
    assert no_difference["triggered"] is True
    assert candidate["coverage_checks"]["candidate_task_family_coverage"] is False
    assert candidate["coverage_contract_passed"] is False
    assert candidate["passes_stage5_long_context_contract"] is False


def test_risk_audit_requires_passing_long_context_report_for_global_kv_clearance(tmp_path: Path) -> None:
    retention = _write_json(
        tmp_path / "retention.json",
        {
            "overall_status": "pass",
            "metrics": {"global_attention_mass": 0.02},
            "checks": {"global_attention_mass_nonzero": True},
        },
    )
    long_context = _write_json(
        tmp_path / "long_context_compare.json",
        {
            "overall_status": "fail",
            "candidate_count": 1,
            "comparisons": [
                {
                    "candidate_report": "global.json",
                    "candidate_run_dir": "global",
                    "status": "pass",
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
        },
    )

    output = make_risk_audit_report(
        output_path=tmp_path / "risk.json",
        global_kv_retention_report_path=retention,
        long_context_compare_report_path=long_context,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    no_difference = _symptom(report, "global_kv_noise", "global_on_off_no_difference")
    candidate = no_difference["evidence"]["comparison_candidates"][0]

    assert report["risks"]["global_kv_noise"]["status"] == "fail"
    assert no_difference["triggered"] is True
    assert no_difference["evidence"]["overall_status"] == "fail"
    assert no_difference["evidence"]["passing_comparison_count"] == 0
    assert candidate["passes_stage5_long_context_contract"] is True


def test_risk_audit_eval_config_resolves() -> None:
    config = load_config("configs/eval/risk_audit.yaml")
    assert config["eval_name"] == "risk_audit_report"
    assert config["thresholds"]["min_block_load_entropy_normalized"] == 0.05


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _symptom(report: dict, risk_name: str, symptom_name: str) -> dict:
    risk = report["risks"][risk_name]
    return next(item for item in risk["symptoms"] if item["name"] == symptom_name)
