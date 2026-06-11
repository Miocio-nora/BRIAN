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
                    "checks": {"not_all_immediate_exit": True},
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
            "checks": {"candidate_present": True, "any_measurable_difference": True},
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
        {"overall_status": "pass", "candidate_count": 1, "comparisons": [{"status": "pass"}]},
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
            },
            "model": {"beam_size": 2},
            "routing": {"parallel_branch_count": {"max": 2}},
        },
    )
    parallel_compare = _write_json(
        tmp_path / "parallel_compare.json",
        {
            "overall_status": "pass",
            "comparisons": [{"status": "pass", "checks": {"parallel_branch_benefit_proxy": True}}],
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
                    "checks": {"not_all_immediate_exit": False},
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
            "checks": {"candidate_present": True, "any_measurable_difference": False},
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
