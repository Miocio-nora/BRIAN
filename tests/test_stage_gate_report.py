import json
from pathlib import Path

import yaml
import pytest

from brian_sphere_llm.eval.difficulty import difficulty_step_correlation
from brian_sphere_llm.eval.stage_gate_report import make_stage_gate_report, pearson_correlation


def _write_run(
    root: Path,
    name: str,
    *,
    stage: str,
    val_loss: float,
    train_row: dict,
    determinism_status: str | None = None,
    resume_event: dict | None = None,
    baseline_difficulty_report: dict | None = None,
    fixed_route_stability_report: dict | None = None,
    pseudo_route_curriculum_report: dict | None = None,
    scheduled_routing_report: dict | None = None,
    difficulty_report: dict | None = None,
    global_kv_retention_report: dict | None = None,
    parallel_passing_report: dict | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_latest").mkdir()
    (run_dir / "checkpoint_latest" / "state.pt").write_bytes(b"stub")
    (run_dir / "checkpoint_best").mkdir()
    (run_dir / "checkpoint_best" / "state.pt").write_bytes(b"stub")
    (run_dir / "model_stats.json").write_text(json.dumps({"model_name": name}), encoding="utf-8")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump({"stage": stage}), encoding="utf-8")
    (run_dir / "eval_log.jsonl").write_text(json.dumps({"validation_loss": val_loss, "perplexity": 1.0}) + "\n", encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(json.dumps(train_row | {"loss": val_loss}) + "\n", encoding="utf-8")
    if determinism_status is not None:
        (run_dir / "eval_determinism_report.json").write_text(
            json.dumps({"overall_status": determinism_status, "checks": {"numeric_metrics_within_tolerance": True}}),
            encoding="utf-8",
        )
    if resume_event is not None:
        (run_dir / "resume_events.jsonl").write_text(json.dumps(resume_event) + "\n", encoding="utf-8")
    if baseline_difficulty_report is not None:
        (run_dir / "baseline_difficulty_report.json").write_text(
            json.dumps(baseline_difficulty_report),
            encoding="utf-8",
        )
    if fixed_route_stability_report is not None:
        (run_dir / "fixed_route_stability_report.json").write_text(
            json.dumps(fixed_route_stability_report),
            encoding="utf-8",
        )
    if pseudo_route_curriculum_report is not None:
        (run_dir / "pseudo_route_curriculum_report.json").write_text(
            json.dumps(pseudo_route_curriculum_report),
            encoding="utf-8",
        )
    if scheduled_routing_report is not None:
        (run_dir / "scheduled_routing_report.json").write_text(
            json.dumps(scheduled_routing_report),
            encoding="utf-8",
        )
    if difficulty_report is not None:
        (run_dir / "difficulty_step_report.json").write_text(
            json.dumps(difficulty_report),
            encoding="utf-8",
        )
    if global_kv_retention_report is not None:
        (run_dir / "global_kv_retention_report.json").write_text(
            json.dumps(global_kv_retention_report),
            encoding="utf-8",
        )
    if parallel_passing_report is not None:
        (run_dir / "parallel_passing_report.json").write_text(
            json.dumps(parallel_passing_report),
            encoding="utf-8",
        )
    return run_dir


def test_pearson_and_difficulty_step_correlation() -> None:
    assert pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert difficulty_step_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert difficulty_step_correlation([1.0], [1.0]) is None


def test_stage_gate_report_writes_json(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={"resumed_from_step": 1, "target_max_steps": 2, "optimizer_state_loaded": True},
        baseline_difficulty_report={
            "sample_count": 3,
            "difficulty_bin_count": 3,
            "by_difficulty": {
                "easy": {"sample_count": 1, "mean_baseline_cross_entropy": 1.0},
                "medium": {"sample_count": 1, "mean_baseline_cross_entropy": 2.0},
                "hard": {"sample_count": 1, "mean_baseline_cross_entropy": 3.0},
            },
        },
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        fixed_route_stability_report={
            "overall_status": "pass",
            "checks": {
                "forward_completed": True,
                "logits_shape_matches": True,
                "logits_finite": True,
                "sample_losses_finite": True,
                "fixed_route_matches_targets": True,
                "route_imitation_accuracy_is_one": True,
                "position_norm_finite": True,
                "routing_summary_finite": True,
            },
        },
    )
    stage2 = _write_run(
        tmp_path,
        "stage2",
        stage="stage2_router_imitation",
        val_loss=10.2,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        pseudo_route_curriculum_report={
            "overall_status": "pass",
            "checks": {
                "baseline_samples_present": True,
                "difficulty_bins_present": True,
                "mixed_skip_recur_policy": True,
                "easy_has_skip_or_small_pool": True,
                "hard_has_recur_transition": True,
                "exit_action_supervised": True,
                "easy_exits_no_later_than_hard": True,
                "route_length_conditioned_by_difficulty": True,
            },
        },
    )
    stage3 = _write_run(
        tmp_path,
        "stage3",
        stage="stage3_scheduled_free_routing",
        val_loss=10.3,
        train_row={
            "route_entropy": 0.5,
            "block_load_entropy": 0.5,
            "route_path_diversity": 0.5,
            "average_route_steps": 2.0,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        difficulty_report={"sample_count": 3, "difficulty_step_correlation": 0.2},
        scheduled_routing_report={
            "overall_status": "pass",
            "checks": {
                "scheduled_stage": True,
                "schedule_present": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
            },
            "logged_schedule_values": [
                {"step": 1, "scheduled_router_probability": 0.1, "scheduled_lambda_route": 1.0},
                {"step": 2, "scheduled_router_probability": 1.0, "scheduled_lambda_route": 0.05},
            ],
        },
    )
    stage5 = _write_run(
        tmp_path,
        "global",
        stage="stage5_global_kv",
        val_loss=11.0,
        train_row={
            "global_attention_mass": 1.0,
            "global_read_gate_mean": 0.01,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        global_kv_retention_report={
            "overall_status": "pass",
            "model": {
                "global_kv_enabled": True,
                "global_sink_slots": 1,
                "global_window_slots": 3,
                "retention_capacity_slots": 4,
            },
            "metrics": {
                "global_attention_mass": 1.0,
                "global_sink_attention_mass": 0.25,
                "global_window_attention_mass": 0.75,
                "global_read_gate_mean": 0.01,
                "global_cache_slots_mean": 2.0,
            },
            "checks": {
                "stage5_global_kv_stage": True,
                "global_kv_enabled": True,
                "sink_slots_configured": True,
                "window_slots_configured": True,
                "retention_capacity_present": True,
                "global_attention_mass_nonzero": True,
                "global_read_gate_nonzero": True,
                "global_cache_slots_present": True,
                "sink_attention_mass_measured": True,
                "window_attention_mass_measured": True,
                "sink_window_mass_conserved": True,
                "cache_slots_within_retention_capacity": True,
            },
        },
    )
    long_context_compare = tmp_path / "long_context_compare.json"
    long_context_compare.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [
                    {
                        "status": "pass",
                        "checks": {
                            "global_kv_active": True,
                            "quality_metrics_present": True,
                            "quality_not_worse": True,
                            "memory_budget_present": True,
                            "global_budget_below_local_context": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "gate.json"
    report_path = make_stage_gate_report(
        [baseline, fixed, stage2, stage3, stage5],
        output_path=output,
        long_context_compare_report_path=long_context_compare,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_count"] == 5
    assert report["gates"]["stage0_to_1"]["status"] == "pass"
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_resume_event"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["baseline_difficulty_bins_present"] is True
    assert report["gates"]["stage1_to_2"]["status"] == "pass"
    assert report["gates"]["stage1_to_2"]["checks"]["fixed_route_stability_passed"] is True
    assert report["gates"]["stage2_to_3"]["status"] == "pass"
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_passed"] is True
    assert report["gates"]["stage3_to_4"]["status"] == "pass"
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_passed"] is True
    assert report["gates"]["stage5_to_6"]["status"] == "pass"
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_passed"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["sink_window_attention_measured"] is True
    assert report["supplemental_reports"]["long_context_compare_report"] == str(long_context_compare)


def test_stage_gate_report_uses_cost_control_report(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 2.0,
            "first_exit_step_histogram": {"1": 1, "2": 2},
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    cost_report = tmp_path / "cost.json"
    cost_report.write_text(
        json.dumps(
            {
                "run_count": 4,
                "analysis": {
                    "status": "pass",
                    "active_block_evals_range": 0.4,
                    "checks": {
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "output_probability_not_decreasing_with_cost": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    out_report = tmp_path / "out_by_difficulty.json"
    out_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": {
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
        ),
        encoding="utf-8",
    )
    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["gates"]["stage4_to_5"]
    assert gate["checks"]["cost_control_report_present"] is True
    assert gate["checks"]["cost_control_active_range_present"] is True
    assert gate["checks"]["out_by_difficulty_report_present"] is True
    assert gate["checks"]["out_by_difficulty_passed"] is True
    assert gate["checks"]["not_never_exit"] is True
    assert gate["checks"]["hard_compute_not_below_easy"] is True
    assert gate["cost_control_status"] == "pass"
    assert report["supplemental_reports"]["cost_control_report"] == str(cost_report)
    assert report["supplemental_reports"]["out_by_difficulty_report"] == str(out_report)


def test_stage4_gate_warns_when_model_never_exits(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 4.0,
            "first_exit_step_histogram": {"0": 2},
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    cost_report = tmp_path / "cost.json"
    cost_report.write_text(
        json.dumps(
            {
                "analysis": {
                    "status": "pass",
                    "checks": {
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "output_probability_not_decreasing_with_cost": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    out_report = tmp_path / "out_by_difficulty.json"
    out_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": {
                    "route_steps_non_decreasing_with_difficulty": True,
                    "active_compute_non_decreasing_with_difficulty": True,
                    "easy_output_probability_at_least_hard": True,
                },
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["status"] == "warn"
    assert gate["checks"]["exit_distribution_present"] is True
    assert gate["checks"]["not_all_immediate_exit"] is True
    assert gate["checks"]["not_never_exit"] is False


def test_stage3_gate_requires_stage1_loss_comparison(tmp_path: Path) -> None:
    stage3 = _write_run(
        tmp_path,
        "stage3",
        stage="stage3_scheduled_free_routing",
        val_loss=10.0,
        train_row={
            "route_entropy": 0.5,
            "block_load_entropy": 0.5,
            "route_path_diversity": 0.5,
            "average_route_steps": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        difficulty_report={"sample_count": 3, "difficulty_step_correlation": 0.2},
        scheduled_routing_report={
            "overall_status": "pass",
            "checks": {
                "scheduled_stage": True,
                "schedule_present": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
            },
        },
    )
    report_path = make_stage_gate_report([stage3], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage3_to_4"]

    assert gate["status"] == "warn"
    assert gate["loss_ratio_vs_stage1"] is None
    assert gate["checks"]["validation_loss_not_collapsed"] is False
    assert gate["checks"]["scheduled_routing_passed"] is True


def test_stage5_gate_warns_without_long_context_compare_report(tmp_path: Path) -> None:
    stage5 = _write_run(
        tmp_path,
        "global",
        stage="stage5_global_kv",
        val_loss=10.0,
        train_row={
            "global_attention_mass": 1.0,
            "global_read_gate_mean": 0.01,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    report_path = make_stage_gate_report([stage5], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]
    assert gate["status"] == "warn"
    assert gate["checks"]["global_kv_retention_report_present"] is False
    assert gate["checks"]["long_context_compare_report_present"] is False
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage5_gate_requires_long_context_compare_key_checks(tmp_path: Path) -> None:
    stage5 = _write_run(
        tmp_path,
        "global",
        stage="stage5_global_kv",
        val_loss=10.0,
        train_row={
            "global_attention_mass": 1.0,
            "global_read_gate_mean": 0.01,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        global_kv_retention_report={
            "overall_status": "pass",
            "model": {
                "global_kv_enabled": True,
                "global_sink_slots": 1,
                "global_window_slots": 3,
                "retention_capacity_slots": 4,
            },
            "metrics": {
                "global_attention_mass": 1.0,
                "global_sink_attention_mass": 0.25,
                "global_window_attention_mass": 0.75,
                "global_read_gate_mean": 0.01,
                "global_cache_slots_mean": 2.0,
            },
            "checks": {
                "stage5_global_kv_stage": True,
                "global_kv_enabled": True,
                "sink_slots_configured": True,
                "window_slots_configured": True,
                "retention_capacity_present": True,
                "global_attention_mass_nonzero": True,
                "global_read_gate_nonzero": True,
                "global_cache_slots_present": True,
                "sink_attention_mass_measured": True,
                "window_attention_mass_measured": True,
                "sink_window_mass_conserved": True,
                "cache_slots_within_retention_capacity": True,
            },
        },
    )
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [{"status": "pass", "checks": {"global_kv_active": True}}],
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_passed"] is True
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage6_gate_uses_parallel_compare_report(tmp_path: Path) -> None:
    stage6 = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        parallel_passing_report={
            "overall_status": "pass",
            "checks": {
                "stage6_parallel_stage": True,
                "parallel_passing_enabled": True,
                "parallel_route_selected": True,
                "beam_size_present": True,
                "beam_size_within_limit": True,
                "branch_cost_enabled": True,
                "branch_metrics_present": True,
                "parallel_branch_active": True,
                "branch_count_bounded_by_beam": True,
                "score_margin_measured": True,
                "delta_memory_policy_present": True,
                "delta_cache_bounded_by_window": True,
            },
            "model": {"beam_size": 2, "branch_cost": 0.01},
            "routing": {"parallel_branch_count": {"max": 2.0}, "parallel_delta_cache_slots": {"max": 2.0}},
        },
    )
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [
                    {
                        "status": "pass",
                        "checks": {
                            "parallel_branch_active": True,
                            "parallel_score_margin_present": True,
                            "quality_not_worse": True,
                            "active_compute_bounded": True,
                            "estimated_flops_bounded": True,
                            "throughput_not_collapsed": True,
                            "parallel_branch_benefit_proxy": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["gates"]["stage6_to_scale"]
    assert gate["status"] == "pass"
    assert gate["checks"]["parallel_passing_report_present"] is True
    assert gate["checks"]["parallel_passing_report_passed"] is True
    assert gate["checks"]["parallel_branch_count_bounded_by_beam"] is True
    assert gate["checks"]["parallel_delta_cache_bounded"] is True
    assert gate["checks"]["parallel_compare_report_present"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is True
    assert report["supplemental_reports"]["parallel_compare_report"] == str(compare_report)


def test_stage6_gate_accepts_stage7_parallel_alias(tmp_path: Path) -> None:
    stage7 = _write_run(
        tmp_path,
        "parallel",
        stage="stage7_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        parallel_passing_report={
            "overall_status": "pass",
            "checks": {
                "stage6_parallel_stage": True,
                "parallel_passing_enabled": True,
                "parallel_route_selected": True,
                "beam_size_present": True,
                "beam_size_within_limit": True,
                "branch_cost_enabled": True,
                "branch_metrics_present": True,
                "parallel_branch_active": True,
                "branch_count_bounded_by_beam": True,
                "score_margin_measured": True,
                "delta_memory_policy_present": True,
                "delta_cache_bounded_by_window": True,
            },
        },
    )
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [
                    {
                        "status": "pass",
                        "checks": {
                            "parallel_branch_active": True,
                            "parallel_score_margin_present": True,
                            "quality_not_worse": True,
                            "active_compute_bounded": True,
                            "estimated_flops_bounded": True,
                            "throughput_not_collapsed": True,
                            "parallel_branch_benefit_proxy": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage7],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] == "pass"
    assert gate["checks"]["parallel_branch_count_present"] is True
    assert gate["checks"]["parallel_passing_report_passed"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is True


def test_stage6_gate_warns_without_parallel_compare_report(tmp_path: Path) -> None:
    stage6 = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )
    report_path = make_stage_gate_report([stage6], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]
    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_passing_report_present"] is False
    assert gate["checks"]["parallel_compare_report_present"] is False
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False


def test_stage6_gate_requires_parallel_compare_key_checks(tmp_path: Path) -> None:
    stage6 = _write_run(
        tmp_path,
        "parallel",
        stage="stage6_parallel_passing",
        val_loss=10.0,
        train_row={
            "parallel_branch_count_mean": 2.0,
            "parallel_score_margin_mean": 0.1,
            "global_cache_slots_mean": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        parallel_passing_report={
            "overall_status": "pass",
            "checks": {
                "stage6_parallel_stage": True,
                "parallel_passing_enabled": True,
                "parallel_route_selected": True,
                "beam_size_present": True,
                "beam_size_within_limit": True,
                "branch_cost_enabled": True,
                "branch_metrics_present": True,
                "parallel_branch_active": True,
                "branch_count_bounded_by_beam": True,
                "score_margin_measured": True,
                "delta_memory_policy_present": True,
                "delta_cache_bounded_by_window": True,
            },
        },
    )
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "candidate_count": 1,
                "comparisons": [{"status": "pass", "checks": {"parallel_branch_benefit_proxy": True}}],
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_compare_passed"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False
