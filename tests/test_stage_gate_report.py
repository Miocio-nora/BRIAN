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
    model_stats: dict | None = None,
    data_manifest_ref: dict | None = None,
    routing: dict | None = None,
    write_data_manifest_ref: bool = True,
    default_routing_metrics: bool = True,
    write_checkpoint_best: bool = True,
    write_rank_state: bool = True,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_latest").mkdir()
    (run_dir / "checkpoint_latest" / "state.pt").write_bytes(b"stub")
    if write_rank_state:
        (run_dir / "checkpoint_latest" / "rank_state_00000.pt").write_bytes(b"stub")
    if write_checkpoint_best:
        (run_dir / "checkpoint_best").mkdir()
        (run_dir / "checkpoint_best" / "state.pt").write_bytes(b"stub")
    stats = _model_stats(name) if model_stats is None else model_stats
    (run_dir / "model_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    if write_data_manifest_ref:
        manifest_ref = _data_manifest_ref() if data_manifest_ref is None else data_manifest_ref
        (run_dir / "data_manifest_ref.json").write_text(json.dumps(manifest_ref), encoding="utf-8")
    train_row = {
        "tokens_per_second": 128.0,
        "train_step_time_seconds": 0.1,
        "train_latency_ms_per_token": 0.8,
    } | train_row
    if default_routing_metrics and stage != "stage0_baseline":
        train_row = _routed_train_row() | train_row
    config = {"stage": stage}
    if routing is not None:
        config["routing"] = routing
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    eval_row = {
        "validation_loss": val_loss,
        "perplexity": 1.0,
        "inference_time_seconds": 0.2,
        "inference_tokens_per_second": 64.0,
        "inference_latency_ms_per_token": 1.0,
    }
    (run_dir / "eval_log.jsonl").write_text(json.dumps(eval_row) + "\n", encoding="utf-8")
    (run_dir / "train_log.jsonl").write_text(json.dumps(train_row | {"loss": val_loss}) + "\n", encoding="utf-8")
    if determinism_status is not None:
        (run_dir / "eval_determinism_report.json").write_text(
            json.dumps(
                {
                    "overall_status": determinism_status,
                    "checks": {
                        "checkpoint_loaded": True,
                        "two_eval_passes_completed": True,
                        "compared_numeric_metrics_present": True,
                        "numeric_metrics_within_tolerance": determinism_status == "pass",
                    },
                }
            ),
            encoding="utf-8",
        )
    if resume_event is not None:
        (run_dir / "resume_events.jsonl").write_text(json.dumps(_resume_event(resume_event)) + "\n", encoding="utf-8")
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


def _baseline_difficulty_report() -> dict:
    return {
        "sample_count": 3,
        "difficulty_bins": ["easy", "medium", "hard"],
        "difficulty_bin_count": 3,
        "by_difficulty": {
            "easy": {"sample_count": 1, "mean_baseline_cross_entropy": 1.0},
            "medium": {"sample_count": 1, "mean_baseline_cross_entropy": 2.0},
            "hard": {"sample_count": 1, "mean_baseline_cross_entropy": 3.0},
        },
    }


def _hard_exit_compare_report(*, overall_status: str = "pass", checks: dict[str, bool] | None = None) -> dict:
    comparison_checks = {
        "baseline_stage4_scheduled_free_routing": True,
        "baseline_without_hard_exit": True,
        "candidate_stage4_output_action": True,
        "candidate_with_hard_exit": True,
        "candidate_top1_hard_exit_rule": True,
        "inference_timing_present": True,
        "latency_ratio_within_threshold": True,
        "inference_time_ratio_within_threshold": True,
        "route_steps_not_increasing": True,
        "validation_loss_not_worse": True,
    }
    if checks is not None:
        comparison_checks |= checks
    return {
        "baseline_run": "stage4_without_hard_exit",
        "baseline": {"hard_exit_enabled": False, "average_route_steps": 2.5},
        "candidate_count": 1,
        "comparisons": [
            {
                "candidate_run": "stage4_hard_exit",
                "candidate": {
                    "hard_exit_enabled": True,
                    "hard_exit_top1_rule": comparison_checks["candidate_top1_hard_exit_rule"],
                    "average_route_steps": 2.0,
                },
                "baseline_comparison": {
                    "validation_loss_delta": 0.0,
                    "validation_loss_ratio": 1.0,
                    "inference_latency_ms_per_token_ratio": 0.9,
                    "inference_time_seconds_ratio": 0.9,
                    "average_route_steps_ratio": 0.8,
                },
                "checks": comparison_checks,
                "status": "pass" if all(comparison_checks.values()) else "warn",
            }
        ],
        "thresholds": {
            "max_validation_loss_delta": 0.0,
            "max_latency_ratio": 1.0,
            "max_inference_time_ratio": 1.0,
            "max_route_step_ratio": 1.0,
        },
        "overall_status": overall_status,
    }


def _global_kv_retention_report(*, checks: dict[str, bool] | None = None) -> dict:
    report_checks = {
        "stage5_global_kv_stage": True,
        "global_kv_enabled": True,
        "sink_slots_configured": True,
        "window_slots_configured": True,
        "retention_capacity_present": True,
        "global_attention_mass_nonzero": True,
        "global_attention_mass_bounded": True,
        "global_read_gate_nonzero": True,
        "global_read_gate_bounded": True,
        "global_cache_slots_present": True,
        "sink_attention_mass_measured": True,
        "window_attention_mass_measured": True,
        "sink_window_mass_conserved": True,
        "cache_slots_within_retention_capacity": True,
        "read_ratio_measured": True,
        "window_utilization_measured": True,
    }
    if checks is not None:
        report_checks |= checks
    return {
        "overall_status": "pass" if all(report_checks.values()) else "fail",
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
        "checks": report_checks,
    }


def _long_context_compare_report(*, checks: dict[str, bool] | None = None) -> dict:
    comparison_checks = {
        "baseline_report_passed": True,
        "candidate_report_passed": True,
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
        "quality_metrics_present": True,
        "quality_not_worse": True,
        "memory_budget_present": True,
        "global_budget_below_local_context": True,
    }
    if checks is not None:
        comparison_checks |= checks
    return {
        "overall_status": "pass" if all(comparison_checks.values()) else "warn",
        "candidate_count": 1,
        "comparisons": [{"status": "pass" if all(comparison_checks.values()) else "warn", "checks": comparison_checks}],
    }


def _parallel_passing_report(*, checks: dict[str, bool] | None = None) -> dict:
    report_checks = {
        "stage6_parallel_stage": True,
        "parallel_passing_enabled": True,
        "parallel_route_selected": True,
        "shared_base_global_memory_enabled": True,
        "beam_size_present": True,
        "beam_size_within_limit": True,
        "branch_cost_enabled": True,
        "branch_metrics_present": True,
        "parallel_branch_active": True,
        "branch_count_bounded_by_beam": True,
        "score_margin_measured": True,
        "score_margin_nonnegative": True,
        "branch_delta_memory_measured": True,
        "delta_cache_nonnegative": True,
        "delta_memory_policy_present": True,
        "delta_cache_bounded_by_window": True,
        "branch_score_decay_configured": True,
    }
    if checks is not None:
        report_checks |= checks
    return {
        "overall_status": "pass" if all(report_checks.values()) else "fail",
        "checks": report_checks,
        "model": {
            "beam_size": 2,
            "branch_cost": 0.01,
            "branch_score_decay": 0.99,
            "memory_policy": "shared_base_global_kv_with_branch_delta",
        },
        "routing": {"parallel_branch_count": {"max": 2.0}, "parallel_delta_cache_slots": {"max": 2.0}},
    }


def _parallel_compare_report(*, checks: dict[str, bool] | None = None) -> dict:
    comparison_checks = {
        "baseline_stage5_global_kv": True,
        "baseline_scheduled_route_mode": True,
        "baseline_global_kv_enabled": True,
        "baseline_parallel_passing_disabled": True,
        "baseline_topk_weighted_fusion": True,
        "candidate_parallel_stage": True,
        "candidate_parallel_route_mode": True,
        "candidate_parallel_passing_enabled": True,
        "candidate_global_kv_enabled": True,
        "parallel_branch_active": True,
        "parallel_score_margin_present": True,
        "quality_not_worse": True,
        "active_compute_bounded": True,
        "estimated_flops_bounded": True,
        "throughput_not_collapsed": True,
        "parallel_branch_benefit_proxy": True,
    }
    if checks is not None:
        comparison_checks |= checks
    return {
        "overall_status": "pass" if all(comparison_checks.values()) else "warn",
        "candidate_count": 1,
        "comparisons": [{"status": "pass" if all(comparison_checks.values()) else "warn", "checks": comparison_checks}],
    }


def _data_manifest_ref() -> dict:
    return {
        "recipe_name": "unit_data",
        "path": "data/manifests/unit_data.jsonl",
        "path_exists": True,
        "tokenized_dir": "data/tokenized/unit_data",
        "tokenized_dir_exists": True,
        "stats_path": "data/tokenized/unit_data/stats.json",
        "stats_path_exists": True,
        "tokenized_artifacts_present": True,
        "sequence_length": 8,
        "num_documents": 4,
        "num_tokens_train": 24,
        "num_tokens_val": 8,
        "avg_tokens_per_doc": 8.0,
        "vocab_size": 259,
        "sha256_manifest": "abc123",
        "sha256_manifest_verified": True,
        "manifest_row_count": 4,
        "manifest_source_text_hashes_verified": True,
        "manifest_token_hashes_verified": True,
        "manifest_source_text_hash_failure_count": 0,
        "manifest_token_hash_failure_count": 0,
        "tokenizer_artifact_count": 2,
        "tokenizer_artifacts_present": True,
        "tokenizer_artifact_hashes": {"tokenizer.json": "abc", "tokenizer_config.json": "def"},
        "tokenizer_artifact_hashes_present": True,
        "tokenizer": {
            "name": "simple-byte-tokenizer",
            "revision": "local",
            "license": "internal-test",
            "vocab_size": 259,
            "special_tokens": {"bos": 256, "eos": 257, "pad": 0, "unk": 258},
        },
        "stats_recipe_name_matches_config": True,
        "stats_sequence_length_matches_config": True,
        "source_mixture_expected": {"unit": 1.0},
        "source_mixture_realized": {"unit": 32},
        "source_mixture_realized_share": {"unit": 1.0},
    }


def _resume_event(overrides: dict) -> dict:
    return {
        "checkpoint": "checkpoint_latest",
        "resumed_from_step": 1,
        "target_max_steps": 2,
        "optimizer_state_loaded": True,
        "rng_state_loaded": True,
        "rank_state_loaded": False,
        "rank_state_path": None,
        "data_epoch": 0,
        "microbatch_in_epoch": 1,
    } | overrides


def _model_stats(name: str) -> dict:
    return {"model_name": name, "parameter_count": 100}


def _routed_train_row() -> dict:
    return {
        "route_entropy": 0.5,
        "block_load_entropy": 0.5,
        "route_path_diversity": 0.5,
        "active_block_evals_per_token": 0.5,
        "average_route_steps": 2.0,
        "advance_ratio": 0.4,
        "skip_ratio": 0.3,
        "recur_ratio": 0.3,
        "position_norm_mean": 1.0,
        "location_distance_mean": 0.25,
        "p_output_mean": 0.5,
        "cost_loss": 0.01,
        "balance_loss": 0.02,
        "location_loss": 0.03,
        "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        "topk_block_histogram": {"0": 1, "1": 1, "2": 1},
        "exit_step_distribution": [1, 1],
        "first_exit_step_histogram": {"2": 1},
        "route_path_examples": [{"sample_index": 0, "actions": [0, 1, 2]}],
        "position_norm_trajectory": [1.0, 0.75],
        "location_distance_trajectory": [0.25, 0.5],
    }


def test_stage_gate_report_writes_json(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
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
        routing={"mode": "pseudo", "pseudo_policy": "sequential"},
    )
    mixed_pseudo = _write_run(
        tmp_path,
        "stage3_pseudo",
        stage="stage3_pseudo_skip_recur",
        val_loss=10.2,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        routing={"mode": "pseudo", "pseudo_policy": "mixed_skip_recur"},
        pseudo_route_curriculum_report={
            "overall_status": "pass",
            "checks": {
                "baseline_samples_present": True,
                "baseline_cross_entropy_numeric": True,
                "baseline_cross_entropy_ordered_by_difficulty": True,
                "difficulty_bins_present": True,
                "mixed_skip_recur_policy": True,
                "stage3_pseudo_skip_recur_stage": True,
                "pseudo_routing_mode": True,
                "easy_has_skip_or_small_pool": True,
                "easy_uses_skip_or_early_exit": True,
                "hard_has_recur_transition": True,
                "hard_uses_recurrence": True,
                "exit_action_supervised": True,
                "out_supervised": True,
                "supervised_out_targets_present": True,
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
                "stage3_scheduled_free_routing_stage": True,
                "scheduled_routing_mode": True,
                "scheduled_stage": True,
                "schedule_present": True,
                "schedule_values_numeric": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
                "logged_eval_schedule_values_present": True,
                "logged_eval_router_probability_matches_schedule": True,
                "logged_eval_lambda_route_matches_schedule": True,
            },
            "logged_schedule_values": [
                {"step": 1, "scheduled_router_probability": 0.1, "scheduled_lambda_route": 1.0},
                {"step": 2, "scheduled_router_probability": 1.0, "scheduled_lambda_route": 0.05},
            ],
        },
    )
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.5,
        train_row={
            "average_route_steps": 2.0,
            "max_route_steps": 4,
            "forced_max_step_exit_count": 0,
            "forced_max_step_exit_fraction": 0.0,
            "first_exit_step_histogram": {"1": 1, "2": 2},
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
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
                "global_attention_mass_bounded": True,
                "global_read_gate_nonzero": True,
                "global_read_gate_bounded": True,
                "global_cache_slots_present": True,
                "sink_attention_mass_measured": True,
                "window_attention_mass_measured": True,
                "sink_window_mass_conserved": True,
                "cache_slots_within_retention_capacity": True,
                "read_ratio_measured": True,
                "window_utilization_measured": True,
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
                            "baseline_report_passed": True,
                            "candidate_report_passed": True,
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
        [baseline, fixed, stage2, mixed_pseudo, stage3, stage4, stage5],
        output_path=output,
        long_context_compare_report_path=long_context_compare,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_count"] == 7
    assert report["gates"]["stage0_to_1"]["status"] == "pass"
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_resume_event"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_resume_event_valid"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_rank_state_present"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_rank_state_resume_metadata_valid"] is True
    assert report["gates"]["stage0_to_1"]["resume_event_checks"]["rng_state_loaded"] is True
    assert report["gates"]["stage0_to_1"]["resume_event_checks"]["data_epoch_nonnegative"] is True
    assert report["gates"]["stage0_to_1"]["resume_event_checks"]["microbatch_in_epoch_nonnegative"] is True
    assert report["gates"]["stage0_to_1"]["rank_state_resume_event_checks"][
        "rank_state_loaded_flag_present"
    ] is True
    assert report["gates"]["stage0_to_1"]["rank_state_resume_event_checks"][
        "rank_state_path_empty_when_not_loaded"
    ] is True
    assert report["gates"]["stage0_to_1"]["checks"]["checkpoint_best_artifact"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["eval_determinism_checks_passed"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["model_stats_valid"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["data_manifest_ref_valid"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["config_resolved_present"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["train_log_present"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["validation_report_valid"] is True
    assert report["gates"]["stage0_to_1"]["validation_report_metrics"]["tokens_per_second"] == 128.0
    assert report["gates"]["stage0_to_1"]["checks"]["baseline_difficulty_bins_present"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["baseline_difficulty_bin_means_present"] is True
    assert report["gates"]["stage0_to_1"]["checks"]["baseline_difficulty_bin_means_ordered"] is True
    assert report["gates"]["stage1_to_2"]["status"] == "pass"
    assert report["gates"]["stage1_to_2"]["checks"]["fixed_route_stability_passed"] is True
    assert report["gates"]["stage1_to_2"]["checks"]["validation_report_valid"] is True
    assert report["gates"]["stage1_to_2"]["validation_report_metrics"]["active_block_evals_per_token"] == 0.5
    assert report["gates"]["stage1_to_2"]["checks"]["routing_report_valid"] is True
    assert report["gates"]["stage1_to_2"]["checks"]["checkpoint_best_present"] is True
    assert report["gates"]["stage2_to_3"]["status"] == "pass"
    assert report["gates"]["stage2_to_3"]["checks"]["sequential_pseudo_policy"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["mixed_pseudo_policy"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_mixed_policy"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_correct_stage"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_pseudo_mode"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_baseline_samples"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_difficulty_bins"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_baseline_ce_numeric"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_baseline_ce_ordered"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_easy_targets"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_hard_targets"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_out_supervised"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_easy_before_hard"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_route_length_conditioned"] is True
    assert report["gates"]["stage2_to_3"]["checks"]["pseudo_route_curriculum_passed"] is True
    assert report["gates"]["stage2_to_3"]["sequential_stage"] == "stage2_router_imitation"
    assert report["gates"]["stage2_to_3"]["mixed_stage"] == "stage3_pseudo_skip_recur"
    assert report["gates"]["stage3_to_4"]["status"] == "pass"
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_correct_stage"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_scheduled_mode"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_scheduled_stage"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_schedule_present"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_schedule_values_numeric"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_router_probability_monotonic"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_lambda_route_monotonic"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_router_probability_increases"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_lambda_route_decays"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_reaches_free_router"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_train_schedule_values_present"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_train_router_probability_matches"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_train_lambda_route_matches"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_eval_schedule_values_present"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_eval_router_probability_matches"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_eval_lambda_route_matches"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["scheduled_routing_passed"] is True
    assert report["gates"]["stage3_to_4"]["checks"]["difficulty_step_correlation_positive"] is True
    assert report["gates"]["stage5_to_6"]["status"] == "pass"
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_passed"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_stage5"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_enabled"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_capacity_present"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["global_kv_retention_cache_slots_present"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["stage4_reference_validation_loss_present"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["validation_loss_not_worse_than_stage4"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["sink_window_attention_measured"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["local_global_read_ratio_measured"] is True
    assert report["gates"]["stage5_to_6"]["checks"]["global_cache_window_utilization_measured"] is True
    assert report["gates"]["stage5_to_6"]["loss_ratio_vs_stage4"] == 11.0 / 10.5
    assert report["supplemental_reports"]["long_context_compare_report"] == str(long_context_compare)


def test_stage0_gate_requires_valid_resume_event(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": False,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["checkpoint_resume_event"] is True
    assert gate["checks"]["checkpoint_resume_event_valid"] is False
    assert gate["resume_event_checks"]["optimizer_state_loaded"] is False


def test_stage0_gate_requires_rng_and_dataloader_resume_state(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "rng_state_loaded": False,
            "data_epoch": -1,
            "microbatch_in_epoch": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["checks"]["checkpoint_resume_event_valid"] is False
    assert gate["resume_event_checks"]["rng_state_loaded"] is False
    assert gate["resume_event_checks"]["data_epoch_nonnegative"] is False
    assert gate["resume_event_checks"]["microbatch_in_epoch_nonnegative"] is False


def test_stage0_gate_requires_ordered_baseline_difficulty_bin_means(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={},
        baseline_difficulty_report={
            "sample_count": 3,
            "difficulty_bins": ["easy", "medium", "hard"],
            "difficulty_bin_count": 3,
            "by_difficulty": {
                "easy": {"sample_count": 1, "mean_baseline_cross_entropy": 3.0},
                "medium": {"sample_count": 1, "mean_baseline_cross_entropy": 2.0},
                "hard": {"sample_count": 1, "mean_baseline_cross_entropy": 1.0},
            },
        },
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["baseline_difficulty_bin_means_present"] is True
    assert gate["checks"]["baseline_difficulty_bin_means_ordered"] is False
    assert gate["baseline_difficulty_checks"]["bin_means_ordered"] is False


def test_stage0_gate_accepts_loaded_rank_state_resume_path(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "rank_state_loaded": True,
            "rank_state_path": "checkpoint_latest/rank_state_00000.pt",
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["checks"]["checkpoint_rank_state_present"] is True
    assert gate["checks"]["checkpoint_rank_state_resume_metadata_valid"] is True
    assert gate["rank_state_resume_event_checks"]["rank_state_path_present_when_loaded"] is True
    assert gate["rank_state_resume_event_checks"]["rank_state_path_name_valid_when_loaded"] is True
    assert gate["rank_state_resume_event_checks"]["rank_state_path_points_to_latest_when_loaded"] is True
    assert gate["rank_state_resume_event_checks"]["rank_state_file_exists_when_loaded"] is True


def test_stage0_gate_requires_loaded_rank_state_file_to_exist(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "rank_state_loaded": True,
            "rank_state_path": "checkpoint_latest/rank_state_00003.pt",
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["checkpoint_rank_state_present"] is True
    assert gate["checks"]["checkpoint_rank_state_resume_metadata_valid"] is False
    assert gate["rank_state_resume_event_checks"]["rank_state_file_exists_when_loaded"] is False


def test_stage0_gate_requires_determinism_key_checks(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    (baseline / "eval_determinism_report.json").write_text(
        json.dumps({"overall_status": "pass", "checks": {"numeric_metrics_within_tolerance": True}}),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["eval_deterministic"] is True
    assert gate["checks"]["eval_determinism_checks_passed"] is False


def test_stage0_gate_requires_valid_data_manifest_ref(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
        data_manifest_ref={
            "recipe_name": "unit_data",
            "path": "",
            "tokenized_dir": "data/tokenized/unit_data",
            "stats_path": "data/tokenized/unit_data/stats.json",
            "sequence_length": 8,
            "num_tokens_train": 24,
            "num_tokens_val": 8,
            "source_mixture_expected": {"unit": 1.0},
            "source_mixture_realized": {},
            "source_mixture_realized_share": {},
        },
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["data_manifest_ref_present"] is True
    assert gate["checks"]["data_manifest_ref_valid"] is False
    assert gate["data_manifest_ref_checks"]["path_present"] is False
    assert gate["data_manifest_ref_checks"]["path_exists"] is False
    assert gate["data_manifest_ref_checks"]["tokenized_dir_exists"] is False
    assert gate["data_manifest_ref_checks"]["stats_path_exists"] is False
    assert gate["data_manifest_ref_checks"]["tokenized_artifacts_present"] is False
    assert gate["data_manifest_ref_checks"]["sha256_manifest_present"] is False
    assert gate["data_manifest_ref_checks"]["sha256_manifest_verified"] is False
    assert gate["data_manifest_ref_checks"]["num_documents_positive"] is False
    assert gate["data_manifest_ref_checks"]["avg_tokens_per_doc_positive"] is False
    assert gate["data_manifest_ref_checks"]["vocab_size_positive"] is False
    assert gate["data_manifest_ref_checks"]["manifest_row_count_positive"] is False
    assert gate["data_manifest_ref_checks"]["manifest_source_text_hashes_verified"] is False
    assert gate["data_manifest_ref_checks"]["manifest_source_text_hash_failure_count_zero"] is False
    assert gate["data_manifest_ref_checks"]["manifest_token_hashes_verified"] is False
    assert gate["data_manifest_ref_checks"]["manifest_token_hash_failure_count_zero"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_count_positive"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifacts_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hashes_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hashes_flag"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hash_count_matches"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_metadata_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_name_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_revision_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_license_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_vocab_size_positive"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_special_tokens_present"] is False
    assert gate["data_manifest_ref_checks"]["stats_recipe_name_matches_config"] is False
    assert gate["data_manifest_ref_checks"]["stats_sequence_length_matches_config"] is False
    assert gate["data_manifest_ref_checks"]["source_mixture_present"] is False
    assert gate["data_manifest_ref_checks"]["source_mixture_realized_share_present"] is False
    assert gate["data_manifest_ref_checks"]["source_mixture_expected_tags_realized"] is False


def test_stage0_gate_requires_manifest_hash_failure_counts_and_tokenizer_hash_count(tmp_path: Path) -> None:
    manifest_ref = _data_manifest_ref()
    manifest_ref["manifest_source_text_hash_failure_count"] = 1
    manifest_ref["manifest_token_hash_failure_count"] = 2
    manifest_ref["tokenizer_artifact_count"] = 3
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
        data_manifest_ref=manifest_ref,
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["data_manifest_ref_valid"] is False
    assert gate["data_manifest_ref_checks"]["manifest_source_text_hashes_verified"] is True
    assert gate["data_manifest_ref_checks"]["manifest_source_text_hash_failure_count_zero"] is False
    assert gate["data_manifest_ref_checks"]["manifest_token_hashes_verified"] is True
    assert gate["data_manifest_ref_checks"]["manifest_token_hash_failure_count_zero"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hashes_present"] is True
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hash_count_matches"] is False


def test_stage0_gate_requires_tokenizer_metadata(tmp_path: Path) -> None:
    manifest_ref = _data_manifest_ref()
    manifest_ref["tokenizer"] = {
        "name": "",
        "revision": "",
        "license": "",
        "vocab_size": 0,
        "special_tokens": {},
    }
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
        data_manifest_ref=manifest_ref,
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["data_manifest_ref_valid"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_artifact_hashes_present"] is True
    assert gate["data_manifest_ref_checks"]["tokenizer_metadata_present"] is True
    assert gate["data_manifest_ref_checks"]["tokenizer_name_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_revision_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_license_present"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_vocab_size_positive"] is False
    assert gate["data_manifest_ref_checks"]["tokenizer_special_tokens_present"] is False


def test_stage0_gate_requires_positive_stats_summary_fields(tmp_path: Path) -> None:
    manifest_ref = _data_manifest_ref()
    manifest_ref["num_documents"] = 0
    manifest_ref["avg_tokens_per_doc"] = 0.0
    manifest_ref["vocab_size"] = 0
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
        data_manifest_ref=manifest_ref,
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["data_manifest_ref_valid"] is False
    assert gate["data_manifest_ref_checks"]["num_documents_positive"] is False
    assert gate["data_manifest_ref_checks"]["avg_tokens_per_doc_positive"] is False
    assert gate["data_manifest_ref_checks"]["vocab_size_positive"] is False
    assert gate["data_manifest_ref_checks"]["num_tokens_train_positive"] is True
    assert gate["data_manifest_ref_checks"]["num_tokens_val_positive"] is True


def test_stage0_gate_requires_valid_model_stats(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
        model_stats={"model_name": "baseline"},
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["model_stats_present"] is True
    assert gate["checks"]["model_stats_valid"] is False
    assert gate["model_stats_checks"]["model_name_present"] is True
    assert gate["model_stats_checks"]["parameter_count_positive_integer"] is False


def test_stage0_gate_requires_valid_validation_report(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    (baseline / "lm_eval_report.json").write_text(
        json.dumps(
            {
                "overall_status": "fail",
                "checks": {
                    "eval_log_present": True,
                    "validation_loss_present": True,
                    "perplexity_present": False,
                    "requested_metrics_present": False,
                },
                "metrics": {"validation_loss": 10.0, "perplexity": None},
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["validation_report_present"] is True
    assert gate["checks"]["validation_report_valid"] is False
    assert gate["validation_report_status"] == "fail"
    assert gate["validation_report_checks"]["perplexity_present"] is False


def test_stage0_gate_rechecks_validation_report_metric_values(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    (baseline / "lm_eval_report.json").write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": {
                    "eval_log_present": True,
                    "validation_loss_present": True,
                    "perplexity_present": True,
                    "requested_metrics_present": True,
                },
                "metrics": {
                    "validation_loss": 10.0,
                    "perplexity": 1.0,
                    "tokens_per_second": None,
                },
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report([baseline], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage0_to_1"]

    assert gate["status"] == "warn"
    assert gate["checks"]["validation_report_present"] is True
    assert gate["checks"]["validation_report_valid"] is False
    assert gate["checks"]["validation_report_metrics_valid"] is False
    assert gate["checks"]["validation_metric_tokens_per_second_finite"] is False
    assert gate["validation_report_status"] == "pass"


def test_stage1_gate_requires_valid_routing_report(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
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
        default_routing_metrics=False,
    )

    report_path = make_stage_gate_report([baseline, fixed], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage1_to_2"]

    assert gate["status"] == "warn"
    assert gate["checks"]["routing_report_present"] is True
    assert gate["checks"]["routing_report_valid"] is False
    assert gate["routing_report_status"] == "warn"
    assert gate["routing_report_checks"]["core_route_metrics_present"] is False
    assert gate["routing_report_checks"]["route_path_examples_present"] is False


def test_stage1_gate_requires_active_compute_in_validation_report(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
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
    (fixed / "lm_eval_report.json").write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": {
                    "eval_log_present": True,
                    "validation_loss_present": True,
                    "perplexity_present": True,
                    "requested_metrics_present": True,
                },
                "metrics": {
                    "validation_loss": 10.1,
                    "perplexity": 1.0,
                    "tokens_per_second": 128.0,
                    "active_block_evals_per_token": None,
                },
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report([baseline, fixed], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage1_to_2"]

    assert gate["status"] == "warn"
    assert gate["checks"]["validation_report_present"] is True
    assert gate["checks"]["validation_report_valid"] is False
    assert gate["checks"]["validation_report_metrics_valid"] is False
    assert gate["checks"]["validation_metric_active_block_evals_per_token_finite"] is False
    assert gate["validation_report_checks"]["requested_metrics_present"] is True
    assert gate["validation_report_metrics"]["active_block_evals_per_token"] is None


def test_stage1_gate_requires_eval_log_artifact(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
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
    (fixed / "eval_log.jsonl").unlink()
    (fixed / "lm_eval_report.json").write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "checks": {
                    "eval_log_present": True,
                    "validation_loss_present": True,
                    "perplexity_present": True,
                    "requested_metrics_present": True,
                },
                "metrics": {
                    "validation_loss": 10.1,
                    "perplexity": 1.0,
                    "tokens_per_second": 128.0,
                    "active_block_evals_per_token": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report([baseline, fixed], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage1_to_2"]

    assert gate["status"] == "warn"
    assert gate["checks"]["eval_log_present"] is False
    assert gate["checks"]["validation_report_valid"] is True
    assert gate["validation_report_checks"]["eval_log_present"] is True


def test_stage1_gate_requires_best_checkpoint_artifact(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        stage="stage0_baseline",
        val_loss=10.0,
        train_row={},
        determinism_status="pass",
        resume_event={
            "checkpoint": "checkpoint_latest",
            "resumed_from_step": 1,
            "target_max_steps": 2,
            "optimizer_state_loaded": True,
        },
        baseline_difficulty_report=_baseline_difficulty_report(),
    )
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.1,
        train_row={
            "route_imitation_accuracy": 0.99,
            "position_norm_mean": 1.0,
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
        write_checkpoint_best=False,
    )

    report_path = make_stage_gate_report([baseline, fixed], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage1_to_2"]

    assert gate["status"] == "warn"
    assert gate["checks"]["checkpoint_present"] is True
    assert gate["checks"]["checkpoint_best_present"] is False
    assert gate["checks"]["routing_report_valid"] is True


def test_stage_gate_report_uses_cost_control_report(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 2.0,
            "max_route_steps": 4,
            "forced_max_step_exit_count": 0,
            "forced_max_step_exit_fraction": 0.0,
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
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
        ),
        encoding="utf-8",
    )
    hard_exit_report = tmp_path / "hard_exit_compare.json"
    hard_exit_report.write_text(json.dumps(_hard_exit_compare_report()), encoding="utf-8")
    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
        hard_exit_compare_report_path=hard_exit_report,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["gates"]["stage4_to_5"]
    assert gate["status"] == "pass"
    assert gate["checks"]["cost_control_report_present"] is True
    assert gate["checks"]["cost_control_stage4_output_action_runs"] is True
    assert gate["checks"]["cost_control_hard_exit_enabled"] is True
    assert gate["checks"]["cost_control_active_range_present"] is True
    assert gate["checks"]["cost_control_average_steps_not_increasing"] is True
    assert gate["checks"]["out_by_difficulty_report_present"] is True
    assert gate["checks"]["out_by_difficulty_passed"] is True
    assert gate["checks"]["out_by_difficulty_stage4_reasoning"] is True
    assert gate["checks"]["out_by_difficulty_hard_exit_reasoning"] is True
    assert gate["checks"]["hard_exit_compare_report_present"] is True
    assert gate["checks"]["hard_exit_compare_passed"] is True
    assert gate["checks"]["hard_exit_compute_adjusted_candidate_passed"] is True
    assert "baseline_stage4_scheduled_free_routing" in gate["hard_exit_compare_required_checks"]
    assert "candidate_stage4_output_action" in gate["hard_exit_compare_required_checks"]
    assert "candidate_top1_hard_exit_rule" in gate["hard_exit_compare_required_checks"]
    assert gate["hard_exit_compare_checks"][0]["checks"]["candidate_top1_hard_exit_rule"] is True
    assert gate["checks"]["max_route_steps_recorded"] is True
    assert gate["checks"]["forced_max_step_exit_fallback_recorded"] is True
    assert gate["checks"]["not_never_exit"] is True
    assert gate["checks"]["hard_compute_not_below_easy"] is True
    assert gate["cost_control_status"] == "pass"
    assert gate["hard_exit_compare_status"] == "pass"
    assert gate["forced_max_step_exit_fraction"] == 0.0
    assert report["supplemental_reports"]["cost_control_report"] == str(cost_report)
    assert report["supplemental_reports"]["out_by_difficulty_report"] == str(out_report)
    assert report["supplemental_reports"]["hard_exit_compare_report"] == str(hard_exit_report)


def test_stage2_gate_rejects_boolean_block_histogram_counts(tmp_path: Path) -> None:
    stage2 = _write_run(
        tmp_path,
        "stage2",
        stage="stage2_router_imitation",
        val_loss=10.0,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": True, "1": 1, "2": 1},
        },
        pseudo_route_curriculum_report={
            "overall_status": "pass",
            "checks": {
                "samples_present": True,
                "easy_uses_skip_or_early_exit": True,
                "hard_uses_recurrence": True,
                "out_supervised": True,
            },
        },
    )

    report_path = make_stage_gate_report([stage2], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage2_to_3"]

    assert gate["checks"]["block_usage_non_degenerate"] is False
    assert gate["checks"]["block_load_entropy_present"] is True


def test_stage2_gate_requires_curriculum_behavior_evidence(tmp_path: Path) -> None:
    stage2 = _write_run(
        tmp_path,
        "stage2",
        stage="stage2_router_imitation",
        val_loss=10.0,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        routing={"mode": "pseudo", "pseudo_policy": "sequential"},
    )
    mixed_pseudo = _write_run(
        tmp_path,
        "stage3_pseudo",
        stage="stage3_pseudo_skip_recur",
        val_loss=10.0,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        routing={"mode": "pseudo", "pseudo_policy": "mixed_skip_recur"},
        pseudo_route_curriculum_report={
            "overall_status": "pass",
            "checks": {
                "baseline_samples_present": True,
                "baseline_cross_entropy_numeric": True,
                "baseline_cross_entropy_ordered_by_difficulty": True,
                "difficulty_bins_present": True,
                "mixed_skip_recur_policy": True,
                "stage3_pseudo_skip_recur_stage": True,
                "pseudo_routing_mode": True,
                "easy_uses_skip_or_early_exit": True,
                "hard_uses_recurrence": False,
                "supervised_out_targets_present": True,
                "easy_exits_no_later_than_hard": True,
                "route_length_conditioned_by_difficulty": True,
            },
        },
    )

    report_path = make_stage_gate_report([stage2, mixed_pseudo], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage2_to_3"]

    assert gate["status"] != "pass"
    assert gate["checks"]["pseudo_route_curriculum_hard_targets"] is False
    assert gate["checks"]["pseudo_route_curriculum_passed"] is False


def test_stage2_gate_requires_mixed_pseudo_policy_for_curriculum(tmp_path: Path) -> None:
    stage2 = _write_run(
        tmp_path,
        "stage2",
        stage="stage2_router_imitation",
        val_loss=10.0,
        train_row={
            "route_imitation_accuracy": 0.95,
            "block_load_entropy": 0.5,
            "top1_block_histogram": {"0": 2, "1": 2, "2": 1},
        },
        routing={"mode": "pseudo", "pseudo_policy": "sequential"},
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

    report_path = make_stage_gate_report([stage2], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage2_to_3"]

    assert gate["status"] != "pass"
    assert gate["checks"]["sequential_pseudo_policy"] is True
    assert gate["checks"]["mixed_pseudo_policy"] is False
    assert gate["checks"]["pseudo_route_curriculum_passed"] is False


def test_stage4_gate_rejects_boolean_exit_histogram_counts(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 2.0,
            "max_route_steps": 4,
            "forced_max_step_exit_count": 0,
            "forced_max_step_exit_fraction": 0.0,
            "first_exit_step_histogram": {"1": True, "2": False},
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
    )

    report_path = make_stage_gate_report([stage4], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["checks"]["exit_distribution_present"] is False
    assert gate["checks"]["not_all_immediate_exit"] is False
    assert gate["checks"]["not_never_exit"] is False


def test_stage4_gate_requires_forced_max_step_fallback_metric(tmp_path: Path) -> None:
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
                "analysis": {
                    "status": "pass",
                    "checks": {
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
                    "route_steps_non_decreasing_with_difficulty": True,
                    "active_compute_non_decreasing_with_difficulty": True,
                    "easy_output_probability_at_least_hard": True,
                },
            }
        ),
        encoding="utf-8",
    )
    hard_exit_report = tmp_path / "hard_exit_compare.json"
    hard_exit_report.write_text(json.dumps(_hard_exit_compare_report()), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
        hard_exit_compare_report_path=hard_exit_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["status"] == "warn"
    assert gate["checks"]["max_route_steps_recorded"] is False
    assert gate["checks"]["forced_max_step_exit_fallback_recorded"] is False


def test_stage4_gate_requires_hard_exit_compare_report(tmp_path: Path) -> None:
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
                "analysis": {
                    "status": "pass",
                    "checks": {
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
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
    assert gate["checks"]["cost_control_report_present"] is True
    assert gate["checks"]["out_by_difficulty_report_present"] is True
    assert gate["checks"]["hard_exit_compare_report_present"] is False
    assert gate["checks"]["hard_exit_compare_passed"] is False
    assert gate["checks"]["hard_exit_compute_adjusted_candidate_passed"] is False


def test_stage4_gate_requires_passing_hard_exit_compare_report(tmp_path: Path) -> None:
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
                "analysis": {
                    "status": "pass",
                    "checks": {
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
                    "route_steps_non_decreasing_with_difficulty": True,
                    "active_compute_non_decreasing_with_difficulty": True,
                    "easy_output_probability_at_least_hard": True,
                },
            }
        ),
        encoding="utf-8",
    )
    hard_exit_report = tmp_path / "hard_exit_compare.json"
    hard_exit_report.write_text(
        json.dumps(
            _hard_exit_compare_report(
                overall_status="warn",
                checks={"latency_ratio_within_threshold": False},
            )
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
        hard_exit_compare_report_path=hard_exit_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["status"] == "warn"
    assert gate["checks"]["hard_exit_compare_report_present"] is True
    assert gate["checks"]["hard_exit_compare_passed"] is False
    assert gate["checks"]["hard_exit_compute_adjusted_candidate_passed"] is False
    assert gate["hard_exit_compare_checks"][0]["checks"]["baseline_stage4_scheduled_free_routing"] is True
    assert gate["hard_exit_compare_checks"][0]["checks"]["candidate_stage4_output_action"] is True
    assert gate["hard_exit_compare_checks"][0]["checks"]["latency_ratio_within_threshold"] is False


def test_stage4_gate_requires_stage4_hard_exit_compare_roles(tmp_path: Path) -> None:
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
                "analysis": {
                    "status": "pass",
                    "checks": {
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
                    "route_steps_non_decreasing_with_difficulty": True,
                    "active_compute_non_decreasing_with_difficulty": True,
                    "easy_output_probability_at_least_hard": True,
                },
            }
        ),
        encoding="utf-8",
    )
    hard_exit_report = tmp_path / "hard_exit_compare.json"
    hard_exit_report.write_text(
        json.dumps(
            _hard_exit_compare_report(
                overall_status="warn",
                checks={
                    "baseline_stage4_scheduled_free_routing": False,
                    "candidate_stage4_output_action": False,
                },
            )
        ),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
        hard_exit_compare_report_path=hard_exit_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["status"] == "warn"
    assert gate["checks"]["hard_exit_compare_report_present"] is True
    assert gate["checks"]["hard_exit_compute_adjusted_candidate_passed"] is False
    assert gate["hard_exit_compare_checks"][0]["checks"]["baseline_without_hard_exit"] is True
    assert gate["hard_exit_compare_checks"][0]["checks"]["candidate_with_hard_exit"] is True
    assert gate["hard_exit_compare_checks"][0]["checks"]["baseline_stage4_scheduled_free_routing"] is False
    assert gate["hard_exit_compare_checks"][0]["checks"]["candidate_stage4_output_action"] is False


def test_stage4_gate_requires_top1_hard_exit_rule(tmp_path: Path) -> None:
    stage4 = _write_run(
        tmp_path,
        "stage4",
        stage="stage4_output_action",
        val_loss=10.0,
        train_row={
            "average_route_steps": 2.0,
            "max_route_steps": 4,
            "forced_max_step_exit_count": 0,
            "forced_max_step_exit_fraction": 0.0,
            "first_exit_step_histogram": {"1": 1, "2": 2},
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
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
                    "route_steps_non_decreasing_with_difficulty": True,
                    "active_compute_non_decreasing_with_difficulty": True,
                    "easy_output_probability_at_least_hard": True,
                },
            }
        ),
        encoding="utf-8",
    )
    hard_exit_report = tmp_path / "hard_exit_compare.json"
    hard_exit_report.write_text(
        json.dumps(_hard_exit_compare_report(checks={"candidate_top1_hard_exit_rule": False})),
        encoding="utf-8",
    )

    report_path = make_stage_gate_report(
        [stage4],
        output_path=tmp_path / "gate.json",
        cost_control_report_path=cost_report,
        out_by_difficulty_report_path=out_report,
        hard_exit_compare_report_path=hard_exit_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage4_to_5"]

    assert gate["status"] != "pass"
    assert gate["checks"]["hard_exit_compare_passed"] is True
    assert gate["checks"]["hard_exit_compute_adjusted_candidate_passed"] is False
    assert gate["hard_exit_compare_checks"][0]["checks"]["candidate_top1_hard_exit_rule"] is False


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
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": True,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
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


def test_stage4_gate_requires_average_steps_cost_control_trend(tmp_path: Path) -> None:
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
                "analysis": {
                    "status": "warn",
                    "checks": {
                        "stage4_output_action_runs": True,
                        "hard_exit_enabled": True,
                        "active_compute_range_present": True,
                        "active_compute_not_increasing_with_cost": True,
                        "average_steps_not_increasing_with_cost": False,
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
                    "stage4_output_action_reasoning": True,
                    "hard_exit_reasoning": True,
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
    assert gate["checks"]["cost_control_active_not_increasing"] is True
    assert gate["checks"]["cost_control_average_steps_not_increasing"] is False


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
                "stage3_scheduled_free_routing_stage": True,
                "scheduled_routing_mode": True,
                "scheduled_stage": True,
                "schedule_present": True,
                "schedule_values_numeric": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
                "logged_eval_schedule_values_present": True,
                "logged_eval_router_probability_matches_schedule": True,
                "logged_eval_lambda_route_matches_schedule": True,
            },
        },
    )
    report_path = make_stage_gate_report([stage3], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage3_to_4"]

    assert gate["status"] == "warn"
    assert gate["loss_ratio_vs_stage1"] is None
    assert gate["checks"]["validation_loss_not_collapsed"] is False
    assert gate["checks"]["scheduled_routing_passed"] is True


def test_stage3_gate_requires_scheduled_routing_behavior_evidence(tmp_path: Path) -> None:
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.0,
        train_row={"route_imitation_accuracy": 0.99, "position_norm_mean": 1.0},
    )
    stage3 = _write_run(
        tmp_path,
        "stage3",
        stage="stage3_scheduled_free_routing",
        val_loss=10.1,
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
                "stage3_scheduled_free_routing_stage": True,
                "scheduled_routing_mode": True,
                "scheduled_stage": True,
                "schedule_present": True,
                "schedule_values_numeric": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
                "logged_eval_schedule_values_present": True,
                "logged_eval_router_probability_matches_schedule": False,
                "logged_eval_lambda_route_matches_schedule": True,
            },
        },
    )

    report_path = make_stage_gate_report([fixed, stage3], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage3_to_4"]

    assert gate["status"] != "pass"
    assert gate["checks"]["scheduled_routing_eval_router_probability_matches"] is False
    assert gate["checks"]["scheduled_routing_passed"] is False


def test_stage3_gate_requires_positive_difficulty_step_correlation(tmp_path: Path) -> None:
    fixed = _write_run(
        tmp_path,
        "fixed",
        stage="stage1_fixed_route",
        val_loss=10.0,
        train_row={"route_imitation_accuracy": 0.99, "position_norm_mean": 1.0},
    )
    stage3 = _write_run(
        tmp_path,
        "stage3",
        stage="stage3_scheduled_free_routing",
        val_loss=10.1,
        train_row={
            "route_entropy": 0.5,
            "block_load_entropy": 0.5,
            "route_path_diversity": 0.5,
            "average_route_steps": 2.0,
            "top1_block_histogram": {"0": 1, "1": 1, "2": 1},
        },
        difficulty_report={"sample_count": 3, "difficulty_step_correlation": -0.2},
        scheduled_routing_report={
            "overall_status": "pass",
            "checks": {
                "stage3_scheduled_free_routing_stage": True,
                "scheduled_routing_mode": True,
                "scheduled_stage": True,
                "schedule_present": True,
                "schedule_values_numeric": True,
                "router_probability_monotonic_nondecreasing": True,
                "lambda_route_monotonic_nonincreasing": True,
                "router_probability_increases": True,
                "lambda_route_decays": True,
                "reaches_free_router": True,
                "logged_schedule_values_present": True,
                "logged_router_probability_matches_schedule": True,
                "logged_lambda_route_matches_schedule": True,
                "logged_eval_schedule_values_present": True,
                "logged_eval_router_probability_matches_schedule": True,
                "logged_eval_lambda_route_matches_schedule": True,
            },
        },
    )

    report_path = make_stage_gate_report([fixed, stage3], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage3_to_4"]

    assert gate["status"] == "warn"
    assert gate["checks"]["difficulty_step_correlation_finite"] is True
    assert gate["checks"]["difficulty_step_correlation_positive"] is False
    assert gate["difficulty_step_correlation"] == -0.2


def test_stage3_gate_rejects_boolean_difficulty_step_correlation(tmp_path: Path) -> None:
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
        difficulty_report={"sample_count": 3, "difficulty_step_correlation": True},
    )

    report_path = make_stage_gate_report([stage3], output_path=tmp_path / "gate.json")
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage3_to_4"]

    assert gate["checks"]["difficulty_report_present"] is True
    assert gate["checks"]["difficulty_step_correlation_finite"] is False
    assert gate["checks"]["difficulty_step_correlation_positive"] is False
    assert gate["difficulty_step_correlation"] is None


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


def test_stage5_gate_requires_retention_enabled_and_capacity_evidence(tmp_path: Path) -> None:
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
    retention_report = _global_kv_retention_report(
        checks={
            "global_kv_enabled": False,
            "retention_capacity_present": False,
            "global_cache_slots_present": False,
        }
    )
    retention_report["overall_status"] = "pass"
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
        global_kv_retention_report=retention_report,
    )
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(_long_context_compare_report()), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] != "pass"
    assert gate["checks"]["global_kv_retention_passed"] is True
    assert gate["checks"]["global_kv_retention_enabled"] is False
    assert gate["checks"]["global_kv_retention_capacity_present"] is False
    assert gate["checks"]["global_kv_retention_cache_slots_present"] is False


def test_stage5_gate_requires_long_context_compare_key_checks(tmp_path: Path) -> None:
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
                "global_attention_mass_bounded": True,
                "global_read_gate_nonzero": True,
                "global_read_gate_bounded": True,
                "global_cache_slots_present": True,
                "sink_attention_mass_measured": True,
                "window_attention_mass_measured": True,
                "sink_window_mass_conserved": True,
                "cache_slots_within_retention_capacity": True,
                "read_ratio_measured": True,
                "window_utilization_measured": True,
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
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_passed"] is True
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage5_gate_requires_long_context_input_reports_to_pass(tmp_path: Path) -> None:
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
        global_kv_retention_report=_global_kv_retention_report(),
    )
    compare = _long_context_compare_report(checks={"candidate_report_passed": False})
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_passed"] is True
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage5_gate_requires_long_context_compare_stage_roles(tmp_path: Path) -> None:
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
        global_kv_retention_report=_global_kv_retention_report(),
    )
    compare = _long_context_compare_report()
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare["comparisons"][0]["checks"]["candidate_stage5_global_kv"] = False
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_passed"] is True
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage5_gate_requires_long_context_compare_coverage_checks(tmp_path: Path) -> None:
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
        global_kv_retention_report=_global_kv_retention_report(),
    )
    compare = _long_context_compare_report()
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare["comparisons"][0]["checks"]["candidate_task_family_coverage"] = False
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["long_context_compare_passed"] is True
    assert gate["checks"]["long_context_global_kv_benefit_proxy"] is False


def test_stage5_gate_requires_lm_loss_not_worse_than_stage4(tmp_path: Path) -> None:
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
        global_kv_retention_report=_global_kv_retention_report(),
    )
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(_long_context_compare_report()), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["stage4_reference_validation_loss_present"] is True
    assert gate["checks"]["validation_loss_not_worse_than_stage4"] is False
    assert gate["loss_ratio_vs_stage4"] == 1.1


def test_stage5_gate_requires_local_global_adapter_diagnostics(tmp_path: Path) -> None:
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
    retention_report = _global_kv_retention_report()
    retention_report["checks"]["read_ratio_measured"] = False
    retention_report["checks"]["window_utilization_measured"] = False
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
        global_kv_retention_report=retention_report,
    )
    compare_report = tmp_path / "long_context_compare.json"
    compare_report.write_text(json.dumps(_long_context_compare_report()), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage4, stage5],
        output_path=tmp_path / "gate.json",
        long_context_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage5_to_6"]

    assert gate["status"] == "warn"
    assert gate["checks"]["global_kv_retention_passed"] is True
    assert gate["checks"]["local_global_read_ratio_measured"] is False
    assert gate["checks"]["global_cache_window_utilization_measured"] is False


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
        parallel_passing_report=_parallel_passing_report(),
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
                            "baseline_stage5_global_kv": True,
                            "baseline_scheduled_route_mode": True,
                            "baseline_global_kv_enabled": True,
                            "baseline_parallel_passing_disabled": True,
                            "baseline_topk_weighted_fusion": True,
                            "candidate_parallel_stage": True,
                            "candidate_parallel_route_mode": True,
                            "candidate_parallel_passing_enabled": True,
                            "candidate_global_kv_enabled": True,
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
    assert gate["checks"]["parallel_passing_stage_reported"] is True
    assert gate["checks"]["parallel_passing_enabled"] is True
    assert gate["checks"]["parallel_route_selected"] is True
    assert gate["checks"]["parallel_shared_base_global_memory_enabled"] is True
    assert gate["checks"]["parallel_score_margin_nonnegative"] is True
    assert gate["checks"]["parallel_branch_delta_memory_measured"] is True
    assert gate["checks"]["parallel_delta_cache_nonnegative"] is True
    assert gate["checks"]["parallel_delta_memory_policy_present"] is True
    assert gate["checks"]["parallel_branch_count_bounded_by_beam"] is True
    assert gate["checks"]["parallel_delta_cache_bounded"] is True
    assert gate["checks"]["parallel_branch_score_decay_configured"] is True
    assert gate["checks"]["parallel_compare_report_present"] is True
    assert gate["checks"]["parallel_compare_throughput_not_collapsed"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is True
    assert report["supplemental_reports"]["parallel_compare_report"] == str(compare_report)


def test_stage6_gate_requires_branch_score_and_delta_memory_checks(tmp_path: Path) -> None:
    passing_report = _parallel_passing_report()
    passing_report["checks"]["shared_base_global_memory_enabled"] = False
    passing_report["checks"]["score_margin_nonnegative"] = False
    passing_report["checks"]["branch_score_decay_configured"] = False
    passing_report["checks"]["branch_delta_memory_measured"] = False
    passing_report["checks"]["delta_cache_nonnegative"] = False
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
        parallel_passing_report=passing_report,
    )
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(json.dumps(_parallel_compare_report()), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_passing_report_passed"] is True
    assert gate["checks"]["parallel_shared_base_global_memory_enabled"] is False
    assert gate["checks"]["parallel_score_margin_nonnegative"] is False
    assert gate["checks"]["parallel_branch_score_decay_configured"] is False
    assert gate["checks"]["parallel_branch_delta_memory_measured"] is False
    assert gate["checks"]["parallel_delta_cache_nonnegative"] is False


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
        parallel_passing_report=_parallel_passing_report(),
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
                            "baseline_stage5_global_kv": True,
                            "baseline_scheduled_route_mode": True,
                            "baseline_global_kv_enabled": True,
                            "baseline_parallel_passing_disabled": True,
                            "baseline_topk_weighted_fusion": True,
                            "candidate_parallel_stage": True,
                            "candidate_parallel_route_mode": True,
                            "candidate_parallel_passing_enabled": True,
                            "candidate_global_kv_enabled": True,
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
    assert gate["checks"]["parallel_compare_throughput_not_collapsed"] is True
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
        parallel_passing_report=_parallel_passing_report(),
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
    assert gate["checks"]["parallel_compare_throughput_not_collapsed"] is False
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False


def test_stage6_gate_requires_parallel_compare_throughput(tmp_path: Path) -> None:
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
        parallel_passing_report=_parallel_passing_report(),
    )
    compare = _parallel_compare_report(checks={"throughput_not_collapsed": False})
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] != "pass"
    assert gate["checks"]["parallel_compare_passed"] is True
    assert gate["checks"]["parallel_compare_throughput_not_collapsed"] is False
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False


def test_stage6_gate_requires_parallel_compare_stage_roles(tmp_path: Path) -> None:
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
        parallel_passing_report=_parallel_passing_report(),
    )
    compare = _parallel_compare_report()
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare["comparisons"][0]["checks"]["candidate_parallel_stage"] = False
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_compare_passed"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False


def test_stage6_gate_requires_topk_weighted_parallel_baseline(tmp_path: Path) -> None:
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
        parallel_passing_report=_parallel_passing_report(),
    )
    compare = _parallel_compare_report()
    compare["overall_status"] = "pass"
    compare["comparisons"][0]["status"] = "pass"
    compare["comparisons"][0]["checks"]["baseline_topk_weighted_fusion"] = False
    compare_report = tmp_path / "parallel_compare.json"
    compare_report.write_text(json.dumps(compare), encoding="utf-8")

    report_path = make_stage_gate_report(
        [stage6],
        output_path=tmp_path / "gate.json",
        parallel_compare_report_path=compare_report,
    )
    gate = json.loads(report_path.read_text(encoding="utf-8"))["gates"]["stage6_to_scale"]

    assert gate["status"] == "warn"
    assert gate["checks"]["parallel_compare_passed"] is True
    assert gate["checks"]["parallel_branch_benefit_proxy"] is False
