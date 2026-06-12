import json
from pathlib import Path

import yaml

from brian_sphere_llm.experiments.runner import build_experiment_plan, make_experiment_package_report, run_experiment
from brian_sphere_llm.train.stage_runner import train_mode_for_stage
from brian_sphere_llm.utils.config import load_config


def test_build_experiment_plan_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_position_ablations"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage0_tiny_debug.yaml"
    assert plan.entries[1].id == "P0"
    assert plan.entries[1].train_config.name == "stage3_no_position_tiny_debug.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    assert plan.entries[2].train_config.name == "stage3_position_random_tiny_debug.yaml"
    assert plan.entries[7].train_config.name == "stage3_position_no_location_bias_tiny_debug.yaml"
    assert plan.entries[8].train_config.name == "stage3_position_no_location_loss_tiny_debug.yaml"
    assert plan.entries[9].train_config.name == "stage3_position_direct_add_tiny_debug.yaml"
    assert plan.entries[10].train_config.name == "stage3_position_separate_state_tiny_debug.yaml"


def test_r125_package_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/route_core_r125_package.yaml", include_baseline=True)
    assert plan.experiment_name == "route_core_r125_package"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage0_baseline.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    assert plan.entries[3].train_config.name == "stage2_sequential_router_imitation.yaml"
    assert plan.entries[8].train_config.name == "ablation_a7_no_location_loss.yaml"
    assert plan.entries[10].train_config.name == "stage3_top2_r125.yaml"
    stages = [load_config(entry.train_config)["stage"] for entry in plan.entries[1:]]
    assert [train_mode_for_stage(stage) for stage in stages] == [
        "baseline",
        "fixed",
        "pseudo",
        "pseudo",
        "scheduled",
        "scheduled",
        "scheduled",
        "scheduled",
        "scheduled",
        "scheduled",
    ]


def test_global_kv_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_global_kv.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_global_kv"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage4_tiny_debug.yaml"
    assert [entry.id for entry in plan.entries[1:]] == [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5a",
        "C5b",
        "C6",
        "C7",
        "C8",
    ]
    assert plan.entries[2].train_config.name == "stage5_tiny_global_kv_uncompressed.yaml"
    assert plan.entries[3].train_config.name == "stage5_tiny_global_kv_compressed.yaml"
    assert plan.entries[5].train_config.name == "stage5_tiny_debug.yaml"
    assert plan.entries[8].train_config.name == "stage5_tiny_global_kv_per_block.yaml"
    assert plan.entries[9].train_config.name == "stage5_tiny_global_kv_per_head_delta.yaml"
    assert plan.entries[10].train_config.name == "stage5_tiny_global_kv_head_delta.yaml"


def test_parallel_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_parallel_passing.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_parallel_passing"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage5_tiny_debug.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["PP0", "PP1", "PP2", "PP3", "PP4", "PP5", "PP6", "PP7"]
    assert plan.entries[2].train_config.name == "stage6_tiny_debug.yaml"
    assert plan.entries[3].train_config.name == "stage6_tiny_parallel_beam4.yaml"
    assert plan.entries[4].train_config.name == "stage6_tiny_parallel_cost_off.yaml"
    assert plan.entries[5].train_config.name == "stage6_tiny_parallel_cost_on.yaml"
    assert plan.entries[6].train_config.name == "stage6_tiny_parallel_top1_exit.yaml"
    assert plan.entries[7].train_config.name == "stage6_tiny_parallel_any_topk_exit.yaml"
    assert plan.entries[8].train_config.name == "stage6_tiny_parallel_global_kv_delta.yaml"


def test_r350_scaling_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/route_core_r350_scaling.yaml", include_baseline=True)
    assert plan.experiment_name == "route_core_r350_scaling"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage0_r350_baseline.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["B0", "B1", "B2", "B3", "B4"]
    assert plan.entries[2].train_config.name == "stage4_r350_output_action.yaml"
    assert plan.entries[4].train_config.name == "ablation_b3_r350_no_output_action.yaml"
    stages = [load_config(entry.train_config)["stage"] for entry in plan.entries[1:]]
    assert [train_mode_for_stage(stage) for stage in stages] == [
        "baseline",
        "scheduled",
        "scheduled",
        "scheduled",
        "pseudo",
    ]


def test_scale_followup_experiment_manifests_resolve_repo_paths() -> None:
    r125 = build_experiment_plan("configs/experiments/route_core_r125_5b_followup.yaml", include_baseline=True)
    r350 = build_experiment_plan("configs/experiments/route_core_r350_30b_followup.yaml", include_baseline=True)
    r1b = build_experiment_plan("configs/experiments/route_core_r1b_main_validation.yaml", include_baseline=True)

    assert r125.experiment_name == "route_core_r125_5b_followup"
    assert [entry.id for entry in r125.entries[1:]] == ["A10", "A11"]
    assert r125.entries[1].train_config.name == "stage0_r125_main5b_baseline.yaml"
    assert r125.entries[2].train_config.name == "stage4_r125_main5b_output_action.yaml"
    assert [train_mode_for_stage(load_config(entry.train_config)["stage"]) for entry in r125.entries[1:]] == [
        "baseline",
        "scheduled",
    ]

    assert r350.experiment_name == "route_core_r350_30b_followup"
    assert [entry.id for entry in r350.entries[1:]] == ["B5", "B6"]
    assert r350.entries[1].train_config.name == "stage0_r350_main30b_baseline.yaml"
    assert r350.entries[2].train_config.name == "stage4_r350_main30b_output_action.yaml"
    assert [train_mode_for_stage(load_config(entry.train_config)["stage"]) for entry in r350.entries[1:]] == [
        "baseline",
        "scheduled",
    ]

    assert r1b.experiment_name == "route_core_r1b_main_validation"
    assert [entry.id for entry in r1b.entries[1:]] == ["D2", "D3"]
    assert r1b.entries[1].train_config.name == "stage0_r1b_main50b_baseline.yaml"
    assert r1b.entries[2].train_config.name == "stage5_r1b_main50b_global_kv.yaml"
    assert [train_mode_for_stage(load_config(entry.train_config)["stage"]) for entry in r1b.entries[1:]] == [
        "baseline",
        "scheduled",
    ]
    assert load_config(r1b.entries[1].train_config)["activation_checkpointing"] is True
    assert load_config(r1b.entries[2].train_config)["activation_checkpointing"] is True
    assert load_config(r1b.entries[1].train_config)["ddp_find_unused_parameters"] is False
    assert load_config(r1b.entries[2].train_config)["ddp_find_unused_parameters"] is True
    assert load_config(r1b.entries[1].train_config)["gradient_accumulation_steps"] == 4
    assert load_config(r1b.entries[2].train_config)["gradient_accumulation_steps"] == 4
    assert load_config(r1b.entries[1].train_config)["lr_schedule"] == "linear_warmup_cosine_decay"
    assert load_config(r1b.entries[2].train_config)["lr_schedule"] == "linear_warmup_cosine_decay"
    assert load_config(r1b.entries[1].train_config)["warmup_steps"] == 2000
    assert load_config(r1b.entries[2].train_config)["warmup_steps"] == 2000


def test_run_experiment_dry_run_writes_resolved_plan(tmp_path: Path) -> None:
    output = run_experiment(
        "configs/experiments/tiny_position_ablations.yaml",
        output_dir=tmp_path,
        include_baseline=True,
        limit=2,
        dry_run=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert len(report["entries"]) == 2
    assert report["entries"][0]["role"] == "baseline"


def test_run_experiment_uses_injected_train_and_report_functions(tmp_path: Path) -> None:
    train_config = tmp_path / "train.yaml"
    train_config.write_text("stage: stage0_baseline\n", encoding="utf-8")
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "stub_experiment",
                "ablations": [
                    {
                        "id": "A0",
                        "name": "stub",
                        "train_config": str(train_config),
                        "purpose": "exercise injected runner",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def train_fn(config_path: Path) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "train_config.txt").write_text(str(config_path), encoding="utf-8")
        return run_dir

    def routing_fn(run_dir: Path) -> Path:
        path = run_dir / "routing_report.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def compute_fn(run_dirs, *, baseline_run=None, output_path=None):
        path = Path(output_path)
        path.write_text(json.dumps({"runs": list(run_dirs), "baseline_run": baseline_run}), encoding="utf-8")
        return path

    output = run_experiment(
        manifest,
        output_dir=tmp_path / "out",
        train_fn=train_fn,
        routing_report_fn=routing_fn,
        compute_report_fn=compute_fn,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["run_dir"].endswith("/run")
    assert report["results"][0]["generated_train_config"].endswith("A0_stub.yaml")
    assert Path(report["compute_report"]).exists()
    assert Path(report["experiment_package_report"]).exists()


def test_experiment_package_report_passes_complete_package(tmp_path: Path) -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    entries = plan.entries[:2]
    baseline_run = tmp_path / "baseline_run"
    candidate_run = tmp_path / "candidate_run"
    baseline_run.mkdir()
    candidate_run.mkdir()
    _write_config_resolved(baseline_run, entries[0])
    _write_config_resolved(candidate_run, entries[1])
    baseline_routing = baseline_run / "routing_report.json"
    candidate_routing = candidate_run / "routing_report.json"
    baseline_routing.write_text("{}", encoding="utf-8")
    candidate_routing.write_text("{}", encoding="utf-8")
    compute_report = tmp_path / "compute_report.json"
    compute_report.write_text(
        json.dumps(
            {
                "baseline_run": str(baseline_run),
                "runs": [
                    {
                        "run_dir": str(baseline_run),
                        "stage": load_config(entries[0].train_config)["stage"],
                        "validation_loss": 10.0,
                    },
                    {
                        "run_dir": str(candidate_run),
                        "stage": load_config(entries[1].train_config)["stage"],
                        "validation_loss": 9.9,
                        "baseline_comparison": {
                            "same_parameter_count_view": True,
                            "same_active_compute_view": True,
                            "similar_training_flops_view": True,
                            "validation_loss_delta": -0.1,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    results = [
        {**entries[0].to_json(), "run_dir": str(baseline_run), "routing_report": str(baseline_routing)},
        {**entries[1].to_json(), "run_dir": str(candidate_run), "routing_report": str(candidate_routing)},
    ]

    output = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=baseline_run,
        compute_report_path=compute_report,
        output_path=tmp_path / "package_report.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["checks"]["all_entries_have_results"] is True
    assert report["checks"]["all_results_have_run_config"] is True
    assert report["checks"]["all_run_stages_match_manifest"] is True
    assert report["checks"]["all_compute_rows_match_run_stage"] is True
    assert report["checks"]["non_baseline_compute_comparisons_present"] is True
    assert report["checks"]["non_baseline_compute_comparison_views_present"] is True
    assert [entry["status"] for entry in report["entries"]] == ["pass", "pass"]


def test_experiment_package_report_warns_missing_compute_row(tmp_path: Path) -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    entries = plan.entries[:2]
    baseline_run = tmp_path / "baseline_run"
    candidate_run = tmp_path / "candidate_run"
    baseline_run.mkdir()
    candidate_run.mkdir()
    _write_config_resolved(baseline_run, entries[0])
    _write_config_resolved(candidate_run, entries[1])
    baseline_routing = baseline_run / "routing_report.json"
    candidate_routing = candidate_run / "routing_report.json"
    baseline_routing.write_text("{}", encoding="utf-8")
    candidate_routing.write_text("{}", encoding="utf-8")
    compute_report = tmp_path / "compute_report.json"
    compute_report.write_text(
        json.dumps(
            {
                "baseline_run": str(baseline_run),
                "runs": [{"run_dir": str(baseline_run), "stage": load_config(entries[0].train_config)["stage"]}],
            }
        ),
        encoding="utf-8",
    )
    results = [
        {**entries[0].to_json(), "run_dir": str(baseline_run), "routing_report": str(baseline_routing)},
        {**entries[1].to_json(), "run_dir": str(candidate_run), "routing_report": str(candidate_routing)},
    ]

    output = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=baseline_run,
        compute_report_path=compute_report,
        output_path=tmp_path / "package_report.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "warn"
    assert report["checks"]["all_results_have_compute_rows"] is False
    assert report["entries"][1]["status"] == "warn"
    assert report["entries"][1]["compute_row_present"] is False


def test_experiment_package_report_warns_incomplete_baseline_comparison(tmp_path: Path) -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    entries = plan.entries[:2]
    baseline_run = tmp_path / "baseline_run"
    candidate_run = tmp_path / "candidate_run"
    baseline_run.mkdir()
    candidate_run.mkdir()
    _write_config_resolved(baseline_run, entries[0])
    _write_config_resolved(candidate_run, entries[1])
    baseline_routing = baseline_run / "routing_report.json"
    candidate_routing = candidate_run / "routing_report.json"
    baseline_routing.write_text("{}", encoding="utf-8")
    candidate_routing.write_text("{}", encoding="utf-8")
    compute_report = tmp_path / "compute_report.json"
    compute_report.write_text(
        json.dumps(
            {
                "baseline_run": str(baseline_run),
                "runs": [
                    {
                        "run_dir": str(baseline_run),
                        "stage": load_config(entries[0].train_config)["stage"],
                        "validation_loss": 10.0,
                    },
                    {
                        "run_dir": str(candidate_run),
                        "stage": load_config(entries[1].train_config)["stage"],
                        "validation_loss": 9.9,
                        "baseline_comparison": {"validation_loss_delta": -0.1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    results = [
        {**entries[0].to_json(), "run_dir": str(baseline_run), "routing_report": str(baseline_routing)},
        {**entries[1].to_json(), "run_dir": str(candidate_run), "routing_report": str(candidate_routing)},
    ]

    output = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=baseline_run,
        compute_report_path=compute_report,
        output_path=tmp_path / "package_report.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "warn"
    assert report["checks"]["non_baseline_compute_comparisons_present"] is True
    assert report["checks"]["non_baseline_compute_comparison_views_present"] is False
    assert report["entries"][1]["baseline_comparison_present"] is True
    assert report["entries"][1]["baseline_comparison_views_present"] is False
    assert report["entries"][1]["status"] == "warn"


def test_experiment_package_report_warns_run_stage_mismatch(tmp_path: Path) -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    entries = plan.entries[:2]
    baseline_run = tmp_path / "baseline_run"
    candidate_run = tmp_path / "candidate_run"
    baseline_run.mkdir()
    candidate_run.mkdir()
    _write_config_resolved(baseline_run, entries[0])
    _write_config_resolved(candidate_run, entries[1], stage="stage1_fixed_route")
    baseline_routing = baseline_run / "routing_report.json"
    candidate_routing = candidate_run / "routing_report.json"
    baseline_routing.write_text("{}", encoding="utf-8")
    candidate_routing.write_text("{}", encoding="utf-8")
    compute_report = tmp_path / "compute_report.json"
    compute_report.write_text(
        json.dumps(
            {
                "baseline_run": str(baseline_run),
                "runs": [
                    {
                        "run_dir": str(baseline_run),
                        "stage": load_config(entries[0].train_config)["stage"],
                        "validation_loss": 10.0,
                    },
                    {
                        "run_dir": str(candidate_run),
                        "stage": "stage1_fixed_route",
                        "validation_loss": 9.9,
                        "baseline_comparison": {
                            "same_parameter_count_view": True,
                            "same_active_compute_view": True,
                            "similar_training_flops_view": True,
                            "validation_loss_delta": -0.1,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    results = [
        {**entries[0].to_json(), "run_dir": str(baseline_run), "routing_report": str(baseline_routing)},
        {**entries[1].to_json(), "run_dir": str(candidate_run), "routing_report": str(candidate_routing)},
    ]

    output = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=baseline_run,
        compute_report_path=compute_report,
        output_path=tmp_path / "package_report.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    candidate = report["entries"][1]

    assert report["overall_status"] == "warn"
    assert report["checks"]["all_run_stages_match_manifest"] is False
    assert candidate["expected_stage"] == load_config(entries[1].train_config)["stage"]
    assert candidate["run_stage"] == "stage1_fixed_route"
    assert candidate["run_stage_matches_entry"] is False


def test_experiment_package_report_warns_compute_stage_mismatch(tmp_path: Path) -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    entries = plan.entries[:2]
    baseline_run = tmp_path / "baseline_run"
    candidate_run = tmp_path / "candidate_run"
    baseline_run.mkdir()
    candidate_run.mkdir()
    _write_config_resolved(baseline_run, entries[0])
    _write_config_resolved(candidate_run, entries[1])
    baseline_routing = baseline_run / "routing_report.json"
    candidate_routing = candidate_run / "routing_report.json"
    baseline_routing.write_text("{}", encoding="utf-8")
    candidate_routing.write_text("{}", encoding="utf-8")
    compute_report = tmp_path / "compute_report.json"
    compute_report.write_text(
        json.dumps(
            {
                "baseline_run": str(baseline_run),
                "runs": [
                    {
                        "run_dir": str(baseline_run),
                        "stage": load_config(entries[0].train_config)["stage"],
                        "validation_loss": 10.0,
                    },
                    {
                        "run_dir": str(candidate_run),
                        "stage": "stage1_fixed_route",
                        "validation_loss": 9.9,
                        "baseline_comparison": {
                            "same_parameter_count_view": True,
                            "same_active_compute_view": True,
                            "similar_training_flops_view": True,
                            "validation_loss_delta": -0.1,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    results = [
        {**entries[0].to_json(), "run_dir": str(baseline_run), "routing_report": str(baseline_routing)},
        {**entries[1].to_json(), "run_dir": str(candidate_run), "routing_report": str(candidate_routing)},
    ]

    output = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=baseline_run,
        compute_report_path=compute_report,
        output_path=tmp_path / "package_report.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    candidate = report["entries"][1]

    assert report["overall_status"] == "warn"
    assert report["checks"]["all_compute_rows_match_run_stage"] is False
    assert candidate["run_stage"] == load_config(entries[1].train_config)["stage"]
    assert candidate["compute_stage"] == "stage1_fixed_route"
    assert candidate["compute_stage_matches_run_config"] is False


def _write_config_resolved(run_dir: Path, entry, *, stage: str | None = None) -> None:
    config = load_config(entry.train_config)
    if stage is not None:
        config["stage"] = stage
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
