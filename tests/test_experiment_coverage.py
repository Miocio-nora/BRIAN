import json
from pathlib import Path

import yaml

from brian_sphere_llm.eval.experiment_coverage import make_experiment_coverage_report
from brian_sphere_llm.utils.config import load_config


def test_r125_formal_package_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r125_package.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "package_a_r125_route_core"
    assert report["checks"]["baseline_train_config_present"] is True
    assert report["checks"]["baseline_train_config_exists"] is True
    assert report["checks"]["baseline_train_config_loads"] is True
    assert report["checks"]["baseline_train_mode_resolves"] is True
    assert report["checks"]["baseline_model_config_valid"] is True
    assert report["checks"]["baseline_data_config_loads"] is True
    assert report["checks"]["baseline_data_config_consistent"] is True
    assert report["checks"]["planned_parameter_estimates_in_range"] is True
    assert report["baseline"]["model"]["estimated_parameter_count_in_plan_range"] is True
    assert report["baseline"]["train_mode"] == "baseline"
    assert [row["id"] for row in report["requirements"]] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
    ]
    assert _requirement(report, "A2")["checks"]["routing_flags_match"] is True
    assert _requirement(report, "A3")["checks"]["routing_flags_match"] is True
    assert _requirement(report, "A5")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A7")["checks"]["loss_weights_match"] is True
    assert _requirement(report, "A8")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A9")["checks"]["model_flags_match"] is True
    assert _entry(report, "A0")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert _entry(report, "A1")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert _entry(report, "A2")["routing"]["pseudo_policy"] == "sequential"
    assert _entry(report, "A3")["routing"]["pseudo_policy"] == "mixed_skip_recur"


def test_r350_scaling_package_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r350_scaling.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "package_b_r350_scaling"
    assert report["checks"]["planned_parameter_estimates_in_range"] is True
    assert [row["id"] for row in report["requirements"]] == ["B0", "B1", "B2", "B3", "B4"]
    assert _requirement(report, "B2")["checks"]["model_flags_match"] is True
    assert _requirement(report, "B4")["checks"]["stage_matches"] is True
    assert _requirement(report, "B4")["checks"]["routing_flags_match"] is True
    for entry_id in ["B0", "B1", "B2", "B3", "B4"]:
        assert _requirement(report, entry_id)["checks"]["train_flags_match"] is True
    assert _entry(report, "B0")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert _entry(report, "B1")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert _entry(report, "B1")["train"]["precision"] == "bf16"
    assert _entry(report, "B4")["stage"] == "stage3_pseudo_skip_recur"
    assert _entry(report, "B4")["routing"]["pseudo_policy"] == "mixed_skip_recur"


def test_r1b_pilot_package_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r1b_pilot.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "package_d_r1b_pilot"
    assert report["checks"]["planned_parameter_estimates_in_range"] is True
    assert [row["id"] for row in report["requirements"]] == ["D0", "D1"]
    assert _requirement(report, "D0")["checks"]["model_flags_match"] is True
    assert _requirement(report, "D0")["checks"]["train_flags_match"] is True
    assert _requirement(report, "D1")["checks"]["model_flags_match"] is True
    assert _requirement(report, "D1")["checks"]["train_flags_match"] is True
    assert _entry(report, "D0")["train"]["precision"] == "bf16"
    assert _entry(report, "D1")["train"]["precision"] == "bf16"
    assert _entry(report, "D0")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert _entry(report, "D1")["model"]["estimated_parameter_count_in_plan_range"] is True
    assert 800_000_000 <= _entry(report, "D0")["model"]["estimated_parameter_count"] <= 1_300_000_000
    assert report["checks"]["baseline_data_config_consistent"] is True


def test_r125_5b_followup_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r125_5b_followup.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "scale_r125_5b_followup"
    assert [row["id"] for row in report["requirements"]] == ["A10", "A11"]
    assert _requirement(report, "A10")["checks"]["data_flags_match"] is True
    assert _requirement(report, "A11")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A11")["checks"]["data_flags_match"] is True
    assert report["checks"]["baseline_data_config_consistent"] is True


def test_r350_30b_followup_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r350_30b_followup.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "scale_r350_30b_followup"
    assert [row["id"] for row in report["requirements"]] == ["B5", "B6"]
    assert _requirement(report, "B5")["checks"]["data_flags_match"] is True
    assert _requirement(report, "B6")["checks"]["model_flags_match"] is True
    assert _requirement(report, "B6")["checks"]["data_flags_match"] is True
    assert report["checks"]["baseline_data_config_consistent"] is True


def test_r1b_main_validation_coverage_passes_with_checkpointing(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r1b_main_validation.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "package_d_r1b_main_validation"
    assert [row["id"] for row in report["requirements"]] == ["D2", "D3"]
    assert _requirement(report, "D2")["checks"]["data_flags_match"] is True
    assert _requirement(report, "D2")["checks"]["train_flags_match"] is True
    assert _requirement(report, "D3")["checks"]["model_flags_match"] is True
    assert _requirement(report, "D3")["checks"]["data_flags_match"] is True
    assert _requirement(report, "D3")["checks"]["train_flags_match"] is True
    assert _entry(report, "D2")["train"]["precision"] == "bf16"
    assert _entry(report, "D3")["train"]["precision"] == "bf16"
    assert report["checks"]["baseline_data_config_consistent"] is True


def test_position_ablation_coverage_passes_geometry_requirements(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_position_ablations.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "block_position_ablation"
    assert [row["id"] for row in report["requirements"]] == [
        "P0",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
    ]
    assert _requirement(report, "P6")["checks"]["model_flags_match"] is True
    assert _requirement(report, "P7")["checks"]["loss_weights_match"] is True
    assert _requirement(report, "P8")["checks"]["model_flags_match"] is True
    assert _requirement(report, "P9")["checks"]["model_flags_match"] is True


def test_cost_control_coverage_passes_loss_weight_sweep(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_cost_control.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "cost_control_sweep"
    assert report["checks"]["model_configs_exist"] is True
    assert report["checks"]["model_configs_load"] is True
    assert report["checks"]["model_configs_valid"] is True
    assert report["checks"]["model_base_configs_exist"] is True
    assert report["checks"]["model_base_configs_load"] is True
    assert report["checks"]["data_configs_exist"] is True
    assert report["checks"]["data_configs_load"] is True
    assert report["checks"]["data_configs_consistent"] is True
    assert len({entry["data_config"] for entry in report["entries"]}) == 1
    assert [row["id"] for row in report["requirements"]] == ["C0", "C1", "C2", "C3"]
    for requirement_id in ["C0", "C1", "C2", "C3"]:
        requirement = _requirement(report, requirement_id)
        assert requirement["checks"]["stage_matches"] is True
        assert requirement["checks"]["mode_matches"] is True
        assert requirement["checks"]["loss_weights_match"] is True


def test_experiment_coverage_fails_inconsistent_data_configs(tmp_path: Path) -> None:
    source = load_config("configs/experiments/route_core_cost_control.yaml")
    mixed_train = tmp_path / "ablation_c3_mixed_data.yaml"
    mixed_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/ablation_c3_cost05.yaml").resolve()),
                "data_config": str(Path("configs/data/r125_tiny_debug.yaml").resolve()),
            }
        ),
        encoding="utf-8",
    )
    source["ablations"][-1]["train_config"] = str(mixed_train)
    manifest = tmp_path / "mixed_data.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="cost_control_sweep",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["data_configs_exist"] is True
    assert report["checks"]["data_configs_consistent"] is False
    assert _requirement(report, "C3")["status"] == "pass"


def test_experiment_coverage_fails_missing_model_config(tmp_path: Path) -> None:
    source = load_config("configs/experiments/route_core_cost_control.yaml")
    broken_train = tmp_path / "ablation_c3_missing_model.yaml"
    broken_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/ablation_c3_cost05.yaml").resolve()),
                "model_config": str(tmp_path / "missing_model_config.yaml"),
            }
        ),
        encoding="utf-8",
    )
    source["ablations"][-1]["train_config"] = str(broken_train)
    manifest = tmp_path / "missing_model_manifest.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="cost_control_sweep",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    broken_entry = next(entry for entry in report["entries"] if entry["id"] == "C3")

    assert report["overall_status"] == "fail"
    assert report["checks"]["model_configs_exist"] is False
    assert report["checks"]["model_configs_load"] is False
    assert report["checks"]["model_configs_valid"] is False
    assert broken_entry["checks"]["model_config_exists"] is False
    assert broken_entry["checks"]["model_config_loads"] is False
    assert broken_entry["checks"]["model_config_valid"] is False
    assert _requirement(report, "C3")["status"] == "pass"


def test_experiment_coverage_fails_data_config_load_error(tmp_path: Path) -> None:
    source = load_config("configs/experiments/route_core_cost_control.yaml")
    bad_data = tmp_path / "bad_data.yaml"
    bad_data.write_text("- not\n- a mapping\n", encoding="utf-8")
    broken_train = tmp_path / "ablation_c3_bad_data.yaml"
    broken_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/ablation_c3_cost05.yaml").resolve()),
                "data_config": str(bad_data),
            }
        ),
        encoding="utf-8",
    )
    source["ablations"][-1]["train_config"] = str(broken_train)
    manifest = tmp_path / "bad_data_manifest.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="cost_control_sweep",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    broken_entry = next(entry for entry in report["entries"] if entry["id"] == "C3")

    assert report["overall_status"] == "fail"
    assert report["checks"]["data_configs_exist"] is True
    assert report["checks"]["data_configs_load"] is False
    assert broken_entry["checks"]["data_config_exists"] is True
    assert broken_entry["checks"]["data_config_loads"] is False
    assert "Expected mapping" in broken_entry["errors"]["data_config"]


def test_experiment_coverage_fails_out_of_range_planned_parameters(tmp_path: Path) -> None:
    source = load_config("configs/experiments/route_core_r1b_pilot.yaml")
    oversized_model = tmp_path / "oversized_1b.yaml"
    oversized_model.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/model/baseline_1b.yaml").resolve()),
                "d_model": 2048,
                "n_heads": 32,
            }
        ),
        encoding="utf-8",
    )
    oversized_train = tmp_path / "oversized_1b_train.yaml"
    oversized_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage0_r1b_baseline.yaml").resolve()),
                "model_config": str(oversized_model),
            }
        ),
        encoding="utf-8",
    )
    source["ablations"][0]["train_config"] = str(oversized_train)
    manifest = tmp_path / "oversized_manifest.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="route_core_r1b_pilot",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["planned_parameter_estimates_in_range"] is False
    assert _entry(report, "D0")["model"]["estimated_parameter_count_in_plan_range"] is False


def test_r1b_pilot_coverage_requires_bf16_precision(tmp_path: Path) -> None:
    source = load_config("configs/experiments/route_core_r1b_pilot.yaml")
    fp32_train = tmp_path / "fp32_r1b_pilot.yaml"
    fp32_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage5_r1b_global_kv_pilot.yaml").resolve()),
                "precision": "fp32",
            }
        ),
        encoding="utf-8",
    )
    source["ablations"][1]["train_config"] = str(fp32_train)
    manifest = tmp_path / "fp32_manifest.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="route_core_r1b_pilot",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert _requirement(report, "D1")["checks"]["train_flags_match"] is False
    assert _entry(report, "D1")["train"]["precision"] == "fp32"


def test_global_kv_package_coverage_passes_window_and_sink_requirements(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_global_kv.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "global_kv_ablation"
    assert _requirement(report, "C0")["status"] == "pass"
    assert _requirement(report, "C1")["status"] == "pass"
    assert _requirement(report, "C2")["status"] == "pass"
    assert _requirement(report, "C2")["matched_entry_ids"] == ["C2"]
    assert _requirement(report, "C3")["status"] == "pass"
    assert _requirement(report, "C4")["status"] == "pass"
    assert _requirement(report, "C4")["matched_entry_ids"] == ["C4"]
    window = _requirement(report, "C5")
    assert window["status"] == "pass"
    assert window["matched_entry_ids"] == ["C5a", "C5b"]
    assert window["plan_aliases"] == ["K5"]
    assert len(window["checks"]["distinct_global_window_slots"]) >= 2
    assert _requirement(report, "C6")["status"] == "pass"
    assert _requirement(report, "C7")["status"] == "pass"
    assert _requirement(report, "C7")["matched_entry_ids"] == ["C7"]
    assert _requirement(report, "C7")["plan_aliases"] == ["K7"]
    assert _requirement(report, "C8")["status"] == "pass"
    assert _requirement(report, "C8")["matched_entry_ids"] == ["C8"]
    assert _requirement(report, "C8")["plan_aliases"] == ["K8"]


def test_parallel_package_coverage_passes_weighted_and_beam2_requirements(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_parallel_passing.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "parallel_passing_beta"
    assert _requirement(report, "PP0")["status"] == "pass"
    assert _requirement(report, "PP1")["status"] == "pass"
    assert _requirement(report, "PP2")["status"] == "pass"
    assert _requirement(report, "PP3")["status"] == "pass"
    assert _requirement(report, "PP4")["status"] == "pass"
    assert _requirement(report, "PP5")["status"] == "pass"
    assert _requirement(report, "PP6")["status"] == "pass"
    assert _requirement(report, "PP7")["status"] == "pass"
    assert _entry(report, "PP1")["model"]["branch_score_decay"] == 0.99
    assert _entry(report, "PP3")["model"]["branch_score_decay"] == 0.99


def test_parallel_package_coverage_requires_branch_score_decay(tmp_path: Path) -> None:
    manifest = tmp_path / "parallel_bad_decay.yaml"
    source = load_config("configs/experiments/route_core_parallel_passing.yaml")
    bad_model = tmp_path / "bad_parallel_model.yaml"
    bad_model.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/model/brian_r125_parallel.yaml").resolve()),
                "branch_score_decay": 1.0,
            }
        ),
        encoding="utf-8",
    )
    bad_train = tmp_path / "bad_parallel_train.yaml"
    bad_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage6_parallel_passing.yaml").resolve()),
                "model_config": str(bad_model),
            }
        ),
        encoding="utf-8",
    )
    for row in source["ablations"]:
        if row["id"] == "PP1":
            row["train_config"] = str(bad_train)
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="parallel_passing_beta",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert _requirement(report, "PP1")["status"] == "fail"
    assert _entry(report, "PP1")["model"]["branch_score_decay"] == 1.0


def test_experiment_coverage_fails_missing_required_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_a7.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    source["ablations"] = [row for row in source["ablations"] if row["id"] != "A7"]
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert _requirement(report, "A7")["checks"]["entry_present"] is False


def test_package_a_coverage_requires_a2_sequential_pseudo_policy(tmp_path: Path) -> None:
    manifest = tmp_path / "wrong_a2_policy.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    mixed_a2_train = tmp_path / "mixed_a2_train.yaml"
    mixed_a2_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage2_sequential_router_imitation.yaml").resolve()),
                "routing": {
                    "mode": "pseudo",
                    "pseudo_policy": "mixed_skip_recur",
                },
            }
        ),
        encoding="utf-8",
    )
    for row in source["ablations"]:
        if row["id"] == "A2":
            row["train_config"] = str(mixed_a2_train)
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    requirement = _requirement(report, "A2")

    assert report["overall_status"] == "fail"
    assert requirement["checks"]["entry_present"] is True
    assert requirement["checks"]["stage_matches"] is True
    assert requirement["checks"]["mode_matches"] is True
    assert requirement["checks"]["routing_flags_match"] is False
    assert _entry(report, "A2")["routing"]["pseudo_policy"] == "mixed_skip_recur"


def test_package_b_coverage_requires_b4_mixed_skip_recur_policy(tmp_path: Path) -> None:
    manifest = tmp_path / "wrong_b4_policy.yaml"
    source = load_config("configs/experiments/route_core_r350_scaling.yaml")
    sequential_b4_train = tmp_path / "sequential_b4_train.yaml"
    sequential_b4_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/ablation_b4_r350_difficulty_route.yaml").resolve()),
                "stage": "stage3_pseudo_skip_recur",
                "routing": {
                    "mode": "pseudo",
                    "pseudo_policy": "sequential",
                },
                "precision": "bf16",
            }
        ),
        encoding="utf-8",
    )
    for row in source["ablations"]:
        if row["id"] == "B4":
            row["train_config"] = str(sequential_b4_train)
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_b",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    requirement = _requirement(report, "B4")

    assert report["overall_status"] == "fail"
    assert requirement["checks"]["entry_present"] is True
    assert requirement["checks"]["stage_matches"] is True
    assert requirement["checks"]["mode_matches"] is True
    assert requirement["checks"]["routing_flags_match"] is False
    assert _entry(report, "B4")["routing"]["pseudo_policy"] == "sequential"


def test_experiment_coverage_rejects_boolean_numeric_model_flags(tmp_path: Path) -> None:
    manifest = tmp_path / "boolean_top_k.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    bool_model = tmp_path / "bool_top_k_model.yaml"
    bool_model.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/model/brian_r125.yaml").resolve()),
                "top_k": True,
            }
        ),
        encoding="utf-8",
    )
    bool_train = tmp_path / "bool_top_k_train.yaml"
    bool_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage3_scheduled_free_routing.yaml").resolve()),
                "model_config": str(bool_model),
            }
        ),
        encoding="utf-8",
    )
    for row in source["ablations"]:
        if row["id"] == "A8":
            row["train_config"] = str(bool_train)
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    requirement = _requirement(report, "A8")

    assert report["overall_status"] == "fail"
    assert requirement["checks"]["entry_present"] is True
    assert requirement["checks"]["model_flags_match"] is False
    assert next(entry for entry in report["entries"] if entry["id"] == "A8")["model"]["top_k"] is True


def test_experiment_coverage_fails_missing_baseline_train_config(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_baseline.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    source.pop("baseline_train_config")
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["baseline_train_config_present"] is False
    assert all(requirement["status"] == "pass" for requirement in report["requirements"])


def test_experiment_coverage_fails_broken_baseline_train_config_path(tmp_path: Path) -> None:
    manifest = tmp_path / "broken_baseline.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    source["baseline_train_config"] = "configs/train/missing_baseline.yaml"
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["baseline_train_config_present"] is True
    assert report["checks"]["baseline_train_config_exists"] is False
    assert report["checks"]["baseline_train_config_loads"] is False
    assert report["baseline"]["checks"]["train_config_exists"] is False
    assert all(requirement["status"] == "pass" for requirement in report["requirements"])


def test_experiment_coverage_fails_baseline_data_config_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "mixed_baseline_data.yaml"
    source = load_config("configs/experiments/route_core_r125_package.yaml")
    baseline_train = tmp_path / "baseline_mixed_data.yaml"
    baseline_train.write_text(
        yaml.safe_dump(
            {
                "extends": str(Path("configs/train/stage0_baseline.yaml").resolve()),
                "data_config": str(Path("configs/data/r125_tiny_debug.yaml").resolve()),
            }
        ),
        encoding="utf-8",
    )
    source["baseline_train_config"] = str(baseline_train)
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="package_a",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["baseline_data_config_loads"] is True
    assert report["checks"]["baseline_data_config_consistent"] is False
    assert all(requirement["status"] == "pass" for requirement in report["requirements"])


def test_global_kv_coverage_requires_dedicated_window_sweep_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_c5.yaml"
    source = load_config("configs/experiments/route_core_global_kv.yaml")
    source["ablations"] = [row for row in source["ablations"] if not str(row["id"]).startswith("C5")]
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="global_kv_ablation",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    window = _requirement(report, "C5")
    assert report["overall_status"] == "fail"
    assert window["status"] == "fail"
    assert window["matched_entry_ids"] == []
    assert window["checks"]["global_window_entry_ids"]


def test_global_kv_coverage_requires_per_block_head_delta_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_c8.yaml"
    source = load_config("configs/experiments/route_core_global_kv.yaml")
    source["ablations"] = [row for row in source["ablations"] if row["id"] != "C8"]
    manifest.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
        profile="global_kv_ablation",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    combined = _requirement(report, "C8")

    assert report["overall_status"] == "fail"
    assert combined["status"] == "fail"
    assert combined["matched_entry_ids"] == []
    assert combined["checks"]["matched"] is False


def test_experiment_coverage_fails_unknown_profile(tmp_path: Path) -> None:
    manifest = tmp_path / "unknown.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "unknown_experiment",
                "ablations": [
                    {
                        "id": "X0",
                        "name": "stub",
                        "train_config": "configs/train/stage0_tiny_debug.yaml",
                        "purpose": "exercise unknown profile handling",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = make_experiment_coverage_report(
        manifest,
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["profile_known"] is False
    assert report["requirements"] == []


def test_experiment_coverage_eval_config_resolves() -> None:
    config = load_config("configs/eval/experiment_coverage.yaml")
    assert config["eval_name"] == "experiment_coverage_report"
    assert config["profile"] == "auto"
    assert load_config("configs/eval/r125_5b_followup_coverage.yaml")["profile"] == "route_core_r125_5b_followup"
    assert load_config("configs/eval/r350_30b_followup_coverage.yaml")["profile"] == "route_core_r350_30b_followup"
    assert load_config("configs/eval/r1b_main_validation_coverage.yaml")["profile"] == "route_core_r1b_main_validation"


def _requirement(report: dict, requirement_id: str) -> dict:
    return next(row for row in report["requirements"] if row["id"] == requirement_id)


def _entry(report: dict, entry_id: str) -> dict:
    return next(row for row in report["entries"] if row["id"] == entry_id)
