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
    assert _requirement(report, "A5")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A7")["checks"]["loss_weights_match"] is True
    assert _requirement(report, "A8")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A9")["checks"]["model_flags_match"] is True


def test_r350_scaling_package_coverage_passes(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/route_core_r350_scaling.yaml",
        output_path=tmp_path / "coverage.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["profile"] == "package_b_r350_scaling"
    assert [row["id"] for row in report["requirements"]] == ["B0", "B1", "B2", "B3", "B4"]
    assert _requirement(report, "B2")["checks"]["model_flags_match"] is True


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
    assert _requirement(report, "C7")["plan_aliases"] == ["K8"]


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


def _requirement(report: dict, requirement_id: str) -> dict:
    return next(row for row in report["requirements"] if row["id"] == requirement_id)
