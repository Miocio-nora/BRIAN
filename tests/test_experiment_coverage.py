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
    assert [row["id"] for row in report["requirements"]] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    assert _requirement(report, "A5")["checks"]["model_flags_match"] is True
    assert _requirement(report, "A7")["checks"]["loss_weights_match"] is True


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
    assert _requirement(report, "C3")["status"] == "pass"
    assert _requirement(report, "C4")["status"] == "pass"
    window = _requirement(report, "C5")
    assert window["status"] == "pass"
    assert len(window["checks"]["distinct_global_window_slots"]) >= 2


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


def test_experiment_coverage_fails_unknown_profile(tmp_path: Path) -> None:
    output = make_experiment_coverage_report(
        "configs/experiments/tiny_position_ablations.yaml",
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
