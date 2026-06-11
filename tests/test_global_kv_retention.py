import json
from pathlib import Path

import pytest
import yaml

from brian_sphere_llm.eval.global_kv_retention import make_global_kv_retention_report


def _write_run(
    root: Path,
    *,
    sink_slots: int,
    window_slots: int,
    summary: dict,
) -> Path:
    run_dir = root / "stage5"
    run_dir.mkdir()
    config = {
        "stage": "stage5_global_kv",
        "model_config_resolved": {
            "global_kv": True,
            "global_sink_slots": sink_slots,
            "global_window_slots": window_slots,
        },
    }
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "routing_report.json").write_text(
        json.dumps({"summary": summary, "latest_eval": {}}),
        encoding="utf-8",
    )
    return run_dir


def test_global_kv_retention_report_passes_sink_window_policy(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        sink_slots=1,
        window_slots=3,
        summary={
            "global_attention_mass": 1.0,
            "global_sink_attention_mass": 0.25,
            "global_window_attention_mass": 0.75,
            "global_read_gate_mean": 0.02,
            "global_cache_slots_mean": 2.0,
        },
    )

    report_path = make_global_kv_retention_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["model"]["retention_capacity_slots"] == 4
    assert report["checks"]["sink_slots_configured"] is True
    assert report["checks"]["window_slots_configured"] is True
    assert report["checks"]["global_attention_mass_bounded"] is True
    assert report["checks"]["global_read_gate_bounded"] is True
    assert report["checks"]["sink_window_mass_conserved"] is True
    assert report["checks"]["cache_slots_within_retention_capacity"] is True
    assert report["checks"]["read_ratio_measured"] is True
    assert report["checks"]["window_utilization_measured"] is True
    assert report["metrics"]["local_read_fraction_mean"] == 0.98
    assert report["metrics"]["global_to_local_read_ratio"] == pytest.approx(0.02 / 0.98)
    assert report["metrics"]["local_to_global_read_ratio"] == pytest.approx(0.98 / 0.02)
    assert report["metrics"]["global_cache_window_utilization"] == pytest.approx(1 / 3)
    assert report["metrics"]["global_cache_capacity_utilization"] == 0.5
    assert report["metric_sources"]["global_sink_attention_mass"] == "routing_summary"
    assert report["metric_sources"]["global_cache_window_utilization"] == "derived_from_global_cache_slots_mean"


def test_global_kv_retention_report_fails_missing_sink_or_window_metrics(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        sink_slots=0,
        window_slots=3,
        summary={
            "global_attention_mass": 1.0,
            "global_window_attention_mass": 1.0,
            "global_read_gate_mean": 0.02,
            "global_cache_slots_mean": 2.0,
        },
    )

    report_path = make_global_kv_retention_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["sink_slots_configured"] is False
    assert report["checks"]["sink_attention_mass_measured"] is False


def test_global_kv_retention_report_fails_unbounded_mass_or_gate(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        sink_slots=1,
        window_slots=3,
        summary={
            "global_attention_mass": 1.2,
            "global_sink_attention_mass": 0.6,
            "global_window_attention_mass": 0.6,
            "global_read_gate_mean": 1.1,
            "global_cache_slots_mean": 2.0,
        },
    )

    report_path = make_global_kv_retention_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["global_attention_mass_nonzero"] is True
    assert report["checks"]["global_attention_mass_bounded"] is False
    assert report["checks"]["global_read_gate_nonzero"] is True
    assert report["checks"]["global_read_gate_bounded"] is False


def test_global_kv_retention_report_rejects_boolean_metrics(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        sink_slots=1,
        window_slots=3,
        summary={
            "global_attention_mass": True,
            "global_sink_attention_mass": True,
            "global_window_attention_mass": True,
            "global_read_gate_mean": True,
            "global_cache_slots_mean": True,
        },
    )

    report_path = make_global_kv_retention_report(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["metrics"]["global_attention_mass"] is None
    assert report["checks"]["global_attention_mass_nonzero"] is False
    assert report["checks"]["global_read_gate_nonzero"] is False
    assert report["checks"]["sink_attention_mass_measured"] is False
    assert report["checks"]["window_utilization_measured"] is False


def test_global_kv_retention_report_rejects_invalid_boolean_model_config(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        sink_slots=1,
        window_slots=3,
        summary={
            "global_attention_mass": 1.0,
            "global_sink_attention_mass": 0.25,
            "global_window_attention_mass": 0.75,
            "global_read_gate_mean": 0.02,
            "global_cache_slots_mean": 2.0,
        },
    )
    config_path = run_dir / "config_resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model_config_resolved"]["global_kv"] = "phase_2_only"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_config_resolved.global_kv"):
        make_global_kv_retention_report(run_dir)

    config["model_config_resolved"]["global_kv"] = True
    config["model_config_resolved"]["global_sink_slots"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_config_resolved.global_sink_slots"):
        make_global_kv_retention_report(run_dir)

    config["model_config_resolved"]["global_sink_slots"] = 1
    config["model_config_resolved"]["global_window_slots"] = False
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_config_resolved.global_window_slots"):
        make_global_kv_retention_report(run_dir)
