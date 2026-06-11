import json
from pathlib import Path

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
    assert report["checks"]["sink_window_mass_conserved"] is True
    assert report["checks"]["cache_slots_within_retention_capacity"] is True
    assert report["metric_sources"]["global_sink_attention_mass"] == "routing_summary"


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
