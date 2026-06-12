import json
from pathlib import Path

import pytest
import yaml

from brian_sphere_llm.eval.global_kv_ablation import make_global_kv_ablation_report
from brian_sphere_llm.utils.config import load_config


def test_global_kv_ablation_report_passes_with_manifest_runs_and_long_context(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "local", global_kv=False, sink_slots=0, window_slots=0, validation_loss=10.0),
        _write_run(
            tmp_path,
            "uncompressed",
            global_kv=True,
            global_code_dim=64,
            sink_slots=1,
            window_slots=3,
            validation_loss=10.1,
            sink_mass=0.5,
            window_mass=0.5,
        ),
        _write_run(
            tmp_path,
            "compressed",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=3,
            validation_loss=10.0,
            sink_mass=0.6,
            window_mass=0.4,
        ),
        _write_run(
            tmp_path,
            "no_sink",
            global_kv=True,
            global_code_dim=16,
            sink_slots=0,
            window_slots=3,
            validation_loss=10.1,
            sink_mass=0.0,
            window_mass=1.0,
        ),
        _write_run(
            tmp_path,
            "with_sink",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=3,
            validation_loss=10.0,
            sink_mass=0.6,
            window_mass=0.4,
        ),
        _write_run(
            tmp_path,
            "window1",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=1,
            validation_loss=10.2,
            sink_mass=0.7,
            window_mass=0.3,
        ),
        _write_run(
            tmp_path,
            "window6",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=6,
            validation_loss=9.9,
            sink_mass=0.4,
            window_mass=0.6,
        ),
        _write_run(
            tmp_path,
            "per_block",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=3,
            global_adapter_scope="per_block",
            validation_loss=9.95,
            sink_mass=0.55,
            window_mass=0.45,
        ),
        _write_run(
            tmp_path,
            "head_delta",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=3,
            global_head_delta_rank=2,
            validation_loss=9.94,
            sink_mass=0.55,
            window_mass=0.45,
        ),
        _write_run(
            tmp_path,
            "per_block_head_delta",
            global_kv=True,
            global_code_dim=16,
            sink_slots=1,
            window_slots=3,
            global_adapter_scope="per_block",
            global_head_delta_rank=2,
            validation_loss=9.93,
            sink_mass=0.55,
            window_mass=0.45,
        ),
    ]
    long_context_reports = [
        _write_long_context(tmp_path, run, index, exact=0.5 + index * 0.01, teacher=0.6 + index * 0.01)
        for index, run in enumerate(runs)
    ]

    output = make_global_kv_ablation_report(
        "configs/experiments/tiny_global_kv.yaml",
        runs,
        output_path=tmp_path / "global_kv_ablation.json",
        long_context_report_paths=long_context_reports,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "pass"
    assert report["checks"]["runs_match_manifest_entries"] is True
    assert report["checks"]["uncompressed_candidate_present"] is True
    assert report["checks"]["compressed_candidate_present"] is True
    assert report["checks"]["with_sink_retention_measured"] is True
    assert report["checks"]["no_sink_zero_sink_attention_measured"] is True
    assert report["checks"]["window_slots_vary"] is True
    assert report["checks"]["long_context_reports_match_run_config"] is True
    assert report["checks"]["per_block_adapter_candidate_present"] is True
    assert report["checks"]["head_delta_adapter_candidate_present"] is True
    assert report["checks"]["per_block_head_delta_adapter_candidate_present"] is True
    assert report["comparisons"]["with_sink_vs_no_sink"]["status"] == "present"
    assert report["comparisons"]["uncompressed_vs_compressed"]["status"] == "present"
    assert report["comparisons"]["per_block_vs_compressed"]["status"] == "present"
    assert report["comparisons"]["head_delta_vs_per_block"]["status"] == "present"
    assert report["comparisons"]["uncompressed_vs_compressed"][
        "global_code_dim_delta_compressed_minus_uncompressed"
    ] == -48
    assert report["comparisons"]["with_sink_vs_no_sink"]["sink_attention_mass_delta"] == 0.6
    assert [row["global_window_slots"] for row in report["comparisons"]["window_sweep"]] == [1, 6]
    assert report["comparisons"]["per_block_vs_compressed"]["global_adapter_scope_per_block"] == "per_block"
    assert report["comparisons"]["head_delta_vs_per_block"]["global_head_delta_rank_head_delta"] == 2


def test_global_kv_ablation_report_records_mismatched_long_context_provenance(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "local", global_kv=False, sink_slots=0, window_slots=0),
        _write_run(tmp_path, "uncompressed", global_kv=True, global_code_dim=64, sink_slots=1, window_slots=3),
    ]
    long_context_reports = [
        _write_long_context(tmp_path, runs[0], 0, exact=0.5, teacher=0.6),
        _write_long_context(
            tmp_path,
            runs[1],
            1,
            exact=0.51,
            teacher=0.61,
            stage="stage4_output_action",
            global_kv_enabled=False,
        ),
    ]

    output = make_global_kv_ablation_report(
        "configs/experiments/tiny_global_kv.yaml",
        runs,
        output_path=tmp_path / "global_kv_ablation.json",
        long_context_report_paths=long_context_reports,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    global_row = report["entries"][1]

    assert report["checks"]["long_context_reports_present"] is True
    assert report["checks"]["long_context_reports_match_run_config"] is False
    assert global_row["stage"] == "stage5_global_kv"
    assert global_row["long_context"]["stage"] == "stage4_output_action"
    assert global_row["long_context"]["global_kv_enabled"] is False


def test_global_kv_ablation_report_fails_without_window_sweep(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "local", global_kv=False, sink_slots=0, window_slots=0),
        _write_run(tmp_path, "uncompressed", global_kv=True, global_code_dim=64, sink_slots=1, window_slots=3),
        _write_run(tmp_path, "compressed", global_kv=True, global_code_dim=16, sink_slots=1, window_slots=3),
        _write_run(
            tmp_path,
            "no_sink",
            global_kv=True,
            global_code_dim=16,
            sink_slots=0,
            window_slots=3,
            sink_mass=0.0,
            window_mass=1.0,
        ),
        _write_run(tmp_path, "with_sink", global_kv=True, global_code_dim=16, sink_slots=1, window_slots=3),
    ]

    output = make_global_kv_ablation_report(
        "configs/experiments/tiny_global_kv.yaml",
        runs,
        output_path=tmp_path / "global_kv_ablation.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall_status"] == "fail"
    assert report["checks"]["runs_match_manifest_entries"] is False
    assert report["checks"]["window_sweep_present"] is False


def test_global_kv_ablation_report_rejects_boolean_metrics(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "local", global_kv=False, sink_slots=0, window_slots=0, validation_loss=False),
        _write_run(
            tmp_path,
            "uncompressed",
            global_kv=True,
            global_code_dim=True,
            sink_slots=True,
            window_slots=True,
            validation_loss=True,
            sink_mass=True,
            window_mass=True,
        ),
    ]
    (runs[1] / "routing_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "global_attention_mass": True,
                    "global_sink_attention_mass": True,
                    "global_window_attention_mass": True,
                    "global_read_gate_mean": True,
                    "global_cache_slots_mean": True,
                }
            }
        ),
        encoding="utf-8",
    )
    long_context_reports = [
        _write_long_context(tmp_path, runs[0], 0, exact=False, teacher=False),
        _write_long_context(tmp_path, runs[1], 1, exact=True, teacher=True),
    ]

    output = make_global_kv_ablation_report(
        "configs/experiments/tiny_global_kv.yaml",
        runs,
        output_path=tmp_path / "global_kv_ablation.json",
        long_context_report_paths=long_context_reports,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    global_row = report["entries"][1]

    assert report["overall_status"] == "fail"
    assert global_row["validation_loss"] is None
    assert global_row["global_code_dim"] == 0
    assert global_row["global_metrics"]["global_attention_mass"] is None
    assert global_row["long_context"]["exact_match_accuracy"] is None
    assert report["checks"]["global_metrics_present"] is False
    assert report["checks"]["long_context_quality_metrics_present"] is False


def test_global_kv_ablation_report_rejects_invalid_boolean_model_config(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "local", global_kv=False, sink_slots=0, window_slots=0)
    config_path = run / "config_resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model_config_resolved"]["global_kv"] = "true_after_route_core"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_config_resolved.global_kv"):
        make_global_kv_ablation_report(
            "configs/experiments/tiny_global_kv.yaml",
            [run],
            output_path=tmp_path / "global_kv_ablation.json",
        )


def test_global_kv_ablation_eval_config_resolves() -> None:
    config = load_config("configs/eval/global_kv_ablation.yaml")
    assert config["eval_name"] == "global_kv_ablation_report"


def _write_run(
    root: Path,
    name: str,
    *,
    global_kv: bool,
    sink_slots: int,
    window_slots: int,
    global_code_dim: int = 16,
    validation_loss: float = 10.0,
    sink_mass: float | None = 0.5,
    window_mass: float | None = 0.5,
    global_adapter_scope: str = "shared",
    global_head_delta_rank: int = 0,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    stage = "stage5_global_kv" if global_kv else "stage4_output_action"
    config = {
        "stage": stage,
        "model_config_resolved": {
            "global_kv": global_kv,
            "global_code_dim": global_code_dim if global_kv else 0,
            "global_sink_slots": sink_slots,
            "global_window_slots": window_slots,
            "global_adapter_scope": global_adapter_scope,
            "global_head_delta_rank": global_head_delta_rank,
        },
    }
    routing = {
        "global_attention_mass": 1.0 if global_kv else None,
        "global_sink_attention_mass": sink_mass if global_kv else None,
        "global_window_attention_mass": window_mass if global_kv else None,
        "global_read_gate_mean": 0.02 if global_kv else None,
        "global_cache_slots_mean": 2.0 if global_kv else None,
    }
    routing = {key: value for key, value in routing.items() if value is not None}
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "model_stats.json").write_text(json.dumps({"model_name": name}), encoding="utf-8")
    (run_dir / "eval_log.jsonl").write_text(
        json.dumps({"validation_loss": validation_loss, "perplexity": 100.0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "routing_report.json").write_text(json.dumps({"summary": routing}), encoding="utf-8")
    return run_dir


def _write_long_context(
    tmp_path: Path,
    run_dir: Path,
    index: int,
    *,
    exact: float,
    teacher: float,
    stage: str | None = None,
    route_mode: str = "scheduled",
    global_kv_enabled: bool | None = None,
) -> Path:
    path = tmp_path / f"long_context_{index}.json"
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    model_config = config.get("model_config_resolved", {})
    if stage is None:
        stage = str(config.get("stage", ""))
    if global_kv_enabled is None:
        global_kv_enabled = bool(model_config.get("global_kv", False))
    report = {
        "run_dir": str(run_dir),
        "stage": stage,
        "route_mode": route_mode,
        "sample_count": 4,
        "overall": {
            "exact_match_accuracy": exact,
            "teacher_forced_token_accuracy": teacher,
            "truncation_rate": 0.0,
        },
        "memory_budget": {
            "global_kv_enabled": global_kv_enabled,
            "estimated_global_cache_capacity_to_local_context_ratio": 0.1 + index * 0.01,
            "estimated_global_cache_mean_to_local_context_ratio": 0.05 + index * 0.01,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path
