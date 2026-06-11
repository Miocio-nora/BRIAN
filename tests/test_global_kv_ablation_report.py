import json
from pathlib import Path

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
    assert report["comparisons"]["with_sink_vs_no_sink"]["status"] == "present"
    assert report["comparisons"]["uncompressed_vs_compressed"]["status"] == "present"
    assert report["comparisons"]["uncompressed_vs_compressed"][
        "global_code_dim_delta_compressed_minus_uncompressed"
    ] == -48
    assert report["comparisons"]["with_sink_vs_no_sink"]["sink_attention_mass_delta"] == 0.6
    assert [row["global_window_slots"] for row in report["comparisons"]["window_sweep"]] == [1, 6]


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


def _write_long_context(tmp_path: Path, run_dir: Path, index: int, *, exact: float, teacher: float) -> Path:
    path = tmp_path / f"long_context_{index}.json"
    report = {
        "run_dir": str(run_dir),
        "sample_count": 4,
        "overall": {
            "exact_match_accuracy": exact,
            "teacher_forced_token_accuracy": teacher,
            "truncation_rate": 0.0,
        },
        "memory_budget": {
            "estimated_global_cache_capacity_to_local_context_ratio": 0.1 + index * 0.01,
            "estimated_global_cache_mean_to_local_context_ratio": 0.05 + index * 0.01,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path
