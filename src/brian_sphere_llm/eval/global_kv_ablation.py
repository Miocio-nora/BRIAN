from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.experiments.runner import build_experiment_plan
from brian_sphere_llm.utils.logging import write_json


GLOBAL_METRICS = [
    "global_attention_mass",
    "global_sink_attention_mass",
    "global_window_attention_mass",
    "global_read_gate_mean",
    "global_cache_slots_mean",
]


def make_global_kv_ablation_report(
    manifest_path: str | Path,
    run_dirs: list[str | Path],
    *,
    output_path: str | Path | None = None,
    long_context_report_paths: list[str | Path] | None = None,
) -> Path:
    plan = build_experiment_plan(manifest_path, include_baseline=False)
    long_context_by_run = _long_context_reports_by_run(long_context_report_paths or [])
    rows = [
        _summarize_run(
            plan.entries[index].to_json() if index < len(plan.entries) else _extra_entry(index),
            Path(run_dir),
            long_context_by_run,
        )
        for index, run_dir in enumerate(run_dirs)
    ]
    required_checks = _required_checks(rows, expected_entry_count=len(plan.entries))
    optional_checks = _optional_checks(rows)
    checks = {**required_checks, **optional_checks}
    report = {
        "overall_status": _status(required_checks, optional_checks),
        "checks": checks,
        "required_checks": required_checks,
        "optional_checks": optional_checks,
        "manifest": plan.to_json(),
        "expected_entry_count": len(plan.entries),
        "run_count": len(rows),
        "entries": rows,
        "comparisons": {
            "local_vs_global": _local_vs_global(rows),
            "uncompressed_vs_compressed": _uncompressed_vs_compressed(rows),
            "with_sink_vs_no_sink": _with_sink_vs_no_sink(rows),
            "window_sweep": _window_sweep(rows),
            "per_block_vs_compressed": _per_block_vs_compressed(rows),
            "head_delta_vs_per_block": _head_delta_vs_per_block(rows),
        },
    }
    if output_path is None:
        output_path = Path("reports") / "global_kv_ablation_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _summarize_run(
    entry: dict[str, Any],
    run_dir: Path,
    long_context_by_run: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not (run_dir / "routing_report.json").exists() and (run_dir / "train_log.jsonl").exists():
        make_routing_report(run_dir)
    config = _read_yaml_if_exists(run_dir / "config_resolved.yaml")
    model_config = config.get("model_config_resolved")
    model_config = model_config if isinstance(model_config, dict) else {}
    model_stats = _read_json_if_exists(run_dir / "model_stats.json")
    routing_report = _read_json_if_exists(run_dir / "routing_report.json")
    retention_report = _read_json_if_exists(run_dir / "global_kv_retention_report.json")
    eval_rows = _read_jsonl(run_dir / "eval_log.jsonl")
    latest_eval = eval_rows[-1] if eval_rows else routing_report.get("latest_eval", {})
    routing_summary = routing_report.get("summary", {}) if isinstance(routing_report.get("summary"), dict) else {}
    long_context = long_context_by_run.get(_normalize_run_dir(run_dir), {})
    global_kv_enabled = _bool(model_config.get("global_kv", False))
    global_code_dim = int(_num(model_config.get("global_code_dim")) or 0)
    sink_slots = int(_num(model_config.get("global_sink_slots")) or 0)
    window_slots = int(_num(model_config.get("global_window_slots")) or 0)
    adapter_scope = str(model_config.get("global_adapter_scope", "shared"))
    head_delta_rank = int(_num(model_config.get("global_head_delta_rank")) or 0)
    global_metrics = _global_metrics(routing_summary, latest_eval, long_context)
    row = {
        **entry,
        "run_dir": str(run_dir),
        "stage": str(config.get("stage", "")),
        "model_name": str(model_stats.get("model_name", "")),
        "global_kv_enabled": global_kv_enabled,
        "global_code_dim": global_code_dim if global_kv_enabled else 0,
        "global_sink_slots": sink_slots,
        "global_window_slots": window_slots,
        "global_adapter_scope": adapter_scope if global_kv_enabled else "none",
        "global_head_delta_rank": head_delta_rank if global_kv_enabled else 0,
        "global_retention_capacity_slots": sink_slots + window_slots if global_kv_enabled else 0,
        "validation_loss": _num(latest_eval.get("validation_loss")),
        "perplexity": _num(latest_eval.get("perplexity")),
        "global_metrics": global_metrics,
        "retention_report_status": retention_report.get("overall_status"),
        "long_context": _long_context_summary(long_context),
    }
    row["kind"] = _entry_kind(row)
    return row


def _required_checks(rows: list[dict[str, Any]], *, expected_entry_count: int) -> dict[str, bool]:
    global_rows = [row for row in rows if row["global_kv_enabled"]]
    uncompressed_rows = [row for row in rows if row["kind"] == "uncompressed"]
    compressed_rows = [row for row in rows if row["kind"] == "compressed"]
    no_sink_rows = [row for row in rows if row["kind"] == "no_sink"]
    with_sink_rows = [row for row in rows if row["kind"] == "with_sink"]
    window_rows = [row for row in rows if row["kind"] == "window_sweep"]
    per_block_rows = [row for row in rows if row["kind"] == "per_block_adapter"]
    head_delta_rows = [row for row in rows if row["kind"] == "head_delta_adapter"]
    return {
        "runs_match_manifest_entries": len(rows) == expected_entry_count,
        "local_baseline_present": any(row["kind"] == "local" for row in rows),
        "global_candidate_present": bool(global_rows),
        "uncompressed_candidate_present": bool(uncompressed_rows),
        "compressed_candidate_present": bool(compressed_rows),
        "no_sink_candidate_present": bool(no_sink_rows),
        "with_sink_candidate_present": bool(with_sink_rows),
        "with_sink_retention_measured": any(_sink_window_measured(row) for row in with_sink_rows),
        "no_sink_zero_sink_attention_measured": any(_zero_sink_attention(row) for row in no_sink_rows),
        "window_sweep_present": len(window_rows) >= 2,
        "window_slots_vary": len({row["global_window_slots"] for row in window_rows}) >= 2,
        "per_block_adapter_candidate_present": bool(per_block_rows),
        "head_delta_adapter_candidate_present": bool(head_delta_rows),
        "global_metrics_present": bool(global_rows) and all(_global_metrics_present(row) for row in global_rows),
    }


def _optional_checks(rows: list[dict[str, Any]]) -> dict[str, bool]:
    long_context_rows = [row for row in rows if row["long_context"]["present"]]
    return {
        "long_context_reports_present": bool(rows) and len(long_context_rows) == len(rows),
        "long_context_quality_metrics_present": bool(rows)
        and len(long_context_rows) == len(rows)
        and all(_finite(row["long_context"].get("exact_match_accuracy")) for row in long_context_rows),
        "memory_budget_metrics_present": bool(rows)
        and len(long_context_rows) == len(rows)
        and any(_finite(row["long_context"].get("global_cache_capacity_ratio")) for row in long_context_rows),
    }


def _local_vs_global(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local = next((row for row in rows if row["kind"] == "local"), None)
    if local is None:
        return []
    comparisons = []
    for row in rows:
        if not row["global_kv_enabled"]:
            continue
        comparisons.append(
            {
                "entry_id": row["id"],
                "entry_name": row["name"],
                "run_dir": row["run_dir"],
                "validation_loss_delta_vs_local": _delta(row.get("validation_loss"), local.get("validation_loss")),
                "exact_match_delta_vs_local": _delta(
                    row["long_context"].get("exact_match_accuracy"),
                    local["long_context"].get("exact_match_accuracy"),
                ),
                "teacher_forced_token_accuracy_delta_vs_local": _delta(
                    row["long_context"].get("teacher_forced_token_accuracy"),
                    local["long_context"].get("teacher_forced_token_accuracy"),
                ),
                "global_cache_capacity_ratio": row["long_context"].get("global_cache_capacity_ratio"),
                "global_metrics": row["global_metrics"],
            }
        )
    return comparisons


def _with_sink_vs_no_sink(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_sink = next((row for row in rows if row["kind"] == "no_sink"), None)
    with_sink = next((row for row in rows if row["kind"] == "with_sink"), None)
    if no_sink is None or with_sink is None:
        return {"status": "missing"}
    return {
        "status": "present",
        "no_sink_run": no_sink["run_dir"],
        "with_sink_run": with_sink["run_dir"],
        "validation_loss_delta_with_sink_minus_no_sink": _delta(
            with_sink.get("validation_loss"),
            no_sink.get("validation_loss"),
        ),
        "sink_attention_mass_delta": _delta(
            with_sink["global_metrics"].get("global_sink_attention_mass"),
            no_sink["global_metrics"].get("global_sink_attention_mass"),
        ),
        "window_attention_mass_delta": _delta(
            with_sink["global_metrics"].get("global_window_attention_mass"),
            no_sink["global_metrics"].get("global_window_attention_mass"),
        ),
        "exact_match_delta": _delta(
            with_sink["long_context"].get("exact_match_accuracy"),
            no_sink["long_context"].get("exact_match_accuracy"),
        ),
        "teacher_forced_token_accuracy_delta": _delta(
            with_sink["long_context"].get("teacher_forced_token_accuracy"),
            no_sink["long_context"].get("teacher_forced_token_accuracy"),
        ),
    }


def _uncompressed_vs_compressed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    uncompressed = next((row for row in rows if row["kind"] == "uncompressed"), None)
    compressed = next((row for row in rows if row["kind"] == "compressed"), None)
    if uncompressed is None or compressed is None:
        return {"status": "missing"}
    return {
        "status": "present",
        "uncompressed_run": uncompressed["run_dir"],
        "compressed_run": compressed["run_dir"],
        "global_code_dim_uncompressed": uncompressed.get("global_code_dim"),
        "global_code_dim_compressed": compressed.get("global_code_dim"),
        "global_code_dim_delta_compressed_minus_uncompressed": _delta(
            compressed.get("global_code_dim"),
            uncompressed.get("global_code_dim"),
        ),
        "validation_loss_delta_compressed_minus_uncompressed": _delta(
            compressed.get("validation_loss"),
            uncompressed.get("validation_loss"),
        ),
        "exact_match_delta": _delta(
            compressed["long_context"].get("exact_match_accuracy"),
            uncompressed["long_context"].get("exact_match_accuracy"),
        ),
        "teacher_forced_token_accuracy_delta": _delta(
            compressed["long_context"].get("teacher_forced_token_accuracy"),
            uncompressed["long_context"].get("teacher_forced_token_accuracy"),
        ),
        "global_cache_capacity_ratio_delta": _delta(
            compressed["long_context"].get("global_cache_capacity_ratio"),
            uncompressed["long_context"].get("global_cache_capacity_ratio"),
        ),
    }


def _window_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    window_rows = sorted(
        (row for row in rows if row["kind"] == "window_sweep"),
        key=lambda row: row["global_window_slots"],
    )
    return [
        {
            "entry_id": row["id"],
            "entry_name": row["name"],
            "run_dir": row["run_dir"],
            "global_window_slots": row["global_window_slots"],
            "global_retention_capacity_slots": row["global_retention_capacity_slots"],
            "validation_loss": row["validation_loss"],
            "exact_match_accuracy": row["long_context"].get("exact_match_accuracy"),
            "teacher_forced_token_accuracy": row["long_context"].get("teacher_forced_token_accuracy"),
            "global_cache_capacity_ratio": row["long_context"].get("global_cache_capacity_ratio"),
            "global_cache_slots_mean": row["global_metrics"].get("global_cache_slots_mean"),
            "global_attention_mass": row["global_metrics"].get("global_attention_mass"),
        }
        for row in window_rows
    ]


def _per_block_vs_compressed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compressed = next((row for row in rows if row["kind"] == "compressed"), None)
    per_block = next((row for row in rows if row["kind"] == "per_block_adapter"), None)
    if compressed is None or per_block is None:
        return {"status": "missing"}
    return {
        "status": "present",
        "compressed_run": compressed["run_dir"],
        "per_block_run": per_block["run_dir"],
        "global_adapter_scope_compressed": compressed.get("global_adapter_scope"),
        "global_adapter_scope_per_block": per_block.get("global_adapter_scope"),
        "validation_loss_delta_per_block_minus_compressed": _delta(
            per_block.get("validation_loss"),
            compressed.get("validation_loss"),
        ),
        "exact_match_delta": _delta(
            per_block["long_context"].get("exact_match_accuracy"),
            compressed["long_context"].get("exact_match_accuracy"),
        ),
        "teacher_forced_token_accuracy_delta": _delta(
            per_block["long_context"].get("teacher_forced_token_accuracy"),
            compressed["long_context"].get("teacher_forced_token_accuracy"),
        ),
        "global_read_gate_delta": _delta(
            per_block["global_metrics"].get("global_read_gate_mean"),
            compressed["global_metrics"].get("global_read_gate_mean"),
        ),
        "global_cache_capacity_ratio_delta": _delta(
            per_block["long_context"].get("global_cache_capacity_ratio"),
            compressed["long_context"].get("global_cache_capacity_ratio"),
        ),
    }


def _head_delta_vs_per_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_block = next((row for row in rows if row["kind"] == "per_block_adapter"), None)
    head_delta = next((row for row in rows if row["kind"] == "head_delta_adapter"), None)
    if per_block is None or head_delta is None:
        return {"status": "missing"}
    return {
        "status": "present",
        "per_block_run": per_block["run_dir"],
        "head_delta_run": head_delta["run_dir"],
        "global_head_delta_rank_per_block": per_block.get("global_head_delta_rank"),
        "global_head_delta_rank_head_delta": head_delta.get("global_head_delta_rank"),
        "validation_loss_delta_head_delta_minus_per_block": _delta(
            head_delta.get("validation_loss"),
            per_block.get("validation_loss"),
        ),
        "exact_match_delta": _delta(
            head_delta["long_context"].get("exact_match_accuracy"),
            per_block["long_context"].get("exact_match_accuracy"),
        ),
        "teacher_forced_token_accuracy_delta": _delta(
            head_delta["long_context"].get("teacher_forced_token_accuracy"),
            per_block["long_context"].get("teacher_forced_token_accuracy"),
        ),
        "global_read_gate_delta": _delta(
            head_delta["global_metrics"].get("global_read_gate_mean"),
            per_block["global_metrics"].get("global_read_gate_mean"),
        ),
        "global_cache_capacity_ratio_delta": _delta(
            head_delta["long_context"].get("global_cache_capacity_ratio"),
            per_block["long_context"].get("global_cache_capacity_ratio"),
        ),
    }


def _entry_kind(row: dict[str, Any]) -> str:
    entry_id = str(row.get("id", "")).lower()
    name = str(row.get("name", "")).lower()
    if not row["global_kv_enabled"]:
        return "local"
    if (
        entry_id == "c7"
        or "head_delta" in name
        or "per_head" in name
        or (int(_num(row.get("global_head_delta_rank")) or 0) > 0)
    ):
        return "head_delta_adapter"
    if entry_id == "c6" or "per_block" in name or row.get("global_adapter_scope") == "per_block":
        return "per_block_adapter"
    if entry_id == "c1" or "uncompressed" in name:
        return "uncompressed"
    if entry_id == "c2" or "compressed" in name:
        return "compressed"
    if "no_sink" in name or row["global_sink_slots"] == 0:
        return "no_sink"
    if entry_id.startswith(("k5", "c5")) or "window" in name:
        return "window_sweep"
    if "with_sink" in name or entry_id in {"k4", "c4"}:
        return "with_sink"
    return "global"


def _global_metrics(
    routing_summary: dict[str, Any],
    latest_eval: dict[str, Any],
    long_context: dict[str, Any],
) -> dict[str, float | None]:
    long_context_metrics = long_context.get("global_kv", {}) if isinstance(long_context.get("global_kv"), dict) else {}
    values: dict[str, float | None] = {}
    for key in GLOBAL_METRICS:
        values[key] = _first_num(routing_summary.get(key), latest_eval.get(key), long_context_metrics.get(key))
    return values


def _long_context_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "present": False,
            "exact_match_accuracy": None,
            "teacher_forced_token_accuracy": None,
            "global_cache_capacity_ratio": None,
            "global_cache_mean_ratio": None,
        }
    overall = report.get("overall", {}) if isinstance(report.get("overall"), dict) else {}
    memory = report.get("memory_budget", {}) if isinstance(report.get("memory_budget"), dict) else {}
    return {
        "present": True,
        "report_path": report.get("_report_path"),
        "sample_count": _num(report.get("sample_count")),
        "exact_match_accuracy": _num(overall.get("exact_match_accuracy")),
        "teacher_forced_token_accuracy": _num(overall.get("teacher_forced_token_accuracy")),
        "truncation_rate": _num(overall.get("truncation_rate")),
        "global_cache_capacity_ratio": _num(memory.get("estimated_global_cache_capacity_to_local_context_ratio")),
        "global_cache_mean_ratio": _num(memory.get("estimated_global_cache_mean_to_local_context_ratio")),
    }


def _global_metrics_present(row: dict[str, Any]) -> bool:
    metrics = row["global_metrics"]
    return (
        _finite(metrics.get("global_attention_mass"))
        and _finite(metrics.get("global_read_gate_mean"))
        and _finite(metrics.get("global_cache_slots_mean"))
    )


def _sink_window_measured(row: dict[str, Any]) -> bool:
    metrics = row["global_metrics"]
    return _finite(metrics.get("global_sink_attention_mass")) and _finite(metrics.get("global_window_attention_mass"))


def _zero_sink_attention(row: dict[str, Any]) -> bool:
    value = row["global_metrics"].get("global_sink_attention_mass")
    return _finite(value) and abs(float(value)) <= 1e-9


def _status(required_checks: dict[str, bool], optional_checks: dict[str, bool]) -> str:
    if not all(required_checks.values()):
        return "fail"
    if not all(optional_checks.values()):
        return "warn"
    return "pass"


def _long_context_reports_by_run(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    reports = {}
    for path in paths:
        report_path = Path(path)
        report = _read_json_if_exists(report_path)
        run_dir = report.get("run_dir")
        if not run_dir:
            continue
        report["_report_path"] = str(report_path)
        reports[_normalize_run_dir(Path(str(run_dir)))] = report
    return reports


def _normalize_run_dir(path: Path) -> str:
    return str(path.expanduser().resolve())


def _extra_entry(index: int) -> dict[str, str]:
    return {
        "id": f"extra_{index}",
        "name": f"extra_{index}",
        "train_config": "",
        "purpose": "Extra run not mapped to the manifest.",
        "role": "extra",
    }


def _first_num(*values: Any) -> float | None:
    for value in values:
        number = _num(value)
        if number is not None:
            return number
    return None


def _delta(value: Any, baseline: Any) -> float | None:
    left = _num(value)
    right = _num(baseline)
    if left is None or right is None:
        return None
    return left - right


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _num(value: Any) -> float | None:
    if _finite(value):
        return float(value)
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on", "enabled"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}
