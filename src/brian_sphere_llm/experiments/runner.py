from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from brian_sphere_llm.eval.compute_report import make_compute_report
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.train.trainer import train_from_config
from brian_sphere_llm.utils.config import load_config, save_yaml
from brian_sphere_llm.utils.logging import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_COMPARISON_VIEW_KEYS = [
    "same_parameter_count_view",
    "same_active_compute_view",
    "similar_training_flops_view",
]


@dataclass(frozen=True)
class ExperimentEntry:
    id: str
    name: str
    train_config: Path
    purpose: str = ""
    role: str = "ablation"

    def to_json(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "train_config": str(self.train_config),
            "purpose": self.purpose,
            "role": self.role,
        }


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_name: str
    description: str
    manifest_path: Path
    entries: list[ExperimentEntry]
    baseline_train_config: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "manifest_path": str(self.manifest_path),
            "baseline_train_config": str(self.baseline_train_config) if self.baseline_train_config else None,
            "entries": [entry.to_json() for entry in self.entries],
        }


def build_experiment_plan(manifest_path: str | Path, *, include_baseline: bool = False) -> ExperimentPlan:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_config(manifest_path)
    baseline_config = _optional_resolve(manifest.get("baseline_train_config"), manifest_path)
    entries: list[ExperimentEntry] = []
    if include_baseline and baseline_config is not None:
        entries.append(
            ExperimentEntry(
                id="baseline",
                name="baseline",
                train_config=baseline_config,
                purpose="Reference run for compute and loss comparisons.",
                role="baseline",
            )
        )
    for item in manifest.get("ablations", []):
        if not isinstance(item, dict):
            raise ValueError("Experiment manifest `ablations` entries must be mappings.")
        entries.append(
            ExperimentEntry(
                id=str(item["id"]),
                name=str(item["name"]),
                train_config=_resolve_path(item["train_config"], manifest_path),
                purpose=str(item.get("purpose", "")),
            )
        )
    if not entries:
        raise ValueError("Experiment manifest produced no runnable entries.")
    return ExperimentPlan(
        experiment_name=str(manifest.get("experiment_name", manifest_path.stem)),
        description=str(manifest.get("description", "")),
        manifest_path=manifest_path,
        baseline_train_config=baseline_config,
        entries=entries,
    )


def run_experiment(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    include_baseline: bool = False,
    baseline_run: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    train_fn: Callable[[Path], Path] = train_from_config,
    routing_report_fn: Callable[[Path], Path] = make_routing_report,
    compute_report_fn: Callable[..., Path] = make_compute_report,
) -> Path:
    plan = build_experiment_plan(manifest_path, include_baseline=include_baseline)
    entries = plan.entries[:limit] if limit is not None else plan.entries
    output_dir = Path(output_dir or Path("experiments") / "generated" / plan.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        output_path = output_dir / "experiment_plan.json"
        write_json({**plan.to_json(), "entries": [entry.to_json() for entry in entries], "dry_run": True}, output_path)
        return output_path

    results: list[dict[str, Any]] = []
    materialized_config_dir = output_dir / "generated_train_configs"
    for entry in entries:
        generated_config = _materialize_train_config(plan, entry, materialized_config_dir, output_dir / "runs")
        run_dir = Path(train_fn(generated_config))
        routing_report = routing_report_fn(run_dir)
        results.append(
            {
                **entry.to_json(),
                "generated_train_config": str(generated_config),
                "run_dir": str(run_dir),
                "routing_report": str(routing_report),
            }
        )

    run_dirs = [result["run_dir"] for result in results]
    effective_baseline = baseline_run
    if effective_baseline is None:
        for result in results:
            if result["role"] == "baseline":
                effective_baseline = result["run_dir"]
                break
    compute_report = None
    if run_dirs:
        compute_report = compute_report_fn(
            run_dirs,
            baseline_run=effective_baseline,
            output_path=output_dir / "compute_report.json",
        )
    package_report = make_experiment_package_report(
        plan,
        results,
        entries=entries,
        baseline_run=effective_baseline,
        compute_report_path=compute_report,
        output_path=output_dir / "experiment_package_report.json",
    )
    output_path = output_dir / "experiment_results.json"
    write_json(
        {
            **plan.to_json(),
            "entries": [entry.to_json() for entry in entries],
            "results": results,
            "baseline_run": str(effective_baseline) if effective_baseline else None,
            "compute_report": str(compute_report) if compute_report else None,
            "experiment_package_report": str(package_report),
        },
        output_path,
    )
    return output_path


def make_experiment_package_report(
    plan: ExperimentPlan,
    results: list[dict[str, Any]],
    *,
    entries: list[ExperimentEntry] | None = None,
    baseline_run: str | Path | None = None,
    compute_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    expected_entries = plan.entries if entries is None else entries
    compute_report = _read_json_if_exists(Path(compute_report_path)) if compute_report_path else {}
    compute_rows = compute_report.get("runs", []) if isinstance(compute_report.get("runs"), list) else []
    compute_by_run = _compute_rows_by_run(compute_rows)
    results_by_id = {str(result.get("id")): result for result in results}
    baseline_run_str = str(baseline_run or compute_report.get("baseline_run") or "")
    result_run_dirs = {str(result["run_dir"]) for result in results if result.get("run_dir") is not None}
    compute_run_dirs = set(compute_by_run)

    entry_rows = []
    for entry in expected_entries:
        result = results_by_id.get(entry.id)
        run_dir = str(result.get("run_dir")) if result and result.get("run_dir") is not None else None
        expected_config = _load_config_if_exists(entry.train_config)
        run_config_path = Path(run_dir) / "config_resolved.yaml" if run_dir else None
        run_config = _load_config_if_exists(run_config_path) if run_config_path else {}
        compute_row = compute_by_run.get(run_dir or "")
        has_routing_report = bool(result and _path_exists(result.get("routing_report")))
        has_compute_row = bool(compute_row)
        expected_stage = expected_config.get("stage")
        run_stage = run_config.get("stage")
        compute_stage = compute_row.get("stage") if isinstance(compute_row, dict) else None
        run_config_present = bool(run_config_path and run_config_path.exists())
        run_config_loads = bool(run_config)
        run_stage_matches_entry = run_stage is not None and run_stage == expected_stage
        compute_stage_matches_run = has_compute_row and compute_stage is not None and compute_stage == run_stage
        comparison = compute_row.get("baseline_comparison") if isinstance(compute_row, dict) else None
        baseline_comparison_view_checks = _baseline_comparison_view_checks(comparison)
        baseline_comparison_present = entry.role == "baseline" or bool(isinstance(comparison, dict))
        baseline_comparison_views_present = entry.role == "baseline" or _baseline_comparison_views_present(comparison)
        baseline_comparison_views_passed = entry.role == "baseline" or _baseline_comparison_views_passed(comparison)
        entry_rows.append(
            {
                **entry.to_json(),
                "result_present": result is not None,
                "run_dir": run_dir,
                "routing_report": result.get("routing_report") if result else None,
                "routing_report_present": has_routing_report,
                "run_config": str(run_config_path) if run_config_path else None,
                "run_config_present": run_config_present,
                "run_config_loads": run_config_loads,
                "expected_stage": expected_stage,
                "run_stage": run_stage,
                "compute_stage": compute_stage,
                "run_stage_matches_entry": run_stage_matches_entry,
                "compute_stage_matches_run_config": compute_stage_matches_run,
                "compute_row_present": has_compute_row,
                "baseline_comparison_present": baseline_comparison_present,
                "baseline_comparison_view_checks": baseline_comparison_view_checks,
                "baseline_comparison_views_present": baseline_comparison_views_present,
                "baseline_comparison_views_passed": baseline_comparison_views_passed,
                "status": _entry_status(
                    result_present=result is not None,
                    run_dir_present=bool(run_dir),
                    run_config_present=run_config_present,
                    run_config_loads=run_config_loads,
                    run_stage_matches_entry=run_stage_matches_entry,
                    routing_report_present=has_routing_report,
                    compute_row_present=has_compute_row,
                    compute_stage_matches_run_config=compute_stage_matches_run,
                    baseline_comparison_present=baseline_comparison_present,
                    baseline_comparison_views_present=baseline_comparison_views_present,
                    baseline_comparison_views_passed=baseline_comparison_views_passed,
                    baseline_required=bool(baseline_run_str) and entry.role != "baseline",
                ),
            }
        )

    non_baseline_rows = [row for row in entry_rows if row["role"] != "baseline"]
    checks = {
        "manifest_entries_present": bool(expected_entries),
        "all_entries_have_results": all(row["result_present"] for row in entry_rows),
        "all_results_have_run_dir": all(bool(row["run_dir"]) for row in entry_rows),
        "all_results_have_run_config": all(row["run_config_present"] for row in entry_rows),
        "all_run_configs_load": all(row["run_config_loads"] for row in entry_rows),
        "all_run_stages_match_manifest": all(row["run_stage_matches_entry"] for row in entry_rows),
        "all_results_have_routing_report": all(row["routing_report_present"] for row in entry_rows),
        "compute_report_present": bool(compute_report),
        "compute_report_run_count_matches_results": bool(result_run_dirs)
        and len(compute_run_dirs) == len(result_run_dirs),
        "compute_report_runs_match_results": bool(result_run_dirs) and compute_run_dirs == result_run_dirs,
        "all_results_have_compute_rows": all(row["compute_row_present"] for row in entry_rows),
        "all_compute_rows_match_run_stage": all(row["compute_stage_matches_run_config"] for row in entry_rows),
        "baseline_run_known": bool(baseline_run_str),
        "non_baseline_compute_comparisons_present": bool(non_baseline_rows)
        and all(row["baseline_comparison_present"] for row in non_baseline_rows),
        "non_baseline_compute_comparison_views_present": bool(non_baseline_rows)
        and all(row["baseline_comparison_views_present"] for row in non_baseline_rows),
        "non_baseline_compute_comparison_views_passed": bool(non_baseline_rows)
        and all(row["baseline_comparison_views_passed"] for row in non_baseline_rows),
    }
    report = {
        "experiment_name": plan.experiment_name,
        "description": plan.description,
        "manifest_path": str(plan.manifest_path),
        "expected_entry_count": len(expected_entries),
        "result_count": len(results),
        "baseline_run": baseline_run_str or None,
        "compute_report": str(compute_report_path) if compute_report_path else None,
        "checks": checks,
        "entries": entry_rows,
        "overall_status": _package_status(checks, entry_rows),
    }
    output_path = Path(output_path or Path("reports") / f"{plan.experiment_name}_package_report.json")
    write_json(report, output_path)
    return output_path


def _optional_resolve(value: Any, manifest_path: Path) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, manifest_path)


def _resolve_path(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [
        (manifest_path.parent / path).resolve(),
        (REPO_ROOT / path).resolve(),
        path.resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _materialize_train_config(plan: ExperimentPlan, entry: ExperimentEntry, output_dir: Path, run_output_root: Path) -> Path:
    config = load_config(entry.train_config)
    for key in ("model_config", "data_config"):
        if key in config:
            config[key] = str(_resolve_config_reference(config[key], entry.train_config))
    config["output_root"] = str(run_output_root)
    if config.get("run_name", "auto") == "auto":
        config["run_name"] = f"{plan.experiment_name}_{entry.id}_{entry.name}"
    output_path = output_dir / f"{entry.id}_{entry.name}.yaml"
    save_yaml(config, output_path)
    return output_path


def _resolve_config_reference(value: Any, source_config: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (source_config.parent / path).resolve()


def _compute_rows_by_run(rows: list[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("run_dir") is not None:
            indexed[str(row["run_dir"])] = row
        elif row is not None:
            indexed[str(row)] = {"run_dir": str(row)}
    return indexed


def _entry_status(
    *,
    result_present: bool,
    run_dir_present: bool,
    run_config_present: bool,
    run_config_loads: bool,
    run_stage_matches_entry: bool,
    routing_report_present: bool,
    compute_row_present: bool,
    compute_stage_matches_run_config: bool,
    baseline_comparison_present: bool,
    baseline_comparison_views_present: bool,
    baseline_comparison_views_passed: bool,
    baseline_required: bool,
) -> str:
    required = [
        result_present,
        run_dir_present,
        run_config_present,
        run_config_loads,
        run_stage_matches_entry,
        routing_report_present,
        compute_row_present,
        compute_stage_matches_run_config,
    ]
    if baseline_required:
        required.append(baseline_comparison_present)
        required.append(baseline_comparison_views_present)
        required.append(baseline_comparison_views_passed)
    if all(required):
        return "pass"
    if any(required):
        return "warn"
    return "fail"


def _baseline_comparison_views_present(comparison: Any) -> bool:
    if not isinstance(comparison, dict):
        return False
    return all(key in comparison for key in [*BASELINE_COMPARISON_VIEW_KEYS, "validation_loss_delta"])


def _baseline_comparison_views_passed(comparison: Any) -> bool:
    checks = _baseline_comparison_view_checks(comparison)
    return all(checks[key] is True for key in BASELINE_COMPARISON_VIEW_KEYS) and _finite_number(
        checks["validation_loss_delta"]
    )


def _baseline_comparison_view_checks(comparison: Any) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        return {key: None for key in [*BASELINE_COMPARISON_VIEW_KEYS, "validation_loss_delta"]}
    return {key: comparison.get(key) for key in [*BASELINE_COMPARISON_VIEW_KEYS, "validation_loss_delta"]}


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _package_status(checks: dict[str, bool], entries: list[dict[str, Any]]) -> str:
    entry_statuses = [str(entry.get("status")) for entry in entries]
    if checks and all(checks.values()) and all(status == "pass" for status in entry_statuses):
        return "pass"
    if any(status == "fail" for status in entry_statuses):
        return "fail"
    if any(not passed for passed in checks.values()):
        return "warn"
    return "unknown"


def _path_exists(value: Any) -> bool:
    return value is not None and Path(str(value)).exists()


def _load_config_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return load_config(path)
    except Exception:
        return {}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}
