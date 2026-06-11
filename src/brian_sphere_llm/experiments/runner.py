from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from brian_sphere_llm.eval.compute_report import make_compute_report
from brian_sphere_llm.eval.routing_report import make_routing_report
from brian_sphere_llm.train.trainer import train_from_config
from brian_sphere_llm.utils.config import load_config, save_yaml
from brian_sphere_llm.utils.logging import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]


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
    output_path = output_dir / "experiment_results.json"
    write_json(
        {
            **plan.to_json(),
            "entries": [entry.to_json() for entry in entries],
            "results": results,
            "baseline_run": str(effective_baseline) if effective_baseline else None,
            "compute_report": str(compute_report) if compute_report else None,
        },
        output_path,
    )
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
