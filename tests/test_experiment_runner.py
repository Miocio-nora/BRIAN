import json
from pathlib import Path

import yaml

from brian_sphere_llm.experiments.runner import build_experiment_plan, run_experiment


def test_build_experiment_plan_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_position_ablations.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_position_ablations"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage0_tiny_debug.yaml"
    assert plan.entries[1].id == "P0"
    assert plan.entries[1].train_config.name == "stage3_no_position_tiny_debug.yaml"


def test_global_kv_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_global_kv.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_global_kv"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage4_tiny_debug.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["K0", "K3", "K4", "K5a", "K5b"]
    assert plan.entries[3].train_config.name == "stage5_tiny_debug.yaml"


def test_parallel_experiment_manifest_resolves_repo_paths() -> None:
    plan = build_experiment_plan("configs/experiments/tiny_parallel_passing.yaml", include_baseline=True)
    assert plan.experiment_name == "tiny_parallel_passing"
    assert plan.entries[0].role == "baseline"
    assert plan.entries[0].train_config.name == "stage5_tiny_debug.yaml"
    assert [entry.id for entry in plan.entries[1:]] == ["PP0", "PP1"]
    assert plan.entries[2].train_config.name == "stage6_tiny_debug.yaml"


def test_run_experiment_dry_run_writes_resolved_plan(tmp_path: Path) -> None:
    output = run_experiment(
        "configs/experiments/tiny_position_ablations.yaml",
        output_dir=tmp_path,
        include_baseline=True,
        limit=2,
        dry_run=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert len(report["entries"]) == 2
    assert report["entries"][0]["role"] == "baseline"


def test_run_experiment_uses_injected_train_and_report_functions(tmp_path: Path) -> None:
    train_config = tmp_path / "train.yaml"
    train_config.write_text("stage: stage0_baseline\n", encoding="utf-8")
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "stub_experiment",
                "ablations": [
                    {
                        "id": "A0",
                        "name": "stub",
                        "train_config": str(train_config),
                        "purpose": "exercise injected runner",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def train_fn(config_path: Path) -> Path:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "train_config.txt").write_text(str(config_path), encoding="utf-8")
        return run_dir

    def routing_fn(run_dir: Path) -> Path:
        path = run_dir / "routing_report.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def compute_fn(run_dirs, *, baseline_run=None, output_path=None):
        path = Path(output_path)
        path.write_text(json.dumps({"runs": list(run_dirs), "baseline_run": baseline_run}), encoding="utf-8")
        return path

    output = run_experiment(
        manifest,
        output_dir=tmp_path / "out",
        train_fn=train_fn,
        routing_report_fn=routing_fn,
        compute_report_fn=compute_fn,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["run_dir"].endswith("/run")
    assert report["results"][0]["generated_train_config"].endswith("A0_stub.yaml")
    assert Path(report["compute_report"]).exists()
