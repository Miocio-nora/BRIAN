from pathlib import Path

from brian_sphere_llm.eval.post_train_benchmarks import build_benchmark_commands, build_post_train_benchmark_commands


def test_post_train_benchmark_commands_use_fixed_s600_suite(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "candidate"
    config = {
        "post_train_benchmarks": {
            "enabled": True,
            "output_dir": "reports/package_a_benchmarks",
            "label": "candidate_a",
            "checkpoint": "checkpoint_latest",
            "reasoning": {
                "enabled": True,
                "config": "configs/eval/reasoning_eval_s600.yaml",
            },
            "public": {
                "enabled": True,
                "config": "configs/eval/public_benchmark_s600.yaml",
            },
        }
    }

    commands = build_post_train_benchmark_commands(run_dir, config, project_root=tmp_path)

    assert [command["name"] for command in commands] == ["reasoning_s600", "public_s600"]
    assert commands[0]["output_path"].endswith("reports/package_a_benchmarks/candidate_a.reasoning_s600.json")
    assert commands[0]["samples_output_path"].endswith(
        "reports/package_a_benchmarks/candidate_a.reasoning_s600_samples.jsonl"
    )
    assert "--checkpoint" in commands[0]["command"]
    assert "checkpoint_latest" in commands[0]["command"]
    assert commands[1]["output_path"].endswith("reports/package_a_benchmarks/public_candidate_a_s200.json")
    assert commands[1]["samples_output_path"].endswith(
        "reports/package_a_benchmarks/public_candidate_a_s200_samples.jsonl"
    )


def test_checkpoint_benchmark_commands_include_step_label_and_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "candidate"
    config = {
        "checkpoint_benchmarks": {
            "enabled": True,
            "output_dir": "reports/package_a_benchmarks",
            "label": "candidate_a",
            "reasoning": {"enabled": False},
            "public": {
                "enabled": True,
                "config": "configs/eval/public_benchmark_s600.yaml",
            },
        }
    }

    commands = build_benchmark_commands(
        run_dir,
        config,
        cfg_key="checkpoint_benchmarks",
        project_root=tmp_path,
        checkpoint="checkpoint_step_00015000",
        label="candidate_a_step00015000",
    )

    assert [command["name"] for command in commands] == ["public_s600"]
    assert commands[0]["output_path"].endswith(
        "reports/package_a_benchmarks/public_candidate_a_step00015000_s200.json"
    )
    assert "checkpoint_step_00015000" in commands[0]["command"]
