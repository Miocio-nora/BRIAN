from brian_sphere_llm.data.synthetic_routing import TASKS, generate_synthetic_samples, pseudo_route_metadata


def test_generate_synthetic_samples_covers_guidance_task_families_and_route_types() -> None:
    samples = list(generate_synthetic_samples(21, seed=1))
    task_families = {str(sample.metadata["task_family"]) for sample in samples}
    route_types = {str(sample.metadata["pseudo_route_type"]) for sample in samples}
    difficulties = {str(sample.metadata["difficulty_bin"]) for sample in samples}

    assert task_families == set(TASKS)
    assert route_types == {"advance", "early_exit", "late_exit", "mixed", "recur", "skip"}
    assert difficulties == {"easy", "medium", "hard"}


def test_synthetic_samples_include_required_pseudo_route_metadata() -> None:
    sample = next(generate_synthetic_samples(1, seed=2))

    assert {
        "task_family",
        "pseudo_route_type",
        "pseudo_route_length",
        "expected_recurrence_count",
        "expected_skip_count",
        "difficulty_bin",
    } <= set(sample.metadata)
    assert isinstance(sample.metadata["pseudo_route_length"], int)


def test_pseudo_route_metadata_rejects_unknown_difficulty() -> None:
    try:
        pseudo_route_metadata("missing", "copy")
    except ValueError as exc:
        assert "Unsupported difficulty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
