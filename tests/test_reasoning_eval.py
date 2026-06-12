import pytest

torch = pytest.importorskip("torch")

import brian_sphere_llm.eval.reasoning as reasoning_eval
from brian_sphere_llm.data.tokenize import SimpleByteTokenizer
from brian_sphere_llm.eval.reasoning import (
    ReasoningSample,
    _context_length,
    _load_tokenizer_from_run_config,
    _overall_status,
    _report_checks,
    _routing_summary,
    _visible_cot_token_count,
    evaluate_reasoning_sample,
    generate_reasoning_samples,
    normalize_answer,
    summarize_reasoning_rows,
)


class TinyReasoningModel:
    def __init__(self, prompt_length: int, answer_ids: list[int], vocab_size: int = 260) -> None:
        self.prompt_length = prompt_length
        self.answer_ids = answer_ids
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        for index, token_id in enumerate(self.answer_ids):
            position = self.prompt_length - 1 + index
            if 0 <= position < seq_len:
                logits[:, position, token_id] = 10.0
        generation_index = max(0, seq_len - self.prompt_length)
        next_id = self.answer_ids[min(generation_index, len(self.answer_ids) - 1)]
        logits[:, -1, next_id] = 10.0
        return {
            "logits": logits,
            "routing_summary": {
                "average_route_steps": 2.0,
                "route_entropy": 0.5,
            },
        }


def test_generate_reasoning_samples_cycles_tasks_and_difficulties() -> None:
    samples = generate_reasoning_samples(4, seed=1, task_families=["copy", "arithmetic"], difficulties=["easy", "hard"])
    assert [sample.task_family for sample in samples] == ["copy", "arithmetic", "copy", "arithmetic"]
    assert [sample.difficulty for sample in samples] == ["easy", "hard", "easy", "hard"]
    assert "->" in samples[0].prompt


def test_generate_reasoning_samples_default_includes_transform() -> None:
    samples = generate_reasoning_samples(5, seed=1, difficulties=["easy"])
    transform = samples[2]

    assert transform.task_family == "transform"
    assert transform.prompt.startswith("transform:")
    assert "->" in transform.prompt


def test_normalize_answer_and_summary() -> None:
    assert normalize_answer("  a   b\n") == "a b"
    summary = summarize_reasoning_rows(
        [
            {"exact_match": True, "teacher_forced_token_accuracy": 1.0},
            {"exact_match": False, "teacher_forced_token_accuracy": 0.5, "visible_cot_tokens": 2},
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["exact_match_accuracy"] == 0.5
    assert summary["teacher_forced_token_accuracy"] == 0.75
    assert summary["visible_cot_tokens_mean"] == 2.0


def test_reasoning_report_checks_and_status() -> None:
    checks = _report_checks(
        {
            "sample_count": 2,
            "exact_match_accuracy": 0.5,
            "teacher_forced_token_accuracy": 0.75,
            "visible_cot_tokens_mean": 2.0,
        }
    )

    assert checks == {
        "samples_present": True,
        "exact_match_accuracy_present": True,
        "teacher_forced_token_accuracy_present": True,
        "visible_cot_tokens_present": True,
    }
    assert _overall_status(checks) == "pass"

    warn_checks = {**checks, "visible_cot_tokens_present": False}
    assert _overall_status(warn_checks) == "warn"

    truthy_optional_checks = {**checks, "visible_cot_tokens_present": "yes"}
    assert _overall_status(truthy_optional_checks) == "warn"

    truthy_required_checks = {**checks, "samples_present": "yes"}
    assert _overall_status(truthy_required_checks) == "fail"

    fail_checks = {**checks, "exact_match_accuracy_present": False}
    assert _overall_status(fail_checks) == "fail"


def test_reasoning_summary_rejects_boolean_numeric_metrics() -> None:
    summary = summarize_reasoning_rows(
        [
            {"exact_match": True, "teacher_forced_token_accuracy": True, "generated_token_count": True},
            {"exact_match": False, "teacher_forced_token_accuracy": False, "generated_token_count": False},
        ]
    )
    routing = _routing_summary(
        [
            {"routing_average_route_steps": True, "routing_route_entropy": 0.5},
            {"routing_average_route_steps": False, "routing_route_entropy": 0.25},
        ]
    )

    assert summary["exact_match_accuracy"] == 0.5
    assert summary["teacher_forced_token_accuracy"] is None
    assert summary["generated_tokens_mean"] is None
    assert routing["average_route_steps"] is None
    assert routing["route_entropy"] == 0.375


def test_reasoning_tokenizer_config_parses_string_booleans(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_load_tokenizer(name, *, revision, local_files_only, fallback_to_byte):
        captured.update(
            {
                "name": name,
                "revision": revision,
                "local_files_only": local_files_only,
                "fallback_to_byte": fallback_to_byte,
            }
        )
        return object()

    monkeypatch.setattr(reasoning_eval, "load_tokenizer", fake_load_tokenizer)

    _load_tokenizer_from_run_config(
        {
            "data_config_resolved": {
                "tokenizer": {
                    "name": "unit-tokenizer",
                    "revision": "abc123",
                    "local_files_only": "false",
                    "fallback_to_byte": "true",
                }
            }
        }
    )

    assert captured == {
        "name": "unit-tokenizer",
        "revision": "abc123",
        "local_files_only": False,
        "fallback_to_byte": True,
    }


def test_reasoning_context_length_rejects_boolean_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        _context_length({"data_config_resolved": {"sequence_length": True}})


def test_visible_cot_token_count_uses_answer_suffix() -> None:
    assert _visible_cot_token_count([8, 9], [8, 9]) == 0
    assert _visible_cot_token_count([1, 2, 8, 9], [8, 9]) == 2
    assert _visible_cot_token_count([1, 2, 8], [8, 9]) == 2
    assert _visible_cot_token_count([1, 2], [8, 9]) == 2


def test_evaluate_reasoning_sample_exact_match_with_fake_model() -> None:
    tokenizer = SimpleByteTokenizer()
    sample = ReasoningSample(task_family="copy", difficulty="easy", prompt="copy: 1 ->", answer=" 1")
    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(sample.prompt, add_special_tokens=False)]
    answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)
    model = TinyReasoningModel(prompt_length=len(prompt_ids), answer_ids=answer_ids)
    row = evaluate_reasoning_sample(
        model,
        tokenizer,
        sample,
        config={"stage": "stage0_baseline"},
        route_mode="baseline",
        global_step=0,
        context_length=64,
        sample_id=0,
        device=torch.device("cpu"),
    )
    assert row["exact_match"] is True
    assert row["teacher_forced_token_accuracy"] == 1.0
    assert row["generated_token_count"] == len(answer_ids)
    assert row["answer_token_count"] == len(answer_ids)
    assert row["visible_cot_tokens"] == 0
    assert row["routing_average_route_steps"] == 2.0
