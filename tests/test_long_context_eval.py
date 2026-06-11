import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.tokenize import SimpleByteTokenizer
from brian_sphere_llm.eval.long_context import (
    LongContextSample,
    _coverage_summary,
    _global_kv_summary,
    _memory_budget_summary,
    evaluate_long_context_sample,
    generate_long_context_samples,
    summarize_long_context_rows,
)


class TinyLongContextModel:
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
                "global_attention_mass": 0.2,
                "global_sink_attention_mass": 0.05,
                "global_window_attention_mass": 0.15,
                "global_read_gate_mean": 0.1,
                "global_cache_slots_mean": 3.0,
            },
        }


def test_generate_long_context_samples_includes_needle_and_two_hop() -> None:
    samples = generate_long_context_samples(4, seed=1, context_length=64)
    assert [sample.task_family for sample in samples] == [
        "needle_retrieval",
        "two_hop_tracing",
        "needle_retrieval",
        "two_hop_tracing",
    ]
    assert all(" Q " in sample.prompt for sample in samples)
    assert all(sample.answer.startswith(" ") for sample in samples)


def test_generate_long_context_samples_covers_package_c_eval_families() -> None:
    families = [
        "needle_retrieval",
        "synthetic_multihop_tracing",
        "ruler_subset",
        "longbench_subset",
        "long_arithmetic_trace",
        "program_trace",
    ]
    samples = generate_long_context_samples(
        len(families),
        seed=2,
        context_length=64,
        task_families=families,
        difficulties=["near"],
    )

    assert [sample.task_family for sample in samples] == families
    assert all(sample.answer.startswith(" ") for sample in samples)
    assert samples[1].key.startswith("K")
    assert samples[2].prompt.startswith("ruler ")
    assert "Question:" in samples[3].prompt
    assert "sum A B C" in samples[4].prompt
    assert "final x" in samples[5].prompt


def test_generate_long_context_samples_rejects_unknown_family_or_difficulty() -> None:
    with pytest.raises(ValueError, match="Unsupported long-context task family"):
        generate_long_context_samples(1, task_families=["missing"], difficulties=["near"])
    with pytest.raises(ValueError, match="Unsupported long-context difficulty"):
        generate_long_context_samples(1, task_families=["needle_retrieval"], difficulties=["extreme"])


def test_summarize_long_context_rows() -> None:
    summary = summarize_long_context_rows(
        [
            {"exact_match": True, "teacher_forced_token_accuracy": 1.0, "truncated": False},
            {"exact_match": False, "teacher_forced_token_accuracy": 0.5, "truncated": True},
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["exact_match_accuracy"] == 0.5
    assert summary["teacher_forced_token_accuracy"] == 0.75
    assert summary["truncation_rate"] == 0.5


def test_long_context_coverage_summary_reports_missing_families_and_difficulties() -> None:
    rows = [
        {"task_family": "needle_retrieval", "difficulty": "near"},
        {"task_family": "ruler_subset", "difficulty": "middle"},
    ]
    summary = _coverage_summary(
        rows,
        ["needle_retrieval", "ruler_subset", "longbench_subset"],
        ["near", "middle", "far"],
    )

    assert summary["observed_task_families"] == ["needle_retrieval", "ruler_subset"]
    assert summary["missing_task_families"] == ["longbench_subset"]
    assert summary["task_family_coverage_passed"] is False
    assert summary["missing_difficulties"] == ["far"]
    assert summary["difficulty_coverage_passed"] is False


def test_memory_budget_summary_estimates_global_cache_ratio() -> None:
    summary = _memory_budget_summary(
        {
            "model_config_resolved": {
                "base": {"layers": 4, "d_model": 64},
                "global_kv": True,
                "global_code_dim": 16,
                "global_sink_slots": 1,
                "global_window_slots": 3,
            },
            "data_config_resolved": {"sequence_length": 8},
        },
        [{"routing_global_cache_slots_mean": 2.0}],
    )
    assert summary["estimated_local_raw_kv_bytes_per_token_fp16"] == 1024.0
    assert summary["estimated_local_raw_kv_context_bytes_fp16"] == 8192.0
    assert summary["estimated_global_cache_capacity_bytes_fp16"] == 128.0
    assert summary["estimated_global_cache_mean_bytes_fp16"] == 64.0
    assert summary["estimated_global_cache_window_used_slots"] == 1.0
    assert summary["estimated_global_cache_window_utilization"] == pytest.approx(1 / 3)
    assert summary["estimated_global_cache_capacity_utilization"] == 0.5
    assert summary["estimated_global_cache_capacity_to_local_context_ratio"] == pytest.approx(128.0 / 8192.0)


def test_long_context_summary_derives_global_read_ratios() -> None:
    summary = _global_kv_summary(
        [
            {
                "exact_match": True,
                "teacher_forced_token_accuracy": 1.0,
                "truncated": False,
                "routing_global_read_gate_mean": 0.25,
            },
            {
                "exact_match": False,
                "teacher_forced_token_accuracy": 0.5,
                "truncated": False,
                "routing_global_read_gate_mean": 0.75,
            },
        ]
    )
    assert summary["global_read_gate_mean"] == 0.5
    assert summary["local_read_fraction_mean"] == 0.5
    assert summary["global_to_local_read_ratio"] == 1.0
    assert summary["local_to_global_read_ratio"] == 1.0


def test_evaluate_long_context_sample_exact_match_with_fake_model() -> None:
    tokenizer = SimpleByteTokenizer()
    sample = LongContextSample(
        task_family="needle_retrieval",
        difficulty="near",
        prompt="ctx n0. K1=42. Q K1? A:",
        answer=" 42",
        key="K1",
    )
    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(sample.prompt, add_special_tokens=False)]
    answer_ids = tokenizer.encode(sample.answer, add_special_tokens=False)
    model = TinyLongContextModel(prompt_length=len(prompt_ids), answer_ids=answer_ids)
    row = evaluate_long_context_sample(
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
    assert row["routing_global_attention_mass"] == 0.2
    assert row["routing_global_sink_attention_mass"] == 0.05
    assert row["routing_global_window_attention_mass"] == 0.15
