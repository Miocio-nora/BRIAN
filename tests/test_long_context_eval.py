import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.tokenize import SimpleByteTokenizer
from brian_sphere_llm.eval.long_context import (
    LongContextSample,
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
