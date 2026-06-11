import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.tokenize import SimpleByteTokenizer
from brian_sphere_llm.eval.reasoning import (
    ReasoningSample,
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
