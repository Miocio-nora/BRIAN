from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

import brian_sphere_llm.eval.classic_math as classic_math
from brian_sphere_llm.utils.config import save_yaml


class _FakeDataset(list):
    _fingerprint = "unit-fingerprint"


class _TinyTokenizer:
    bos_token_id = 256
    eos_token_id = 257

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        return [self.bos_token_id, *tokens, self.eos_token_id] if add_special_tokens else tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return bytes(token for token in token_ids if 0 <= token < 256).decode("utf-8")


class _TinyModel:
    def eval(self) -> "_TinyModel":
        return self


def test_gsm8k_and_math500_rows_use_final_gold_answers() -> None:
    gsm = classic_math._math_sample(
        "gsm8k",
        3,
        {"question": "What is 9 + 9?", "answer": "Nine plus nine is eighteen.\n#### 18"},
    )
    math = classic_math._math_sample(
        "math500",
        4,
        {
            "problem": "Write one half.",
            "answer": r"\frac{1}{2}",
            "subject": "Algebra",
            "level": 1,
            "unique_id": "test/algebra/4.json",
        },
    )

    assert gsm.answer == "18"
    assert math.answer == r"\frac{1}{2}"
    assert math.source_id == "test/algebra/4.json"
    assert math.level == "1"


def test_math_verify_accepts_numeric_and_symbolic_equivalence() -> None:
    pytest.importorskip("math_verify")

    assert classic_math.grade_math_answer("18", "The final answer is 18.")[:2] == (True, True)
    assert classic_math.grade_math_answer(r"\frac{1}{2}", "Final answer is $0.5$.")[:2] == (True, True)
    assert classic_math.grade_math_answer("18", "Final answer is 17.")[:2] == (True, False)


def test_math_sampling_is_task_stable_and_records_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _FakeDataset(
        {
            "question": f"Question {index}",
            "answer": f"work\n#### {index}",
        }
        for index in range(20)
    )
    monkeypatch.setattr(classic_math, "load_dataset", lambda *_args, **_kwargs: rows)

    first, first_contract = classic_math.load_math_samples("gsm8k", sample_count=5, seed=7)
    second, second_contract = classic_math.load_math_samples("gsm8k", sample_count=5, seed=7)

    assert [asdict(sample) for sample in first] == [asdict(sample) for sample in second]
    assert first_contract == second_contract
    assert first_contract["available_rows"] == 20
    assert first_contract["selected_rows"] == 5
    assert len(first_contract["selection_sha256"]) == 64


def test_math_prompt_and_eos_truncation_are_explicit() -> None:
    sample = classic_math.MathSample("gsm8k", 0, "0", "What is 1 + 1?", "2")

    assert "step by step" in classic_math._format_prompt(sample, prompt_style="zero_shot_cot")
    assert "Give only the final answer" in classic_math._format_prompt(sample, prompt_style="zero_shot_direct")
    assert classic_math._truncate_at_token([4, 5, 2, 9], 2) == [4, 5]
    assert classic_math._truncate_at_token([4, 5], None) == [4, 5]


def test_classic_math_report_writes_strict_task_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("math_verify")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_yaml(
        {
            "stage": "stage0_baseline",
            "model_config_resolved": {"context_length": 512},
            "data_config_resolved": {"tokenizer": {"name": "unit"}},
        },
        run_dir / "config_resolved.yaml",
    )
    samples = {
        "gsm8k": [classic_math.MathSample("gsm8k", 0, "0", "What is 9 + 9?", "18")],
        "math500": [
            classic_math.MathSample("math500", 0, "m0", "Write one half.", r"\frac{1}{2}", "Algebra", "1")
        ],
    }
    tokenizer = _TinyTokenizer()
    monkeypatch.setattr(classic_math, "_load_tokenizer_from_run_config", lambda _config: tokenizer)
    monkeypatch.setattr(classic_math, "_load_model_for_run", lambda *_args: _TinyModel())
    monkeypatch.setattr(classic_math, "_device", lambda _name: "cpu")
    monkeypatch.setattr(classic_math, "_checkpoint_step", lambda *_args: 0)
    monkeypatch.setattr(
        classic_math,
        "load_math_samples",
        lambda task, **_kwargs: (samples[task], {"selected_rows": 1, "selection_sha256": task}),
    )
    predictions = [
        tokenizer.encode("Final answer is 18."),
        tokenizer.encode("Final answer is $0.5$"),
    ]
    monkeypatch.setattr(classic_math, "batched_greedy_generate", lambda *_args, **_kwargs: predictions)

    output = classic_math.make_classic_math_report(
        run_dir,
        output_path=run_dir / "report.json",
        tasks=["gsm8k", "math500"],
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["overall"]["accuracy"] == 1.0
    assert report["by_task"]["gsm8k"]["correct"] == 1
    assert report["by_task"]["math500"]["correct"] == 1
    assert report["protocol"]["leaderboard_comparable"] is False
    assert Path(report["samples_path"]).exists()
