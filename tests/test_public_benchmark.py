import importlib.util
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")


class TinyChoiceModel:
    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)
        logits.scatter_(2, next_ids.unsqueeze(-1), 4.0)
        return {"logits": logits}


def test_exact_length_batched_choice_scores_match_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_benchmark = _load_public_benchmark_module()
    original_forward = public_benchmark._forward_routed_for_eval
    summarize_calls: list[bool] = []

    def record_forward(*args, **kwargs):
        summarize_calls.append(kwargs["summarize_routing"])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(public_benchmark, "_forward_routed_for_eval", record_forward)
    requests = [
        public_benchmark.ChoiceRequest(0, 0, (1, 2, 3, 4), 2),
        public_benchmark.ChoiceRequest(0, 1, (5, 6, 9, 10), 3),
        public_benchmark.ChoiceRequest(1, 0, (3, 4, 5), 1),
    ]
    model = TinyChoiceModel()
    expected = [
        public_benchmark._prepared_choice_score(
            model,
            request,
            config={"stage": "stage0_baseline"},
            route_mode="baseline",
            global_step=0,
            device=torch.device("cpu"),
            length_normalized=True,
        )
        for request in requests
    ]

    actual = public_benchmark._batched_choice_scores(
        model,
        requests,
        batch_size=2,
        config={"stage": "stage0_baseline"},
        route_mode="baseline",
        global_step=0,
        device=torch.device("cpu"),
        length_normalized=True,
    )

    assert actual == pytest.approx(expected, abs=1e-7, rel=1e-7)
    assert summarize_calls == [True, True, True, False, True]


def _load_public_benchmark_module():
    path = Path("scripts/public_benchmark.py")
    spec = importlib.util.spec_from_file_location("public_benchmark_for_tests", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
