from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SyntheticSample:
    text: str
    metadata: dict[str, str | int]


def _copy_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 4, "medium": 8, "hard": 16}[difficulty]
    values = [str(rng.randint(0, 9)) for _ in range(length)]
    return f"copy: {' '.join(values)} -> {' '.join(values)}"


def _reverse_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 4, "medium": 8, "hard": 16}[difficulty]
    values = [str(rng.randint(0, 9)) for _ in range(length)]
    return f"reverse: {' '.join(values)} -> {' '.join(reversed(values))}"


def _arithmetic_task(rng: random.Random, difficulty: str) -> str:
    terms = {"easy": 2, "medium": 4, "hard": 8}[difficulty]
    values = [rng.randint(0, 20) for _ in range(terms)]
    return f"sum: {' + '.join(map(str, values))} = {sum(values)}"


def _rewrite_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 6, "medium": 10, "hard": 18}[difficulty]
    chars = [rng.choice(["a", "b", "c"]) for _ in range(length)]
    rewritten = [char.upper() if char in {"a", "c"} else char for char in chars]
    return f"rewrite a,c upper: {''.join(chars)} -> {''.join(rewritten)}"


TASKS = {
    "copy": _copy_task,
    "reverse": _reverse_task,
    "arithmetic": _arithmetic_task,
    "rewrite": _rewrite_task,
}


def pseudo_route_metadata(difficulty: str) -> dict[str, str | int]:
    if difficulty == "easy":
        return {
            "pseudo_route_type": "skip",
            "pseudo_route_length": 2,
            "expected_recurrence_count": 0,
            "expected_skip_count": 1,
            "difficulty_bin": "easy",
        }
    if difficulty == "hard":
        return {
            "pseudo_route_type": "recur",
            "pseudo_route_length": 6,
            "expected_recurrence_count": 2,
            "expected_skip_count": 0,
            "difficulty_bin": "hard",
        }
    return {
        "pseudo_route_type": "advance",
        "pseudo_route_length": 4,
        "expected_recurrence_count": 0,
        "expected_skip_count": 0,
        "difficulty_bin": "medium",
    }


def generate_synthetic_samples(count: int, seed: int = 1) -> Iterator[SyntheticSample]:
    rng = random.Random(seed)
    difficulties = ["easy", "medium", "hard"]
    task_names = list(TASKS)
    for index in range(count):
        difficulty = difficulties[index % len(difficulties)]
        task_name = task_names[index % len(task_names)]
        text = TASKS[task_name](rng, difficulty)
        metadata = {"task_family": task_name, **pseudo_route_metadata(difficulty)}
        yield SyntheticSample(text=text, metadata=metadata)
