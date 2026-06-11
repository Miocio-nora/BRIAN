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


def _transform_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 4, "medium": 8, "hard": 16}[difficulty]
    values = [rng.randint(0, 9) for _ in range(length)]
    transformed = [(value + 1) % 10 for value in values]
    return f"transform +1 mod10: {' '.join(map(str, values))} -> {' '.join(map(str, transformed))}"


def _arithmetic_task(rng: random.Random, difficulty: str) -> str:
    terms = {"easy": 2, "medium": 4, "hard": 8}[difficulty]
    values = [rng.randint(0, 20) for _ in range(terms)]
    return f"sum: {' + '.join(map(str, values))} = {sum(values)}"


def _rewrite_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 6, "medium": 10, "hard": 18}[difficulty]
    chars = [rng.choice(["a", "b", "c"]) for _ in range(length)]
    rewritten = [char.upper() if char in {"a", "c"} else char for char in chars]
    return f"rewrite a,c upper: {''.join(chars)} -> {''.join(rewritten)}"


def _parentheses_task(rng: random.Random, difficulty: str) -> str:
    pairs = {"easy": 2, "medium": 4, "hard": 8}[difficulty]
    parts: list[str] = []
    depth = 0
    max_depth = 0
    for _ in range(pairs):
        parts.append("(")
        depth += 1
        max_depth = max(max_depth, depth)
        if rng.random() < 0.45 and depth:
            parts.append(")")
            depth -= 1
    parts.extend(")" for _ in range(depth))
    text = "".join(parts)
    return f"parentheses stack: {text} -> balanced depth {max_depth}"


def _repeat_transform_task(rng: random.Random, difficulty: str) -> str:
    length = {"easy": 3, "medium": 5, "hard": 8}[difficulty]
    repeats = {"easy": 1, "medium": 2, "hard": 4}[difficulty]
    values = [rng.randint(0, 9) for _ in range(length)]
    transformed = [(value + repeats) % 10 for value in values]
    return f"repeat transform +1 x{repeats}: {' '.join(map(str, values))} -> {' '.join(map(str, transformed))}"


TASKS = {
    "copy": _copy_task,
    "reverse": _reverse_task,
    "transform": _transform_task,
    "arithmetic": _arithmetic_task,
    "rewrite": _rewrite_task,
    "parentheses": _parentheses_task,
    "repeat_transform": _repeat_transform_task,
}


ROUTE_TYPE_STATS = {
    "early_exit": {"pseudo_route_length": 1, "expected_recurrence_count": 0, "expected_skip_count": 1},
    "skip": {"pseudo_route_length": 2, "expected_recurrence_count": 0, "expected_skip_count": 1},
    "advance": {"pseudo_route_length": 4, "expected_recurrence_count": 0, "expected_skip_count": 0},
    "mixed": {"pseudo_route_length": 5, "expected_recurrence_count": 1, "expected_skip_count": 1},
    "recur": {"pseudo_route_length": 6, "expected_recurrence_count": 2, "expected_skip_count": 0},
    "late_exit": {"pseudo_route_length": 8, "expected_recurrence_count": 2, "expected_skip_count": 0},
}


def pseudo_route_metadata(difficulty: str, task_family: str = "copy") -> dict[str, str | int]:
    route_type = _route_type_for(difficulty, task_family)
    return {
        "pseudo_route_type": route_type,
        **ROUTE_TYPE_STATS[route_type],
        "difficulty_bin": difficulty,
    }


def _route_type_for(difficulty: str, task_family: str) -> str:
    if difficulty == "easy":
        if task_family in {"copy", "transform"}:
            return "early_exit"
        return "skip"
    if difficulty == "medium":
        if task_family in {"parentheses", "repeat_transform"}:
            return "mixed"
        return "advance"
    if difficulty == "hard":
        if task_family in {"arithmetic", "parentheses"}:
            return "late_exit"
        return "recur"
    raise ValueError(f"Unsupported difficulty: {difficulty}")


def generate_synthetic_samples(count: int, seed: int = 1) -> Iterator[SyntheticSample]:
    rng = random.Random(seed)
    difficulties = ["easy", "medium", "hard"]
    task_names = list(TASKS)
    for index in range(count):
        difficulty = difficulties[index % len(difficulties)]
        task_name = task_names[index % len(task_names)]
        text = TASKS[task_name](rng, difficulty)
        metadata = {"task_family": task_name, **pseudo_route_metadata(difficulty, task_name)}
        yield SyntheticSample(text=text, metadata=metadata)
