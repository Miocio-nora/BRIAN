from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brian_sphere_llm.utils.logging import write_json


def make_out_by_difficulty_report(
    *,
    reasoning_report_path: str | Path | None = None,
    samples_path: str | Path | None = None,
    output_path: str | Path | None = None,
    difficulty_order: list[str] | None = None,
    min_step_delta: float = 0.0,
    min_output_probability_delta: float = 0.0,
) -> Path:
    if not reasoning_report_path and not samples_path:
        raise ValueError("out_by_difficulty report requires a reasoning report or samples JSONL path.")

    reasoning_report: dict[str, Any] = {}
    report_path: Path | None = None
    if reasoning_report_path:
        report_path = Path(reasoning_report_path)
        reasoning_report = _read_json(report_path)
        if samples_path is None:
            report_samples = reasoning_report.get("samples_path")
            if not report_samples:
                raise ValueError(f"Reasoning report does not include samples_path: {report_path}")
            samples_path = _resolve_path(report_samples, report_path.parent)

    sample_path = Path(samples_path) if samples_path is not None else None
    if sample_path is None:
        raise ValueError("samples_path resolution failed.")
    rows = _read_jsonl(sample_path)
    order = difficulty_order or ["easy", "medium", "hard"]
    by_difficulty = {difficulty: _difficulty_summary(rows, difficulty) for difficulty in order}
    deltas = _deltas(by_difficulty)
    checks = _checks(
        by_difficulty,
        deltas,
        reasoning_report=reasoning_report,
        reasoning_report_path=report_path,
        min_step_delta=min_step_delta,
        min_output_probability_delta=min_output_probability_delta,
    )
    report = {
        "overall_status": _status(checks),
        "sample_count": len(rows),
        "difficulty_order": order,
        "by_difficulty": by_difficulty,
        "deltas": deltas,
        "checks": checks,
        "thresholds": {
            "min_step_delta": min_step_delta,
            "min_output_probability_delta": min_output_probability_delta,
        },
        "inputs": {
            "reasoning_report": str(report_path) if report_path else None,
            "samples_path": str(sample_path),
            "reasoning_run_dir": reasoning_report.get("run_dir"),
            "reasoning_stage": reasoning_report.get("stage"),
            "reasoning_route_mode": reasoning_report.get("route_mode"),
            "reasoning_hard_exit": reasoning_report.get("hard_exit"),
            "reasoning_checkpoint": reasoning_report.get("checkpoint"),
        },
    }
    if output_path is None:
        output_path = Path("reports") / "out_by_difficulty_report.json"
    output_path = Path(output_path)
    write_json(report, output_path)
    return output_path


def _difficulty_summary(rows: list[dict[str, Any]], difficulty: str) -> dict[str, Any]:
    matching = [row for row in rows if row.get("difficulty") == difficulty]
    return {
        "sample_count": len(matching),
        "mean_route_steps": _mean([row.get("routing_average_route_steps") for row in matching]),
        "mean_active_block_evals_per_token": _mean(
            [row.get("routing_active_block_evals_per_token") for row in matching]
        ),
        "mean_p_output": _mean([row.get("routing_p_output_mean") for row in matching]),
    }


def _deltas(by_difficulty: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    easy = by_difficulty.get("easy", {})
    hard = by_difficulty.get("hard", {})
    return {
        "hard_minus_easy_route_steps": _delta(hard.get("mean_route_steps"), easy.get("mean_route_steps")),
        "hard_minus_easy_active_block_evals_per_token": _delta(
            hard.get("mean_active_block_evals_per_token"),
            easy.get("mean_active_block_evals_per_token"),
        ),
        "easy_minus_hard_p_output": _delta(easy.get("mean_p_output"), hard.get("mean_p_output")),
    }


def _checks(
    by_difficulty: dict[str, dict[str, Any]],
    deltas: dict[str, float | None],
    *,
    reasoning_report: dict[str, Any],
    reasoning_report_path: Path | None,
    min_step_delta: float,
    min_output_probability_delta: float,
) -> dict[str, bool | None]:
    easy = by_difficulty.get("easy", {})
    hard = by_difficulty.get("hard", {})
    easy_and_hard_present = int(easy.get("sample_count") or 0) > 0 and int(hard.get("sample_count") or 0) > 0
    route_delta = deltas["hard_minus_easy_route_steps"]
    active_delta = deltas["hard_minus_easy_active_block_evals_per_token"]
    output_delta = deltas["easy_minus_hard_p_output"]
    return {
        "reasoning_report_present": reasoning_report_path is not None,
        "reasoning_report_passed": _report_passed(reasoning_report),
        "stage4_output_action_reasoning": reasoning_report.get("stage") == "stage4_output_action",
        "hard_exit_reasoning": reasoning_report.get("hard_exit") is True,
        "easy_and_hard_present": easy_and_hard_present,
        "route_steps_non_decreasing_with_difficulty": _at_least(route_delta, min_step_delta),
        "active_compute_non_decreasing_with_difficulty": _at_least(active_delta, 0.0),
        "easy_output_probability_at_least_hard": _at_least(output_delta, min_output_probability_delta),
    }


def _status(checks: dict[str, bool | None]) -> str:
    values = list(checks.values())
    if any(value is False for value in values):
        return "fail"
    if any(value is None for value in values):
        return "warn"
    return "pass"


def _report_passed(report: dict[str, Any]) -> bool | None:
    if not report:
        return False
    checks = report.get("checks")
    checks_passed = None
    if isinstance(checks, dict):
        checks_passed = bool(checks) and all(value is True for value in checks.values())
    if report.get("overall_status") in {"fail", "warn"} or report.get("status") in {"fail", "warn"}:
        return False
    if report.get("overall_status") == "pass" or report.get("status") == "pass":
        return False if checks_passed is False else True
    return checks_passed


def _at_least(value: float | None, threshold: float) -> bool | None:
    if value is None:
        return None
    return value >= threshold


def _delta(value: Any, baseline: Any) -> float | None:
    value_num = _num(value)
    baseline_num = _num(baseline)
    if value_num is None or baseline_num is None:
        return None
    return value_num - baseline_num


def _mean(values: list[Any]) -> float | None:
    nums = [_num(value) for value in values]
    nums = [value for value in nums if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path
