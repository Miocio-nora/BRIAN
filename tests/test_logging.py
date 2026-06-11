import json
from pathlib import Path

import pytest

from brian_sphere_llm.utils.logging import JsonlLogger, write_json, write_jsonl


def test_write_json_emits_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    write_json({"loss": 1.25, "status": "pass"}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"loss": 1.25, "status": "pass"}


def test_write_json_rejects_nonfinite_numbers_before_writing_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json({"loss": float("nan")}, path)

    assert not path.exists()


def test_jsonl_logger_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "train_log.jsonl"
    logger = JsonlLogger(path)

    logger.write({"loss": 1.0})
    with pytest.raises(ValueError, match="Out of range float values"):
        logger.write({"loss": float("inf")})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["loss"] == 1.0


def test_write_jsonl_rejects_nonfinite_rows_before_writing_file(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_jsonl([{"loss": 1.0}, {"loss": float("-inf")}], path)

    assert not path.exists()


def test_write_jsonl_emits_strict_rows(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"

    write_jsonl([{"sample_id": "a", "score": 1.0}, {"sample_id": "b", "score": 0.0}], path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"sample_id": "a", "score": 1.0}, {"sample_id": "b", "score": 0.0}]
