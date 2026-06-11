from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        payload = {"created_at": utc_now_iso(), **row}
        line = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in rows]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
