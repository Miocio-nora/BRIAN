from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from brian_sphere_llm.utils.logging import utc_now_iso, write_jsonl


REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "source_dataset",
    "source_url_or_id",
    "split",
    "token_count",
    "byte_count",
    "sha256_text",
    "sha256_tokens",
    "license",
    "path",
    "mixture_tag",
    "created_at",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_tokens(tokens: Iterable[int]) -> str:
    packed = ",".join(str(token) for token in tokens).encode("utf-8")
    return sha256_bytes(packed)


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    source_dataset: str
    source_url_or_id: str
    split: str
    token_count: int
    byte_count: int
    sha256_text: str
    sha256_tokens: str
    license: str
    path: str
    mixture_tag: str
    created_at: str
    route_metadata: dict[str, str | int] | None = None

    @classmethod
    def from_sample(
        cls,
        *,
        sample_id: str,
        text: str,
        tokens: list[int],
        source_dataset: str,
        source_url_or_id: str,
        split: str,
        license: str,
        path: str,
        mixture_tag: str,
        route_metadata: dict[str, str | int] | None = None,
        created_at: str | None = None,
    ) -> "ManifestRow":
        return cls(
            sample_id=sample_id,
            source_dataset=source_dataset,
            source_url_or_id=source_url_or_id,
            split=split,
            token_count=len(tokens),
            byte_count=len(text.encode("utf-8")),
            sha256_text=sha256_text(text),
            sha256_tokens=sha256_tokens(tokens),
            license=license,
            path=path,
            mixture_tag=mixture_tag,
            created_at=created_at or utc_now_iso(),
            route_metadata=route_metadata,
        )

    def validate(self) -> None:
        _validate_manifest_row(asdict(self))


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    payloads = []
    for row in rows:
        row.validate()
        payloads.append(asdict(row))
    write_jsonl(payloads, path)


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                _validate_manifest_row(row)
                rows.append(row)
    return rows


def _validate_manifest_row(row: dict) -> None:
    if not isinstance(row, dict):
        raise ValueError("Manifest row must be a JSON object")
    missing = REQUIRED_MANIFEST_FIELDS - set(row)
    if missing:
        raise ValueError(f"Manifest row missing fields: {sorted(missing)}")
    if type(row.get("token_count")) is not int or type(row.get("byte_count")) is not int:
        raise ValueError("Manifest counts must be integers")
    if row["token_count"] < 0 or row["byte_count"] < 0:
        raise ValueError("Manifest counts must be non-negative")
    if row["split"] not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {row['split']}")
