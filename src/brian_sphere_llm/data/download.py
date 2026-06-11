from __future__ import annotations

from collections.abc import Iterator
from typing import Any


TEXT_FIELDS = ("text", "content", "completion", "document")


def iter_hf_text_dataset(
    *,
    dataset_name: str,
    split: str,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install `datasets` to prepare non-synthetic recipes.") from exc
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for index, row in enumerate(dataset):
        text = None
        for field in TEXT_FIELDS:
            if isinstance(row.get(field), str):
                text = row[field]
                break
        if text is None:
            continue
        yield {
            "sample_id": str(row.get("id", index)),
            "text": text,
            "source_url_or_id": str(row.get("url", row.get("id", index))),
        }
