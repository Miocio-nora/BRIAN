from __future__ import annotations

import json
from array import array
from pathlib import Path


def pack_fixed_length(
    documents: list[list[int]],
    *,
    sequence_length: int,
    pad_token_id: int,
    drop_remainder: bool = False,
) -> list[list[int]]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1")
    stream: list[int] = []
    for tokens in documents:
        stream.extend(tokens)
    sequences: list[list[int]] = []
    for start in range(0, len(stream), sequence_length):
        chunk = stream[start : start + sequence_length]
        if len(chunk) < sequence_length:
            if drop_remainder:
                break
            chunk = [*chunk, *([pad_token_id] * (sequence_length - len(chunk)))]
        sequences.append(chunk)
    return sequences


def write_token_bin(sequences: list[list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array("I")
    for sequence in sequences:
        values.extend(sequence)
    with path.open("wb") as handle:
        values.tofile(handle)


def read_token_bin(path: str | Path) -> list[int]:
    values = array("I")
    with Path(path).open("rb") as handle:
        values.fromfile(handle, Path(path).stat().st_size // values.itemsize)
    return list(values)


def write_index(path: str | Path, *, sequence_length: int, num_sequences: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"format": "uint32-flat-fixed", "sequence_length": sequence_length, "num_sequences": num_sequences},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
