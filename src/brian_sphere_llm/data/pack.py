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


class FixedLengthTokenBinWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        sequence_length: int,
        pad_token_id: int,
        flush_sequences: int = 1024,
    ) -> None:
        if sequence_length <= 1:
            raise ValueError("sequence_length must be greater than 1")
        if flush_sequences <= 0:
            raise ValueError("flush_sequences must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.flush_values = sequence_length * flush_sequences
        self.num_sequences = 0
        self._tokens: list[int] = []
        self._values = array("I")
        self._handle = self.path.open("wb")
        self._closed = False

    def add_document(self, tokens: list[int]) -> None:
        if self._closed:
            raise ValueError("Cannot write to a closed token bin writer")
        if not tokens:
            return
        self._tokens.extend(tokens)
        while len(self._tokens) >= self.sequence_length:
            sequence = self._tokens[: self.sequence_length]
            del self._tokens[: self.sequence_length]
            self._append_sequence(sequence)

    def close(self, *, drop_remainder: bool = False) -> int:
        if self._closed:
            return self.num_sequences
        if self._tokens and not drop_remainder:
            sequence = [*self._tokens, *([self.pad_token_id] * (self.sequence_length - len(self._tokens)))]
            self._append_sequence(sequence)
        self._flush()
        self._handle.close()
        self._closed = True
        return self.num_sequences

    def _append_sequence(self, sequence: list[int]) -> None:
        self._values.extend(sequence)
        self.num_sequences += 1
        if len(self._values) >= self.flush_values:
            self._flush()

    def _flush(self) -> None:
        if not self._values:
            return
        self._values.tofile(self._handle)
        self._values = array("I")

    def __enter__(self) -> "FixedLengthTokenBinWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(drop_remainder=exc_type is not None)


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
