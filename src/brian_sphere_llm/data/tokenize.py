from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class TokenizerMetadata:
    name: str
    revision: str
    vocab_size: int
    license: str
    special_tokens: dict[str, int | None]


class SimpleByteTokenizer:
    """Small offline tokenizer for tests and synthetic smoke generation."""

    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    unk_token_id = 259
    vocab_size = 260
    name_or_path = "simple-byte-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_special_tokens:
            return [self.bos_token_id, *tokens, self.eos_token_id]
        return tokens


def load_tokenizer(
    name: str,
    *,
    revision: str = "main",
    local_files_only: bool = False,
    fallback_to_byte: bool = False,
) -> TokenizerLike:
    if name == "simple-byte-tokenizer":
        return SimpleByteTokenizer()
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
    except ModuleNotFoundError:
        if fallback_to_byte:
            return SimpleByteTokenizer()
        raise


def tokenizer_metadata(
    tokenizer: TokenizerLike,
    *,
    name: str,
    revision: str,
    license: str,
) -> TokenizerMetadata:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    return TokenizerMetadata(
        name=name,
        revision=revision,
        vocab_size=vocab_size,
        license=license,
        special_tokens={
            "bos": getattr(tokenizer, "bos_token_id", None),
            "eos": getattr(tokenizer, "eos_token_id", None),
            "pad": getattr(tokenizer, "pad_token_id", None),
            "unk": getattr(tokenizer, "unk_token_id", None),
        },
    )
