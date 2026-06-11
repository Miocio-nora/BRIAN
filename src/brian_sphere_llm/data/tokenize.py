from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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

    def save_pretrained(self, output_dir: str | Path) -> tuple[str, ...]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_json = {
            "format": "simple-byte-tokenizer-v1",
            "model": {
                "type": "byte-level",
                "byte_tokens": {f"<0x{value:02x}>": value for value in range(256)},
            },
            "special_tokens": {
                "<bos>": self.bos_token_id,
                "<eos>": self.eos_token_id,
                "<pad>": self.pad_token_id,
                "<unk>": self.unk_token_id,
            },
        }
        tokenizer_config = {
            "tokenizer_class": self.__class__.__name__,
            "name_or_path": self.name_or_path,
            "vocab_size": self.vocab_size,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }
        special_tokens_map = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        }
        paths = (
            output_dir / "tokenizer.json",
            output_dir / "tokenizer_config.json",
            output_dir / "special_tokens_map.json",
        )
        for path, payload in zip(paths, (tokenizer_json, tokenizer_config, special_tokens_map), strict=True):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        return tuple(str(path) for path in paths)


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
