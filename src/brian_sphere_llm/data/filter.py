from __future__ import annotations


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def keep_text(text: str, *, min_chars: int = 8) -> bool:
    return len(normalize_text(text)) >= min_chars
