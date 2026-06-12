from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.download import iter_hf_text_dataset
from brian_sphere_llm.data.filter import keep_text, normalize_text
from brian_sphere_llm.data.manifest import ManifestRow, read_manifest, sha256_bytes, sha256_text, sha256_tokens
from brian_sphere_llm.data.pack import FixedLengthTokenBinWriter, write_index
from brian_sphere_llm.data.synthetic_routing import generate_synthetic_samples
from brian_sphere_llm.data.tokenize import load_tokenizer, tokenizer_metadata
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json


DEFAULT_MANIFEST_CREATED_AT = "1970-01-01T00:00:00+00:00"
UNSAVED_SOURCE_PATH_PREFIX = "unsaved://"


def prepare_data(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_cfg = config["tokenizer"]
    tokenizer = load_tokenizer(
        tokenizer_cfg["name"],
        revision=str(tokenizer_cfg.get("revision", "main")),
        local_files_only=_bool_config(tokenizer_cfg.get("local_files_only", False), "tokenizer.local_files_only"),
        fallback_to_byte=_bool_config(tokenizer_cfg.get("fallback_to_byte", False), "tokenizer.fallback_to_byte"),
    )
    metadata = tokenizer_metadata(
        tokenizer,
        name=tokenizer_cfg["name"],
        revision=str(tokenizer_cfg.get("revision", "main")),
        license=str(tokenizer_cfg.get("license", "unknown")),
    )
    tokenizer_metadata_path = output_dir / "tokenizer_metadata.json"
    write_json(asdict(metadata), tokenizer_metadata_path)
    saved_tokenizer_paths: tuple[str, ...] = ()
    if hasattr(tokenizer, "save_pretrained"):
        saved_paths = tokenizer.save_pretrained(output_dir)
        saved_tokenizer_paths = tuple(str(path) for path in saved_paths) if saved_paths else ()
    tokenizer_artifact_audit = _audit_tokenizer_artifacts(output_dir, saved_tokenizer_paths)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    target_tokens = _int_config(config, "target_tokens", default=0, minimum=0)
    val_tokens_target = _int_config(config, "validation_tokens", default=0, minimum=0)
    sequence_length = _int_config(config, "sequence_length", minimum=2)
    manifest_created_at = str(config.get("manifest_created_at", DEFAULT_MANIFEST_CREATED_AT))
    source_text_enabled = _source_text_enabled(config)
    synthetic_cfg = _mapping_config(config.get("synthetic_only", {}), "synthetic_only")
    synthetic_only = _bool_config(synthetic_cfg.get("enabled", False), "synthetic_only.enabled")
    if synthetic_only:
        samples = _synthetic_rows(config)
    else:
        samples = _mixture_rows(config)

    train_tokens = 0
    val_tokens = 0
    output_manifest_path = output_dir / "manifest.jsonl"
    manifest_path = Path(config.get("manifest_path", output_dir / "manifest.jsonl"))
    manifest_hash = hashlib.sha256()
    manifest_row_count = 0
    source_mixture_realized: dict[str, int] = {}
    first_train_tokens: list[int] | None = None

    with ExitStack() as stack:
        train_writer = stack.enter_context(
            FixedLengthTokenBinWriter(
                output_dir / "train.bin",
                sequence_length=sequence_length,
                pad_token_id=int(pad_token_id),
            )
        )
        val_writer = stack.enter_context(
            FixedLengthTokenBinWriter(
                output_dir / "val.bin",
                sequence_length=sequence_length,
                pad_token_id=int(pad_token_id),
            )
        )
        manifest_handles = stack.enter_context(_open_manifest_writers(output_manifest_path, manifest_path))
        try:
            for index, sample in enumerate(samples):
                text = normalize_text(sample["text"])
                if not keep_text(text):
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=True)
                if not tokens:
                    continue
                split = "val" if val_tokens < val_tokens_target and index % 10 == 0 else "train"
                source_path = _write_source_text(output_dir, split, index, text) if source_text_enabled else None
                row = ManifestRow.from_sample(
                    sample_id=str(sample["sample_id"]),
                    text=text,
                    tokens=tokens,
                    source_dataset=str(sample["source_dataset"]),
                    source_url_or_id=str(sample["source_url_or_id"]),
                    split=split,
                    license=str(sample.get("license", metadata.license)),
                    path=str(source_path) if source_path is not None else _unsaved_source_path(str(config["recipe_name"]), split, index),
                    mixture_tag=str(sample["mixture_tag"]),
                    route_metadata=sample.get("route_metadata"),
                    created_at=manifest_created_at,
                )
                _write_manifest_row(row, manifest_handles, manifest_hash)
                manifest_row_count += 1
                source_mixture_realized[row.mixture_tag] = source_mixture_realized.get(row.mixture_tag, 0) + row.token_count
                if split == "val":
                    val_writer.add_document(tokens)
                    val_tokens += len(tokens)
                else:
                    train_writer.add_document(tokens)
                    train_tokens += len(tokens)
                    if first_train_tokens is None:
                        first_train_tokens = list(tokens)
                if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
                    break
        finally:
            _close_iterator(samples)
        if val_tokens == 0 and first_train_tokens is not None:
            val_writer.add_document(first_train_tokens)
        train_sequence_count = train_writer.close()
        val_sequence_count = val_writer.close()

    write_index(output_dir / "train.idx", sequence_length=sequence_length, num_sequences=train_sequence_count)
    write_index(output_dir / "val.idx", sequence_length=sequence_length, num_sequences=val_sequence_count)
    manifest_audit = _generated_manifest_audit(manifest_row_count, source_text_enabled=source_text_enabled)
    source_mixture_realized_share = _normalize_mixture(source_mixture_realized)
    source_mixture_expected = _expected_mixture(config, source_mixture_realized)
    stats = {
        "recipe_name": config["recipe_name"],
        "num_documents": manifest_row_count,
        "num_tokens_train": train_tokens,
        "num_tokens_val": val_tokens,
        "avg_tokens_per_doc": (train_tokens + val_tokens) / max(1, manifest_row_count),
        "sequence_length": sequence_length,
        "vocab_size": metadata.vocab_size,
        "source_mixture_expected": source_mixture_expected,
        "source_mixture_realized": source_mixture_realized,
        "source_mixture_realized_share": source_mixture_realized_share,
        "sha256_manifest": manifest_hash.hexdigest(),
        **manifest_audit,
        **tokenizer_artifact_audit,
        "tokenizer": asdict(metadata),
    }
    write_json(stats, output_dir / "stats.json")
    return output_dir


@contextmanager
def _open_manifest_writers(*paths: Path):
    handles = []
    seen: set[Path] = set()
    try:
        for path in paths:
            normalized = path.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(path.open("w", encoding="utf-8"))
        yield tuple(handles)
    finally:
        for handle in handles:
            handle.close()


def _write_manifest_row(row: ManifestRow, handles: tuple[Any, ...], manifest_hash: "hashlib._Hash") -> None:
    row.validate()
    line = json.dumps(asdict(row), sort_keys=True, allow_nan=False)
    payload = f"{line}\n"
    for handle in handles:
        handle.write(payload)
    manifest_hash.update(payload.encode("utf-8"))


def _source_text_enabled(config: dict[str, Any]) -> bool:
    value = config.get("source_text", {})
    if isinstance(value, bool | str):
        return _bool_config(value, "source_text")
    source_text_cfg = _mapping_config(value, "source_text")
    return _bool_config(source_text_cfg.get("enabled", False), "source_text.enabled")


def _write_source_text(output_dir: Path, split: str, index: int, text: str) -> Path:
    source_path = output_dir / "source_text" / split / f"{index:012d}.txt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(text, encoding="utf-8")
    return source_path


def _unsaved_source_path(recipe_name: str, split: str, index: int) -> str:
    return f"{UNSAVED_SOURCE_PATH_PREFIX}{recipe_name}/{split}/{index:012d}"


def _generated_manifest_audit(row_count: int, *, source_text_enabled: bool) -> dict[str, Any]:
    return {
        "manifest_row_count": row_count,
        "manifest_source_text_hashes_verified": row_count > 0,
        "manifest_token_hashes_verified": row_count > 0,
        "manifest_source_text_hash_failure_count": 0,
        "manifest_token_hash_failure_count": 0,
        "manifest_source_text_storage": "files" if source_text_enabled else "none",
    }


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _audit_prepared_manifest(manifest_path: Path, tokenizer: Any) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    source_text_failures = 0
    token_failures = 0
    skipped_unsaved = 0
    for row in rows:
        if str(row["path"]).startswith(UNSAVED_SOURCE_PATH_PREFIX):
            skipped_unsaved += 1
            continue
        source_path = Path(str(row["path"]))
        if not source_path.exists():
            source_text_failures += 1
            token_failures += 1
            continue
        text = source_path.read_text(encoding="utf-8")
        if len(text.encode("utf-8")) != row["byte_count"] or sha256_text(text) != row["sha256_text"]:
            source_text_failures += 1
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) != row["token_count"] or sha256_tokens(tokens) != row["sha256_tokens"]:
            token_failures += 1
    return {
        "manifest_row_count": len(rows),
        "manifest_source_text_hashes_verified": bool(rows) and source_text_failures == 0,
        "manifest_token_hashes_verified": bool(rows) and token_failures == 0,
        "manifest_source_text_hash_failure_count": source_text_failures,
        "manifest_token_hash_failure_count": token_failures,
        "manifest_source_text_hash_skipped_count": skipped_unsaved,
    }


def _audit_tokenizer_artifacts(output_dir: Path, saved_paths: tuple[str, ...]) -> dict[str, Any]:
    paths = {Path(path) for path in saved_paths}
    paths.add(output_dir / "tokenizer_metadata.json")
    existing = sorted((path for path in paths if path.exists() and path.is_file()), key=lambda path: path.name)
    artifact_hashes = {
        path.name: sha256_bytes(path.read_bytes())
        for path in existing
    }
    return {
        "tokenizer_artifact_count": len(existing),
        "tokenizer_artifacts_present": bool(existing),
        "tokenizer_artifact_hashes": artifact_hashes,
        "tokenizer_artifact_hashes_present": bool(artifact_hashes),
    }


def _synthetic_rows(config: dict[str, Any]):
    synthetic_cfg = _mapping_config(config.get("synthetic_only", {}), "synthetic_only")
    count = _int_config(synthetic_cfg, "sample_count", default=1000, minimum=1)
    seed = _int_config(config, "seed", default=1, minimum=0)
    for index, sample in enumerate(generate_synthetic_samples(count, seed)):
        yield {
            "sample_id": f"synthetic-{index}",
            "text": sample.text,
            "source_dataset": "brian_synthetic_routing",
            "source_url_or_id": f"synthetic-{index}",
            "license": "internal-test",
            "mixture_tag": str(sample.metadata["task_family"]),
            "route_metadata": sample.metadata,
        }


def _mixture_rows(config: dict[str, Any]):
    mixture_cfg = _mapping_config(config.get("mixture", {}), "mixture")
    target_tokens = _int_config(config, "target_tokens", default=0, minimum=0)
    seed = _int_config(config, "seed", default=1, minimum=0)
    synthetic_count = max(1000, target_tokens // 1000)
    sources: list[dict[str, Any]] = []
    for order, (tag, item) in enumerate(mixture_cfg.items()):
        item = _mapping_config(item, f"mixture.{tag}")
        weight = _float_config(item, "weight", minimum=0.0)
        if weight <= 0.0:
            continue
        if tag in {"synthetic_routing", "math_symbolic_qa", "code_structured"}:
            row_iter = _local_synthetic_mixture_rows(tag, item, synthetic_count, seed)
        else:
            row_iter = _hf_mixture_rows(tag, item)
        sources.append({"order": order, "weight": weight, "emitted": 0, "rows": row_iter})
    try:
        while sources:
            source = min(sources, key=lambda entry: (entry["emitted"] / entry["weight"], entry["order"]))
            try:
                yield next(source["rows"])
            except StopIteration:
                _close_iterator(source["rows"])
                sources.remove(source)
                continue
            source["emitted"] += 1
    finally:
        for source in sources:
            _close_iterator(source["rows"])


def _local_synthetic_mixture_rows(tag: str, item: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    if tag == "synthetic_routing":
        yield from _synthetic_mixture_rows(item, count, seed)
        return
    if tag == "math_symbolic_qa":
        yield from _math_symbolic_rows(item, count, seed)
        return
    if tag == "code_structured":
        yield from _code_structured_rows(item, count, seed)
        return
    raise ValueError(f"Unsupported local synthetic mixture tag: {tag}")


def _synthetic_mixture_rows(item: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    for index, sample in enumerate(generate_synthetic_samples(count, seed)):
        yield {
            "sample_id": f"synthetic-{index}",
            "text": sample.text,
            "source_dataset": item.get("source_dataset", "brian_synthetic_routing"),
            "source_url_or_id": f"synthetic-{index}",
            "license": "internal-test",
            "mixture_tag": "synthetic_routing",
            "route_metadata": sample.metadata,
        }


def _math_symbolic_rows(item: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    for index in range(count):
        left = (seed * 17 + index * 3) % 97
        right = (seed * 31 + index * 5) % 89
        operation = ["+", "-", "*"][index % 3]
        if operation == "+":
            answer = left + right
        elif operation == "-":
            answer = left - right
        else:
            answer = left * right
        text = f"symbolic math: {left} {operation} {right} = {answer}"
        yield {
            "sample_id": f"math-symbolic-{index}",
            "text": text,
            "source_dataset": item.get("source_dataset", "synthetic_math_symbolic"),
            "source_url_or_id": f"math-symbolic-{index}",
            "license": "internal-test",
            "mixture_tag": "math_symbolic_qa",
            "route_metadata": {
                "task_family": "math_symbolic_qa",
                "operator": operation,
                "difficulty_bin": "easy" if index % 3 == 0 else "medium" if index % 3 == 1 else "hard",
            },
        }


def _code_structured_rows(item: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    for index in range(count):
        variable = f"x{(seed + index) % 13}"
        increment = (seed + index * 7) % 11 + 1
        repeat = index % 4 + 1
        final_value = increment * repeat
        text = (
            f"code trace: {variable}=0; "
            f"for i in range({repeat}): {variable}={variable}+{increment}; "
            f"return {variable} -> {final_value}"
        )
        yield {
            "sample_id": f"code-structured-{index}",
            "text": text,
            "source_dataset": item.get("source_dataset", "synthetic_code_structured"),
            "source_url_or_id": f"code-structured-{index}",
            "license": "internal-test",
            "mixture_tag": "code_structured",
            "route_metadata": {
                "task_family": "code_structured",
                "loop_count": repeat,
                "difficulty_bin": "easy" if repeat <= 1 else "medium" if repeat <= 3 else "hard",
            },
        }


def _hf_mixture_rows(tag: str, item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    source_dataset = str(item["source_dataset"])
    split = str(item.get("split", "train"))
    for row in iter_hf_text_dataset(dataset_name=source_dataset, split=split, streaming=True):
        yield {
            "sample_id": f"{tag}-{row['sample_id']}",
            "text": row["text"],
            "source_dataset": source_dataset,
            "source_url_or_id": row["source_url_or_id"],
            "license": str(item.get("license", "unknown")),
            "mixture_tag": tag,
        }


def _float_config(
    config: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite numeric value.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return number


def _realized_mixture(rows: list[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mixture_tag] = counts.get(row.mixture_tag, 0) + row.token_count
    return counts


def _expected_mixture(config: dict[str, Any], realized: dict[str, int]) -> dict[str, float]:
    synthetic_cfg = _mapping_config(config.get("synthetic_only", {}), "synthetic_only")
    synthetic_only = _bool_config(synthetic_cfg.get("enabled", False), "synthetic_only.enabled")
    if synthetic_only:
        return _normalize_mixture(realized)

    mixture_cfg = _mapping_config(config.get("mixture", {}), "mixture")
    weights: dict[str, float] = {}
    for tag, item in mixture_cfg.items():
        item = _mapping_config(item, f"mixture.{tag}")
        weight = _float_config(item, "weight", minimum=0.0)
        if weight > 0.0:
            weights[str(tag)] = weight
    return _normalize_mixture(weights)


def _normalize_mixture(values: dict[str, int | float]) -> dict[str, float]:
    positive_values = {
        str(tag): float(value)
        for tag, value in values.items()
        if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    }
    total = sum(positive_values.values())
    if total <= 0.0:
        return {}
    return {tag: value / total for tag, value in positive_values.items()}


def _mapping_config(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _bool_config(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _int_config(
    config: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key in config:
        value = config[key]
    elif default is not None:
        value = default
    else:
        raise KeyError(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not a boolean.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    else:
        raise ValueError(f"{key} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return number
