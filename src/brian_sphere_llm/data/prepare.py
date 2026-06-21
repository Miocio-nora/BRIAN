from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

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
    tokenization_cfg = _mapping_config(config.get("tokenization", {}), "tokenization")
    tokenization_parallelism = _bool_config(tokenization_cfg.get("parallelism", True), "tokenization.parallelism")
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenization_parallelism else "false"
    tokenization_worker_threads: int | None = None
    if "worker_threads" in tokenization_cfg:
        tokenization_worker_threads = _int_config(tokenization_cfg, "worker_threads", minimum=1)
        os.environ["RAYON_NUM_THREADS"] = str(tokenization_worker_threads)
    tokenization_batch_size = _int_config(tokenization_cfg, "batch_size", default=512, minimum=1)
    tokenization_prefetch_batches = _int_config(tokenization_cfg, "prefetch_batches", default=1, minimum=1)
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
    mixture_balance = _mixture_balance_mode(config)
    samples: Iterator[dict[str, Any]] | None = None
    if synthetic_only:
        samples = _synthetic_rows(config)
        encoded_batches = _encoded_sample_batches(
            samples,
            tokenizer,
            batch_size=tokenization_batch_size,
            prefetch_batches=tokenization_prefetch_batches,
        )
    elif mixture_balance == "token":
        encoded_batches = _token_balanced_encoded_sample_batches(
            config,
            tokenizer,
            batch_size=tokenization_batch_size,
        )
    else:
        samples = _mixture_rows(config)
        encoded_batches = _encoded_sample_batches(
            samples,
            tokenizer,
            batch_size=tokenization_batch_size,
            prefetch_batches=tokenization_prefetch_batches,
        )

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
            for pending, token_batch in encoded_batches:
                for (index, sample, text), tokens in zip(pending, token_batch, strict=True):
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
                if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
                    break
        finally:
            _close_iterator(encoded_batches)
            if samples is not None:
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
        "mixture_balance": mixture_balance,
        "sha256_manifest": manifest_hash.hexdigest(),
        "tokenization_batch_size": tokenization_batch_size,
        "tokenization_prefetch_batches": tokenization_prefetch_batches,
        "tokenization_parallelism": tokenization_parallelism,
        "tokenization_worker_threads": tokenization_worker_threads,
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


def _sample_batches(samples: Iterator[dict[str, Any]], *, batch_size: int) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    batch: list[tuple[int, dict[str, Any]]] = []
    for index, sample in enumerate(samples):
        batch.append((index, sample))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _normalized_sample_batch(
    sample_batch: list[tuple[int, dict[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any], str]], list[str]]:
    pending: list[tuple[int, dict[str, Any], str]] = []
    texts: list[str] = []
    for index, sample in sample_batch:
        text = normalize_text(sample["text"])
        if not keep_text(text):
            continue
        pending.append((index, sample, text))
        texts.append(text)
    return pending, texts


def _normalized_sample_batches(
    samples: Iterator[dict[str, Any]],
    *,
    batch_size: int,
) -> Iterator[tuple[list[tuple[int, dict[str, Any], str]], list[str]]]:
    for sample_batch in _sample_batches(samples, batch_size=batch_size):
        pending, texts = _normalized_sample_batch(sample_batch)
        if pending:
            yield pending, texts


def _encoded_sample_batches(
    samples: Iterator[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int,
    prefetch_batches: int,
) -> Iterator[tuple[list[tuple[int, dict[str, Any], str]], list[list[int]]]]:
    normalized_batches = _normalized_sample_batches(samples, batch_size=batch_size)
    if prefetch_batches <= 1:
        for pending, texts in normalized_batches:
            yield pending, _encode_text_batch(tokenizer, texts)
        return

    in_flight: deque[tuple[list[tuple[int, dict[str, Any], str]], Future[list[list[int]]]]] = deque()
    with ThreadPoolExecutor(max_workers=prefetch_batches, thread_name_prefix="tokenize-batch") as executor:
        for pending, texts in normalized_batches:
            in_flight.append((pending, executor.submit(_encode_text_batch, tokenizer, texts)))
            if len(in_flight) >= prefetch_batches:
                next_pending, future = in_flight.popleft()
                yield next_pending, future.result()
        while in_flight:
            next_pending, future = in_flight.popleft()
            yield next_pending, future.result()


def _encode_text_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    if not texts:
        return []
    try:
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    except TypeError:
        return [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    except AttributeError:
        return [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError, AttributeError):
        input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        return [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    return [list(tokens) for tokens in input_ids]


def _token_balanced_encoded_sample_batches(
    config: dict[str, Any],
    tokenizer: Any,
    *,
    batch_size: int,
) -> Iterator[tuple[list[tuple[int, dict[str, Any], str]], list[list[int]]]]:
    sources = _mixture_sources(config, token_balanced=True)
    sample_index = 0
    try:
        while sources:
            source = min(sources, key=lambda entry: (entry["emitted_tokens"] / entry["weight"], entry["order"]))
            raw_batch: list[tuple[int, dict[str, Any]]] = []
            while len(raw_batch) < batch_size:
                try:
                    sample = _next_source_sample(source)
                except StopIteration:
                    _close_iterator(source["rows"])
                    if source["repeat"]:
                        source["repeat_count"] += 1
                        source["rows"] = source["factory"]()
                        continue
                    sources.remove(source)
                    break
                raw_batch.append((sample_index, sample))
                sample_index += 1
            if not raw_batch:
                continue
            pending, texts = _normalized_sample_batch(raw_batch)
            if not pending:
                continue
            token_batch = _encode_text_batch(tokenizer, texts)
            emitted_tokens = sum(len(tokens) for tokens in token_batch if tokens)
            source["emitted_tokens"] += emitted_tokens
            source["emitted_rows"] += len(token_batch)
            if emitted_tokens <= 0:
                source["empty_batches"] += 1
            yield pending, token_batch
    finally:
        for source in sources:
            _close_iterator(source["rows"])


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
    sources = _mixture_sources(config, token_balanced=False)
    try:
        while sources:
            source = min(sources, key=lambda entry: (entry["emitted"] / entry["weight"], entry["order"]))
            try:
                yield _next_source_sample(source)
            except StopIteration:
                _close_iterator(source["rows"])
                if source["repeat"]:
                    source["repeat_count"] += 1
                    source["rows"] = source["factory"]()
                    continue
                sources.remove(source)
                continue
            source["emitted"] += 1
    finally:
        for source in sources:
            _close_iterator(source["rows"])


def _mixture_balance_mode(config: dict[str, Any]) -> str:
    value = str(config.get("mixture_balance", "document")).strip().lower().replace("_", "-")
    if value in {"document", "documents", "doc", "docs"}:
        return "document"
    if value in {"token", "tokens"}:
        return "token"
    raise ValueError("mixture_balance must be 'document' or 'token'.")


def _mixture_sources(config: dict[str, Any], *, token_balanced: bool) -> list[dict[str, Any]]:
    mixture_cfg = _mapping_config(config.get("mixture", {}), "mixture")
    target_tokens = _int_config(config, "target_tokens", default=0, minimum=0)
    val_tokens = _int_config(config, "validation_tokens", default=0, minimum=0)
    seed = _int_config(config, "seed", default=1, minimum=0)
    total_target_tokens = target_tokens + val_tokens
    sources: list[dict[str, Any]] = []
    for order, (tag, item) in enumerate(mixture_cfg.items()):
        item = _mapping_config(item, f"mixture.{tag}")
        weight = _float_config(item, "weight", minimum=0.0)
        if weight <= 0.0:
            continue
        repeat = _bool_config(item.get("repeat", False), f"mixture.{tag}.repeat")
        factory = _mixture_row_factory(
            str(tag),
            item,
            seed=seed,
            total_target_tokens=total_target_tokens,
            token_balanced=token_balanced,
        )
        sources.append(
            {
                "order": order,
                "weight": weight,
                "emitted": 0,
                "emitted_tokens": 0,
                "emitted_rows": 0,
                "empty_batches": 0,
                "repeat": repeat,
                "repeat_count": 0,
                "factory": factory,
                "rows": factory(),
            }
        )
    return sources


def _mixture_row_factory(
    tag: str,
    item: dict[str, Any],
    *,
    seed: int,
    total_target_tokens: int,
    token_balanced: bool,
) -> Callable[[], Iterator[dict[str, Any]]]:
    def factory() -> Iterator[dict[str, Any]]:
        if tag in {"synthetic_routing", "math_symbolic_qa", "code_structured"}:
            count = _synthetic_source_count(item, total_target_tokens, token_balanced=token_balanced)
            return _local_synthetic_mixture_rows(tag, item, count, seed)
        return _hf_mixture_rows(tag, item)

    return factory


def _synthetic_source_count(item: dict[str, Any], total_target_tokens: int, *, token_balanced: bool) -> int:
    if "sample_count" in item:
        return _int_config(item, "sample_count", minimum=1)
    divisor = 100 if token_balanced else 1000
    return max(1000, total_target_tokens // divisor)


def _next_source_sample(source: dict[str, Any]) -> dict[str, Any]:
    sample = next(source["rows"])
    repeat_count = int(source.get("repeat_count", 0))
    if repeat_count <= 0:
        return sample
    repeated = dict(sample)
    repeated["sample_id"] = f"{sample['sample_id']}-repeat{repeat_count}"
    repeated["source_url_or_id"] = f"{sample['source_url_or_id']}#repeat{repeat_count}"
    return repeated


def _local_synthetic_mixture_rows(tag: str, item: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    pack_examples = _int_config(item, "pack_examples_per_doc", default=1, minimum=1)
    if tag == "synthetic_routing":
        rows = _synthetic_mixture_rows(item, count, seed)
    elif tag == "math_symbolic_qa":
        rows = _math_symbolic_rows(item, count, seed)
    elif tag == "code_structured":
        rows = _code_structured_rows(item, count, seed)
    else:
        raise ValueError(f"Unsupported local synthetic mixture tag: {tag}")
    if pack_examples <= 1:
        yield from rows
        return
    yield from _packed_local_rows(tag, rows, pack_examples)


def _packed_local_rows(tag: str, rows: Iterator[dict[str, Any]], pack_examples: int) -> Iterator[dict[str, Any]]:
    packed_index = 0
    buffer: list[dict[str, Any]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= pack_examples:
            yield _packed_local_row(tag, buffer, packed_index)
            packed_index += 1
            buffer = []
    if buffer:
        yield _packed_local_row(tag, buffer, packed_index)


def _packed_local_row(tag: str, rows: list[dict[str, Any]], packed_index: int) -> dict[str, Any]:
    first = rows[0]
    return {
        "sample_id": f"{tag}-pack-{packed_index}",
        "text": "\n".join(str(row["text"]) for row in rows),
        "source_dataset": first.get("source_dataset", tag),
        "source_url_or_id": f"{tag}-pack-{packed_index}",
        "license": first.get("license", "internal-test"),
        "mixture_tag": tag,
        "route_metadata": {"task_family": tag, "packed_examples": len(rows)},
    }


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
