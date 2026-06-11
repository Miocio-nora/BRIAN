from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.download import iter_hf_text_dataset
from brian_sphere_llm.data.filter import keep_text, normalize_text
from brian_sphere_llm.data.manifest import ManifestRow, read_manifest, sha256_text, sha256_tokens, write_manifest
from brian_sphere_llm.data.pack import pack_fixed_length, write_index, write_token_bin
from brian_sphere_llm.data.synthetic_routing import generate_synthetic_samples
from brian_sphere_llm.data.tokenize import load_tokenizer, tokenizer_metadata
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json


DEFAULT_MANIFEST_CREATED_AT = "1970-01-01T00:00:00+00:00"


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
    write_json(asdict(metadata), output_dir / "tokenizer_metadata.json")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    target_tokens = _int_config(config, "target_tokens", default=0, minimum=0)
    val_tokens_target = _int_config(config, "validation_tokens", default=0, minimum=0)
    sequence_length = _int_config(config, "sequence_length", minimum=2)
    documents_train: list[list[int]] = []
    documents_val: list[list[int]] = []
    manifest_rows: list[ManifestRow] = []
    manifest_created_at = str(config.get("manifest_created_at", DEFAULT_MANIFEST_CREATED_AT))
    synthetic_cfg = _mapping_config(config.get("synthetic_only", {}), "synthetic_only")
    synthetic_only = _bool_config(synthetic_cfg.get("enabled", False), "synthetic_only.enabled")
    if synthetic_only:
        samples = _synthetic_rows(config)
    else:
        samples = _mixture_rows(config)

    train_tokens = 0
    val_tokens = 0
    for index, sample in enumerate(samples):
        text = normalize_text(sample["text"])
        if not keep_text(text):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if not tokens:
            continue
        split = "val" if val_tokens < val_tokens_target and index % 10 == 0 else "train"
        source_path = output_dir / "source_text" / split / f"{index:012d}.txt"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(text, encoding="utf-8")
        row = ManifestRow.from_sample(
            sample_id=str(sample["sample_id"]),
            text=text,
            tokens=tokens,
            source_dataset=str(sample["source_dataset"]),
            source_url_or_id=str(sample["source_url_or_id"]),
            split=split,
            license=str(sample.get("license", metadata.license)),
            path=str(source_path),
            mixture_tag=str(sample["mixture_tag"]),
            route_metadata=sample.get("route_metadata"),
            created_at=manifest_created_at,
        )
        manifest_rows.append(row)
        if split == "val":
            documents_val.append(tokens)
            val_tokens += len(tokens)
        else:
            documents_train.append(tokens)
            train_tokens += len(tokens)
        if train_tokens >= target_tokens and val_tokens >= val_tokens_target:
            break

    train_sequences = pack_fixed_length(documents_train, sequence_length=sequence_length, pad_token_id=int(pad_token_id))
    val_sequences = pack_fixed_length(documents_val or documents_train[:1], sequence_length=sequence_length, pad_token_id=int(pad_token_id))
    write_token_bin(train_sequences, output_dir / "train.bin")
    write_token_bin(val_sequences, output_dir / "val.bin")
    write_index(output_dir / "train.idx", sequence_length=sequence_length, num_sequences=len(train_sequences))
    write_index(output_dir / "val.idx", sequence_length=sequence_length, num_sequences=len(val_sequences))
    output_manifest_path = output_dir / "manifest.jsonl"
    write_manifest(manifest_rows, output_manifest_path)
    manifest_path = Path(config.get("manifest_path", output_dir / "manifest.jsonl"))
    write_manifest(manifest_rows, manifest_path)
    manifest_text = output_manifest_path.read_text(encoding="utf-8")
    manifest_audit = _audit_prepared_manifest(output_manifest_path, tokenizer)
    source_mixture_realized = _realized_mixture(manifest_rows)
    source_mixture_realized_share = _normalize_mixture(source_mixture_realized)
    source_mixture_expected = _expected_mixture(config, source_mixture_realized)
    stats = {
        "recipe_name": config["recipe_name"],
        "num_documents": len(manifest_rows),
        "num_tokens_train": train_tokens,
        "num_tokens_val": val_tokens,
        "avg_tokens_per_doc": (train_tokens + val_tokens) / max(1, len(manifest_rows)),
        "sequence_length": sequence_length,
        "vocab_size": metadata.vocab_size,
        "source_mixture_expected": source_mixture_expected,
        "source_mixture_realized": source_mixture_realized,
        "source_mixture_realized_share": source_mixture_realized_share,
        "sha256_manifest": sha256_text(manifest_text),
        **manifest_audit,
        "tokenizer": asdict(metadata),
    }
    write_json(stats, output_dir / "stats.json")
    return output_dir


def _audit_prepared_manifest(manifest_path: Path, tokenizer: Any) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    source_text_failures = 0
    token_failures = 0
    for row in rows:
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
        if tag == "synthetic_routing":
            row_iter = _synthetic_mixture_rows(item, synthetic_count, seed)
        else:
            row_iter = _hf_mixture_rows(tag, item)
        sources.append({"order": order, "weight": weight, "emitted": 0, "rows": row_iter})
    while sources:
        source = min(sources, key=lambda entry: (entry["emitted"] / entry["weight"], entry["order"]))
        try:
            yield next(source["rows"])
        except StopIteration:
            sources.remove(source)
            continue
        source["emitted"] += 1


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
