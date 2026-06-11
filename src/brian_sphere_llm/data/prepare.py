from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from brian_sphere_llm.data.download import iter_hf_text_dataset
from brian_sphere_llm.data.filter import keep_text, normalize_text
from brian_sphere_llm.data.manifest import ManifestRow, sha256_text, write_manifest
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
        local_files_only=bool(tokenizer_cfg.get("local_files_only", False)),
        fallback_to_byte=bool(tokenizer_cfg.get("fallback_to_byte", False)),
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

    target_tokens = int(config.get("target_tokens", 0))
    val_tokens_target = int(config.get("validation_tokens", 0))
    sequence_length = int(config["sequence_length"])
    documents_train: list[list[int]] = []
    documents_val: list[list[int]] = []
    manifest_rows: list[ManifestRow] = []
    manifest_created_at = str(config.get("manifest_created_at", DEFAULT_MANIFEST_CREATED_AT))
    synthetic_only = bool(config.get("synthetic_only", {}).get("enabled", False))
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
    stats = {
        "recipe_name": config["recipe_name"],
        "num_documents": len(manifest_rows),
        "num_tokens_train": train_tokens,
        "num_tokens_val": val_tokens,
        "avg_tokens_per_doc": (train_tokens + val_tokens) / max(1, len(manifest_rows)),
        "sequence_length": sequence_length,
        "vocab_size": metadata.vocab_size,
        "source_mixture_realized": _realized_mixture(manifest_rows),
        "sha256_manifest": sha256_text(manifest_text),
        "tokenizer": asdict(metadata),
    }
    write_json(stats, output_dir / "stats.json")
    return output_dir


def _synthetic_rows(config: dict[str, Any]):
    count = int(config.get("synthetic_only", {}).get("sample_count", 1000))
    seed = int(config.get("seed", 1))
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
    synthetic_cfg = config.get("mixture", {}).get("synthetic_routing", {})
    synthetic_count = max(1000, int(config.get("target_tokens", 0) // 1000))
    for index, sample in enumerate(generate_synthetic_samples(synthetic_count, int(config.get("seed", 1)))):
        yield {
            "sample_id": f"synthetic-{index}",
            "text": sample.text,
            "source_dataset": synthetic_cfg.get("source_dataset", "brian_synthetic_routing"),
            "source_url_or_id": f"synthetic-{index}",
            "license": "internal-test",
            "mixture_tag": "synthetic_routing",
            "route_metadata": sample.metadata,
        }
    for tag, item in config.get("mixture", {}).items():
        if tag == "synthetic_routing":
            continue
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


def _realized_mixture(rows: list[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mixture_tag] = counts.get(row.mixture_tag, 0) + row.token_count
    return counts
