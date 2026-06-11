import json
from pathlib import Path

import pytest

from brian_sphere_llm.data.manifest import ManifestRow, read_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    row = ManifestRow.from_sample(
        sample_id="sample-1",
        text="hello world",
        tokens=[1, 2, 3],
        source_dataset="unit",
        source_url_or_id="unit-1",
        split="train",
        license="test",
        path="memory",
        mixture_tag="synthetic",
        route_metadata={"pseudo_route_type": "skip", "pseudo_route_length": 2},
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest([row], path)
    rows = read_manifest(path)
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["token_count"] == 3
    assert rows[0]["route_metadata"]["pseudo_route_type"] == "skip"


def test_read_manifest_rejects_invalid_counts_and_split(tmp_path: Path) -> None:
    row = ManifestRow.from_sample(
        sample_id="sample-1",
        text="hello world",
        tokens=[1, 2, 3],
        source_dataset="unit",
        source_url_or_id="unit-1",
        split="train",
        license="test",
        path="memory",
        mixture_tag="synthetic",
    )
    invalid = row.__dict__ | {"token_count": -1, "split": "dev"}
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest counts must be non-negative"):
        read_manifest(path)

    invalid = row.__dict__ | {"split": "dev"}
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported split"):
        read_manifest(path)

    invalid = row.__dict__ | {"token_count": True}
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest counts must be integers"):
        read_manifest(path)

    invalid = row.__dict__ | {"byte_count": False}
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest counts must be integers"):
        read_manifest(path)

    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest row must be a JSON object"):
        read_manifest(path)


def test_write_manifest_rejects_nonfinite_metadata_before_writing_file(tmp_path: Path) -> None:
    row = ManifestRow(
        sample_id="sample-1",
        source_dataset="unit",
        source_url_or_id="unit-1",
        split="train",
        token_count=1,
        byte_count=1,
        sha256_text="text-hash",
        sha256_tokens="token-hash",
        license="test",
        path="memory",
        mixture_tag="synthetic",
        created_at="1970-01-01T00:00:00+00:00",
        route_metadata={"pseudo_route_length": float("nan")},
    )
    path = tmp_path / "manifest.jsonl"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_manifest([row], path)

    assert not path.exists()
