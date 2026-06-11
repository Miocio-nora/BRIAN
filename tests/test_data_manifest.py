from pathlib import Path

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
