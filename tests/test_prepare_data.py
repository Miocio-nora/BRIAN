import json
from pathlib import Path

import pytest

from brian_sphere_llm.data.manifest import REQUIRED_MANIFEST_FIELDS, sha256_text
from brian_sphere_llm.data.prepare import DEFAULT_MANIFEST_CREATED_AT, _bool_config, prepare_data
from brian_sphere_llm.utils.config import load_yaml, save_yaml


def test_prepare_tiny_synthetic_data(tmp_path: Path) -> None:
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = 1000
    cfg["validation_tokens"] = 100
    cfg["synthetic_only"]["sample_count"] = 32
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)
    output_dir = prepare_data(config_path)
    manifest_text = (output_dir / "manifest.jsonl").read_text(encoding="utf-8")
    assert (output_dir / "train.bin").exists()
    assert (output_dir / "train.idx").exists()
    assert (output_dir / "val.bin").exists()
    assert (output_dir / "val.idx").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert Path(cfg["manifest_path"]).exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "tokenizer_config.json").exists()
    assert (output_dir / "tokenizer_metadata.json").exists()
    assert (output_dir / "stats.json").exists()
    tokenizer_config = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert tokenizer_config["tokenizer_class"] == "SimpleByteTokenizer"
    assert REQUIRED_MANIFEST_FIELDS <= set(manifest_rows[0])
    for key in [
        "num_documents",
        "num_tokens_train",
        "num_tokens_val",
        "avg_tokens_per_doc",
        "sequence_length",
        "vocab_size",
        "source_mixture_realized",
        "sha256_manifest",
    ]:
        assert key in stats
    assert stats["sha256_manifest"]
    assert stats["sha256_manifest"] == sha256_text(manifest_text)
    assert stats["source_mixture_realized"]
    assert {row["created_at"] for row in manifest_rows} == {DEFAULT_MANIFEST_CREATED_AT}
    assert manifest_rows[0]["route_metadata"]["pseudo_route_type"] in {
        "advance",
        "early_exit",
        "late_exit",
        "mixed",
        "recur",
        "skip",
    }
    assert "difficulty_bin" in manifest_rows[0]["route_metadata"]

    prepare_data(config_path)
    rerun_stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert (output_dir / "manifest.jsonl").read_text(encoding="utf-8") == manifest_text
    assert rerun_stats["sha256_manifest"] == stats["sha256_manifest"]


def test_prepare_data_rejects_boolean_numeric_config(tmp_path: Path) -> None:
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = True
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)

    with pytest.raises(ValueError, match="target_tokens"):
        prepare_data(config_path)


def test_prepare_data_rejects_boolean_synthetic_sample_count(tmp_path: Path) -> None:
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = 1000
    cfg["validation_tokens"] = 100
    cfg["synthetic_only"]["sample_count"] = False
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)

    with pytest.raises(ValueError, match="sample_count"):
        prepare_data(config_path)


def test_prepare_data_rejects_non_mapping_mixture_config(tmp_path: Path) -> None:
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = 1000
    cfg["validation_tokens"] = 100
    cfg["synthetic_only"]["enabled"] = False
    cfg["mixture"] = True
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)

    with pytest.raises(ValueError, match="mixture"):
        prepare_data(config_path)


def test_prepare_bool_config_parses_false_string_without_truthiness() -> None:
    assert _bool_config("false", "field") is False
    assert _bool_config("true", "field") is True
    with pytest.raises(ValueError, match="field"):
        _bool_config("maybe", "field")
