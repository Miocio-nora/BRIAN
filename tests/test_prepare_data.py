import json
from itertools import islice
import math
from pathlib import Path

import pytest

from brian_sphere_llm.data.manifest import REQUIRED_MANIFEST_FIELDS, sha256_text
from brian_sphere_llm.data.prepare import DEFAULT_MANIFEST_CREATED_AT, _audit_prepared_manifest, _bool_config, _mixture_rows, prepare_data
from brian_sphere_llm.data.tokenize import load_tokenizer
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
        "source_mixture_expected",
        "source_mixture_realized",
        "source_mixture_realized_share",
        "sha256_manifest",
        "manifest_row_count",
        "manifest_source_text_hashes_verified",
        "manifest_token_hashes_verified",
        "manifest_source_text_hash_failure_count",
        "manifest_token_hash_failure_count",
        "tokenizer_artifact_count",
        "tokenizer_artifacts_present",
        "tokenizer_artifact_hashes",
        "tokenizer_artifact_hashes_present",
    ]:
        assert key in stats
    assert stats["sha256_manifest"]
    assert stats["sha256_manifest"] == sha256_text(manifest_text)
    assert stats["manifest_row_count"] == len(manifest_rows)
    assert stats["manifest_source_text_hashes_verified"] is True
    assert stats["manifest_token_hashes_verified"] is True
    assert stats["manifest_source_text_hash_failure_count"] == 0
    assert stats["manifest_token_hash_failure_count"] == 0
    assert stats["tokenizer_artifact_count"] >= 4
    assert stats["tokenizer_artifacts_present"] is True
    assert stats["tokenizer_artifact_hashes_present"] is True
    assert set(stats["tokenizer_artifact_hashes"]) >= {
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer_metadata.json",
    }
    assert stats["source_mixture_realized"]
    assert stats["source_mixture_expected"] == stats["source_mixture_realized_share"]
    assert math.isclose(sum(stats["source_mixture_expected"].values()), 1.0)
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


def test_prepared_manifest_audit_detects_source_text_drift(tmp_path: Path) -> None:
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = 1000
    cfg["validation_tokens"] = 100
    cfg["synthetic_only"]["sample_count"] = 16
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)
    output_dir = prepare_data(config_path)
    manifest_row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    Path(manifest_row["path"]).write_text("tampered text", encoding="utf-8")

    audit = _audit_prepared_manifest(output_dir / "manifest.jsonl", load_tokenizer("simple-byte-tokenizer"))

    assert audit["manifest_source_text_hashes_verified"] is False
    assert audit["manifest_token_hashes_verified"] is False
    assert audit["manifest_source_text_hash_failure_count"] == 1
    assert audit["manifest_token_hash_failure_count"] == 1


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


def test_prepare_mixture_data_records_expected_and_realized_shares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_hf_rows(*, dataset_name: str, split: str, streaming: bool):
        for index in range(30):
            yield {
                "sample_id": f"{dataset_name}-{index}",
                "text": f"{dataset_name} row {index}",
                "source_url_or_id": f"{split}-{index}",
            }

    monkeypatch.setattr("brian_sphere_llm.data.prepare.iter_hf_text_dataset", fake_hf_rows)
    cfg = load_yaml("configs/data/r125_tiny_debug.yaml")
    cfg["output_dir"] = str(tmp_path / "tokenized")
    cfg["manifest_path"] = str(tmp_path / "manifest.jsonl")
    cfg["target_tokens"] = 1200
    cfg["validation_tokens"] = 120
    cfg["sequence_length"] = 32
    cfg["synthetic_only"]["enabled"] = False
    cfg["mixture"] = {
        "synthetic_routing": {"weight": 0.5, "source_dataset": "brian_synthetic_routing"},
        "fineweb_edu": {"weight": 0.25, "source_dataset": "fineweb", "split": "train"},
        "code_structured": {"weight": 0.25, "source_dataset": "code", "split": "train"},
    }
    config_path = tmp_path / "data.yaml"
    save_yaml(cfg, config_path)

    output_dir = prepare_data(config_path)
    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))

    assert stats["source_mixture_expected"] == {
        "synthetic_routing": 0.5,
        "fineweb_edu": 0.25,
        "code_structured": 0.25,
    }
    assert set(stats["source_mixture_realized"]) == set(stats["source_mixture_expected"])
    assert set(stats["source_mixture_realized_share"]) == set(stats["source_mixture_expected"])
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    code_rows = [row for row in manifest_rows if row["mixture_tag"] == "code_structured"]
    assert code_rows
    assert code_rows[0]["source_dataset"] == "code"
    assert code_rows[0]["route_metadata"]["task_family"] == "code_structured"
    assert math.isclose(sum(stats["source_mixture_realized_share"].values()), 1.0)


def test_prepare_mixture_uses_local_synthetic_math_and_code_without_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_hf_rows(*, dataset_name: str, split: str, streaming: bool):
        raise AssertionError(f"unexpected HF load for {dataset_name}")

    monkeypatch.setattr("brian_sphere_llm.data.prepare.iter_hf_text_dataset", fail_hf_rows)

    rows = list(
        islice(
            _mixture_rows(
                {
                    "target_tokens": 1000,
                    "seed": 1,
                    "mixture": {
                        "math_symbolic_qa": {
                            "weight": 0.5,
                            "source_dataset": "synthetic_math_symbolic",
                        },
                        "code_structured": {
                            "weight": 0.5,
                            "source_dataset": "synthetic_code_structured",
                        },
                    },
                }
            ),
            6,
        )
    )

    assert {row["mixture_tag"] for row in rows} == {"math_symbolic_qa", "code_structured"}
    assert {row["source_dataset"] for row in rows} == {"synthetic_math_symbolic", "synthetic_code_structured"}
    assert all(row["route_metadata"]["task_family"] in {"math_symbolic_qa", "code_structured"} for row in rows)


def test_mixture_rows_interleave_sources_by_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_hf_rows(*, dataset_name: str, split: str, streaming: bool):
        for index in range(20):
            yield {
                "sample_id": f"{dataset_name}-{index}",
                "text": f"{dataset_name} row {index}",
                "source_url_or_id": f"{split}-{index}",
            }

    monkeypatch.setattr("brian_sphere_llm.data.prepare.iter_hf_text_dataset", fake_hf_rows)
    rows = list(
        islice(
            _mixture_rows(
                {
                    "target_tokens": 1000,
                    "seed": 1,
                    "mixture": {
                        "synthetic_routing": {
                            "weight": 0.5,
                            "source_dataset": "brian_synthetic_routing",
                        },
                        "fineweb_edu": {
                            "weight": 0.25,
                            "source_dataset": "fineweb",
                            "split": "train",
                        },
                        "code_structured": {
                            "weight": 0.25,
                            "source_dataset": "code",
                            "split": "train",
                        },
                    },
                }
            ),
            16,
        )
    )
    tag_counts = {tag: [row["mixture_tag"] for row in rows].count(tag) for tag in {row["mixture_tag"] for row in rows}}

    assert tag_counts == {"synthetic_routing": 8, "fineweb_edu": 4, "code_structured": 4}
    assert rows[0]["mixture_tag"] == "synthetic_routing"
    assert rows[1]["mixture_tag"] == "fineweb_edu"
    assert rows[2]["mixture_tag"] == "code_structured"


def test_prepare_bool_config_parses_false_string_without_truthiness() -> None:
    assert _bool_config("false", "field") is False
    assert _bool_config("true", "field") is True
    with pytest.raises(ValueError, match="field"):
        _bool_config("maybe", "field")
