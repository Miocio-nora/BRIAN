from pathlib import Path

from brian_sphere_llm.data.prepare import prepare_data
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
    assert (output_dir / "train.bin").exists()
    assert (output_dir / "train.idx").exists()
    assert (output_dir / "val.bin").exists()
    assert (output_dir / "stats.json").exists()
