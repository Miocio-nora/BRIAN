from brian_sphere_llm.utils.config import load_config


def test_load_config_extends() -> None:
    cfg = load_config("configs/train/stage0_tiny_debug.yaml")
    assert cfg["stage"] == "stage0_baseline"
    assert cfg["model_config"] == "../model/baseline_tiny.yaml"
    assert cfg["batch_size"] == 2
