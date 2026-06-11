import pytest

from brian_sphere_llm.utils import distributed


def test_distributed_env_helpers_default_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert distributed.world_size() == 1
    assert distributed.rank() == 0
    assert distributed.local_rank() == 0
    assert distributed.is_distributed() is False
    assert distributed.is_main_process() is True


def test_distributed_env_helpers_parse_rank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert distributed.world_size() == 8
    assert distributed.rank() == 3
    assert distributed.local_rank() == 1
    assert distributed.is_distributed() is True
    assert distributed.is_main_process() is False


def test_distributed_env_helpers_reject_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        distributed.world_size()

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "-1")
    with pytest.raises(ValueError, match="RANK"):
        distributed.rank()


def test_unwrap_model_returns_module_attribute() -> None:
    class Wrapper:
        module = "inner"

    assert distributed.unwrap_model(Wrapper()) == "inner"
    assert distributed.unwrap_model("plain") == "plain"
