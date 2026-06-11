import pytest

torch = pytest.importorskip("torch")

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


def test_mean_scalar_noops_when_distributed_is_not_initialized() -> None:
    assert distributed.mean_scalar(3.5) == 3.5


def test_mean_scalar_uses_all_reduce_sum_and_world_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distributed.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.torch.distributed, "get_world_size", lambda: 4)

    def fake_all_reduce(tensor, op=None):
        tensor.fill_(20.0)

    monkeypatch.setattr(distributed.torch.distributed, "all_reduce", fake_all_reduce)

    assert distributed.mean_scalar(99.0, device=torch.device("cpu")) == pytest.approx(5.0)
