from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.train.trainer import _backward_stateful_tbptt_microbatch


@dataclass(frozen=True)
class _ToyState:
    value: torch.Tensor

    def detached(self) -> "_ToyState":
        return _ToyState(self.value.detach())


class _ToyStreamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def prepare_stream_route_targets(self, *args, **kwargs) -> list[torch.Tensor]:
        return []

    def forward(self, input_ids: torch.Tensor, *, stream_chunk_options: dict) -> dict:
        state = stream_chunk_options["state"]
        previous = self.weight.new_zeros(()) if state is None else state.value
        current = previous + self.weight * input_ids.to(self.weight.dtype).sum()
        return {
            "loss": current.square(),
            "loss_components": {},
            "incremental_state": _ToyState(current),
        }


def _config() -> dict[str, object]:
    return {
        "stage": "stage5_bdre_shared_kv",
        "precision": "fp32",
        "routing": {
            "pseudo_policy": "sequential",
            "hard_exit": True,
            "constraints": {},
        },
        "loss_weights": {},
    }


def _run(detach_interval_chunks: int) -> tuple[dict, torch.Tensor]:
    model = _ToyStreamModel()
    output = _backward_stateful_tbptt_microbatch(
        model,
        torch.tensor([[1, 2, 3, 4]]),
        config=_config(),
        route_mode="fixed",
        global_step=1,
        chunk_size=2,
        detach_interval_chunks=detach_interval_chunks,
        gradient_scale=1.0,
        device=torch.device("cpu"),
        summarize_routing=False,
    )
    assert model.weight.grad is not None
    return output, model.weight.grad.detach().clone()


def test_detach_interval_preserves_forward_loss_and_extends_gradient_horizon() -> None:
    u1_output, u1_gradient = _run(1)
    u2_output, u2_gradient = _run(2)

    assert torch.equal(u1_output["loss"], u2_output["loss"])
    assert not torch.equal(u1_gradient, u2_gradient)
    assert u1_output["stateful_tbptt_chunks"] == 2
    assert u1_output["stateful_tbptt_backward_groups"] == 2
    assert u1_output["stateful_tbptt_gradient_horizon_tokens"] == 2
    assert u2_output["stateful_tbptt_backward_groups"] == 1
    assert u2_output["stateful_tbptt_gradient_horizon_tokens"] == 4
