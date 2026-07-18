from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.bdre_model import BDREConfig, BrianBDRERouteCore
from brian_sphere_llm.model.brian_model import BrianRouteConfig
from brian_sphere_llm.train.trainer import (
    _backward_stateful_tbptt_microbatch,
    _sync_stateful_ddp_gradients,
)


def _tiny_synchronous_config(*, cache_layout: str = "shared") -> BDREConfig:
    base = BaselineConfig(
        model_name="tiny_stateful_ddp_test",
        layers=4,
        d_model=32,
        n_heads=4,
        context_length=16,
        vocab_size=64,
        dropout=0.0,
    )
    route = BrianRouteConfig(
        base=base,
        pre_blocks=1,
        route_pool_blocks=2,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=3,
        model_name="tiny_stateful_ddp_test",
        top_k=1,
        later_top_k=1,
        hard_exit=True,
        block_position_mode="spherical_code",
        independent_input_position=True,
        location_bias_weight=0.0,
    )
    return BDREConfig(
        route=route,
        key_dim=8,
        value_dim=8,
        cache_layout=cache_layout,
        depth_mode="synchronous_prefix",
        step_lambda=0.25,
        normalize_step_distance=True,
        reader_step_cache="eager",
        self_kv_mode="bdre_prefix",
        execution_mode="synchronous_prefix",
        dispatch_mode="grouped_host",
        chunk_size=8,
        synchronous_attention_backend="shared_padded_explicit",
    )


def _stateful_train_config() -> dict[str, object]:
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


class _RankSplitParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))


def _stateful_ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        split = _RankSplitParameters()
        (split.first.sum() if rank == 0 else split.second.sum()).backward()
        split_stats = _sync_stateful_ddp_gradients(split, bucket_cap_mb=1)
        expected_split_gradient = torch.full((2,), 0.5)
        assert torch.equal(split.first.grad, expected_split_gradient)
        assert torch.equal(split.second.grad, expected_split_gradient)
        assert split_stats["stateful_ddp_global_used_parameters"] == 2
        assert split_stats["stateful_ddp_local_missing_parameters"] == 1

        torch.manual_seed(101)
        distributed_model = BrianBDRERouteCore(_tiny_synchronous_config()).train()
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            distributed_model,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        reference_model = BrianBDRERouteCore(_tiny_synchronous_config()).train()
        reference_model.load_state_dict(ddp_model.module.state_dict())

        batches = (
            torch.tensor([[1, 2, 3, 4, 5, 6]]),
            torch.tensor([[7, 8, 9, 10, 11, 12]]),
        )
        global_batch = torch.cat(batches, dim=0)
        reference_output = _backward_stateful_tbptt_microbatch(
            reference_model,
            global_batch,
            config=_stateful_train_config(),
            route_mode="fixed",
            global_step=1,
            chunk_size=3,
            detach_interval_chunks=2,
            gradient_scale=1.0,
            device=torch.device("cpu"),
            summarize_routing=False,
        )
        with ddp_model.no_sync():
            local_output = _backward_stateful_tbptt_microbatch(
                ddp_model,
                batches[rank],
                config=_stateful_train_config(),
                route_mode="fixed",
                global_step=1,
                chunk_size=3,
                detach_interval_chunks=2,
                gradient_scale=1.0,
                device=torch.device("cpu"),
                summarize_routing=False,
            )
        sync_stats = _sync_stateful_ddp_gradients(ddp_model, bucket_cap_mb=1)
        assert sync_stats["stateful_ddp_gradient_sync_buckets"] > 0

        distributed_loss = local_output["loss"].detach().clone()
        torch.distributed.all_reduce(distributed_loss)
        distributed_loss /= world_size
        assert torch.allclose(distributed_loss, reference_output["loss"], atol=1e-6, rtol=1e-6)

        for distributed_parameter, reference_parameter in zip(
            ddp_model.module.parameters(),
            reference_model.parameters(),
        ):
            if reference_parameter.grad is None:
                assert distributed_parameter.grad is None
                continue
            assert distributed_parameter.grad is not None
            assert torch.allclose(
                distributed_parameter.grad,
                reference_parameter.grad,
                atol=2e-5,
                rtol=2e-5,
            )

        distributed_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
        distributed_optimizer.step()
        reference_optimizer.step()
        for distributed_parameter, reference_parameter in zip(
            ddp_model.module.parameters(),
            reference_model.parameters(),
        ):
            assert torch.allclose(
                distributed_parameter,
                reference_parameter,
                atol=2e-5,
                rtol=2e-5,
            )
    finally:
        torch.distributed.destroy_process_group()


def _per_head_stateful_ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(102)
        distributed_model = BrianBDRERouteCore(
            _tiny_synchronous_config(cache_layout="per_head")
        ).train()
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            distributed_model,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        reference_model = BrianBDRERouteCore(
            _tiny_synchronous_config(cache_layout="per_head")
        ).train()
        reference_model.load_state_dict(ddp_model.module.state_dict())

        batches = (
            torch.tensor([[1, 2, 3, 4, 5, 6]]),
            torch.tensor([[7, 8, 9, 10, 11, 12]]),
        )
        reference_output = _backward_stateful_tbptt_microbatch(
            reference_model,
            torch.cat(batches, dim=0),
            config=_stateful_train_config(),
            route_mode="fixed",
            global_step=1,
            chunk_size=3,
            detach_interval_chunks=2,
            gradient_scale=1.0,
            device=torch.device("cpu"),
            summarize_routing=False,
        )
        with ddp_model.no_sync():
            local_output = _backward_stateful_tbptt_microbatch(
                ddp_model,
                batches[rank],
                config=_stateful_train_config(),
                route_mode="fixed",
                global_step=1,
                chunk_size=3,
                detach_interval_chunks=2,
                gradient_scale=1.0,
                device=torch.device("cpu"),
                summarize_routing=False,
            )
        _sync_stateful_ddp_gradients(ddp_model, bucket_cap_mb=1)

        distributed_loss = local_output["loss"].detach().clone()
        torch.distributed.all_reduce(distributed_loss)
        distributed_loss /= world_size
        assert torch.allclose(distributed_loss, reference_output["loss"], atol=1e-6, rtol=1e-6)
        for distributed_parameter, reference_parameter in zip(
            ddp_model.module.parameters(),
            reference_model.parameters(),
        ):
            if reference_parameter.grad is None:
                assert distributed_parameter.grad is None
                continue
            assert distributed_parameter.grad is not None
            assert torch.allclose(
                distributed_parameter.grad,
                reference_parameter.grad,
                atol=2e-5,
                rtol=2e-5,
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed support is required",
)
def test_stateful_tbptt_ddp_matches_merged_batch_and_handles_rank_local_unused_parameters(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "stateful_ddp_init"
    torch.multiprocessing.spawn(
        _stateful_ddp_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed support is required",
)
def test_per_head_stateful_tbptt_ddp_matches_merged_batch(tmp_path: Path) -> None:
    init_file = tmp_path / "per_head_stateful_ddp_init"
    torch.multiprocessing.spawn(
        _per_head_stateful_ddp_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
