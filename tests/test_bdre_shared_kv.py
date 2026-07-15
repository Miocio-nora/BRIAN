from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from brian_sphere_llm.eval.bdre_cache_visualization import make_bdre_cache_visualization_from_payload
from brian_sphere_llm.memory.bdre_shared_kv import BDRECacheState, BDRECompiler
from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.bdre_model import BDREConfig, BrianBDRERouteCore
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.routing.block_position import BlockPositionTable
from brian_sphere_llm.train.stage_runner import build_model_from_config


def _config(
    *,
    depth_mode: str = "none",
    reader_step_cache: str = "lazy",
    self_kv_mode: str = "bdre_prefix",
    execution_mode: str = "token_by_token",
    chunk_size: int = 1,
) -> BDREConfig:
    base = BaselineConfig(
        model_name="tiny_bdre_test",
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
        model_name="tiny_bdre_test",
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
        depth_mode=depth_mode,
        step_lambda=0.7 if depth_mode == "reader_step" else 0.25 if depth_mode == "synchronous_prefix" else 0.0,
        normalize_step_distance=depth_mode == "synchronous_prefix",
        reader_step_cache=reader_step_cache,
        self_kv_mode=self_kv_mode,
        execution_mode=execution_mode,
        dispatch_mode="grouped_host" if execution_mode == "synchronous_prefix" else "legacy_cuda_scan",
        chunk_size=chunk_size,
    )


def _synchronous_config(
    *,
    chunk_size: int = 8,
    attention_backend: str = "per_query_reference",
) -> BDREConfig:
    return replace(
        _config(
            depth_mode="synchronous_prefix",
            reader_step_cache="eager",
            execution_mode="synchronous_prefix",
            chunk_size=chunk_size,
        ),
        synchronous_attention_backend=attention_backend,
    )


def test_spherical_code_matches_in_internal_out_gram_contract() -> None:
    table = BlockPositionTable(8, 64, mode="spherical_code", independent_input_position=True)
    points = torch.cat(
        [table.initial(1, torch.device("cpu")), table.embeddings[:8], table.embeddings[8:]],
        dim=0,
    )
    gram = F.normalize(points, dim=-1) @ F.normalize(points, dim=-1).T

    assert torch.allclose(torch.diag(gram), torch.ones(10), atol=2e-6)
    assert gram[0, -1].item() == pytest.approx(-1.0, abs=2e-6)
    assert torch.allclose(gram[0, 1:-1], torch.zeros(8), atol=2e-6)
    assert torch.allclose(gram[-1, 1:-1], torch.zeros(8), atol=2e-6)
    off_diagonal = gram[1:-1, 1:-1][~torch.eye(8, dtype=torch.bool)]
    assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, -1.0 / 7.0), atol=2e-6)
    assert table.geometry_loss().item() < 1e-10


def test_spherical_code_requires_independent_input_position() -> None:
    with pytest.raises(ValueError, match="independent_input_position"):
        BlockPositionTable(2, 8, mode="spherical_code", independent_input_position=False)


def test_embedding_position_lookup_matches_advanced_index_gradients() -> None:
    torch.manual_seed(11)
    actions = torch.tensor([[0, 2, 2, 1], [1, 0, 2, 2]])
    legacy_positions = torch.randn(3, 8, requires_grad=True)
    optimized_positions = legacy_positions.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 4, 8)

    legacy = F.normalize(legacy_positions, dim=-1)[actions]
    optimized = F.embedding(actions, F.normalize(optimized_positions, dim=-1))
    (legacy * upstream).sum().backward()
    (optimized * upstream).sum().backward()

    assert torch.equal(legacy, optimized)
    assert legacy_positions.grad is not None and optimized_positions.grad is not None
    assert torch.equal(legacy_positions.grad, optimized_positions.grad)


def test_bdre_compiler_emits_one_pair_per_reader_and_cache_per_token() -> None:
    compiler = BDRECompiler(max_route_steps=4, key_temperature=0.5, value_temperature=1.0)
    writer_key = torch.randn(2, 4, 5)
    writer_value = torch.randn(2, 4, 7)
    writer_blocks = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    positions = F.normalize(torch.randn(3, 8), dim=-1)

    compiled = compiler.compile(writer_key, writer_value, writer_blocks, valid, positions)
    assert compiled.keys.shape == (2, 3, 5)
    assert compiled.values.shape == (2, 3, 7)
    state = BDRECacheState().append(compiled.keys, compiled.values)
    state = state.append(compiled.keys + 1.0, compiled.values + 1.0)
    stacked_key, stacked_value = state.stacked_blocks()
    assert stacked_key.shape == (2, 2, 3, 5)
    assert stacked_value.shape == (2, 2, 3, 7)
    assert state.tokens == 2


def test_bdre_hard_topk_and_temperatures_are_effective_and_differentiable() -> None:
    compiler = BDRECompiler(
        max_route_steps=4,
        key_temperature=0.25,
        value_temperature=2.0,
        compile_top_k=2,
    )
    writer_key = torch.randn(2, 4, 5, requires_grad=True)
    writer_value = torch.randn(2, 4, 7, requires_grad=True)
    writer_blocks = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    valid = torch.ones(2, 4, dtype=torch.bool)
    positions = F.normalize(torch.randn(3, 8), dim=-1)
    output = compiler.compile(writer_key, writer_value, writer_blocks, valid, positions)

    assert torch.equal((output.key_weights > 0).sum(dim=-1), torch.full((2, 3), 2))
    assert not torch.allclose(output.key_weights, output.value_weights)
    (output.keys.square().mean() + output.values.square().mean()).backward()
    assert writer_key.grad is not None and torch.isfinite(writer_key.grad).all()
    assert writer_value.grad is not None and torch.isfinite(writer_value.grad).all()


def test_synchronous_prefix_compiler_masks_future_writer_steps() -> None:
    compiler = BDRECompiler(
        max_route_steps=4,
        key_temperature=0.5,
        value_temperature=1.0,
        step_lambda=0.25,
        normalize_step_distance=True,
    )
    writer_key = torch.randn(2, 4, 5)
    writer_value = torch.randn(2, 4, 7)
    writer_blocks = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    valid = torch.ones(2, 4, dtype=torch.bool)
    positions = F.normalize(torch.randn(3, 8), dim=-1)
    changed_key = writer_key.clone()
    changed_value = writer_value.clone()
    changed_key[:, 2:] += 100.0
    changed_value[:, 2:] -= 100.0

    prefix = compiler.compile_prefix(
        writer_key,
        writer_value,
        writer_blocks,
        valid,
        positions,
        reader_step=1,
    )
    changed = compiler.compile_prefix(
        changed_key,
        changed_value,
        writer_blocks,
        valid,
        positions,
        reader_step=1,
    )
    later = compiler.compile_prefix(
        changed_key,
        changed_value,
        writer_blocks,
        valid,
        positions,
        reader_step=3,
    )

    assert torch.allclose(prefix.keys, changed.keys)
    assert torch.allclose(prefix.values, changed.values)
    assert not torch.allclose(prefix.keys, later.keys)
    assert torch.equal((prefix.key_weights > 0).sum(dim=-1), torch.full((2, 3), 2))


def test_bdre_cache_appends_contiguous_synchronous_token_chunks() -> None:
    state = BDRECacheState()
    block_key = torch.randn(2, 3, 4, 5)
    block_value = torch.randn(2, 3, 4, 7)
    step_key = torch.randn(2, 3, 6, 4, 5)
    step_value = torch.randn(2, 3, 6, 4, 7)
    state = state.append_tokens(
        block_key,
        block_value,
        step_key=step_key,
        step_value=step_value,
    )
    state = state.append_tokens(
        block_key + 1.0,
        block_value + 1.0,
        step_key=step_key + 1.0,
        step_value=step_value + 1.0,
    )

    assert state.tokens == 6
    assert state.block_keys is not None and state.block_keys.shape == (2, 6, 4, 5)
    assert state.step_keys is not None and state.step_keys.shape == (2, 6, 6, 4, 5)


def test_bdre_projection_parameters_are_independent_per_block_and_head() -> None:
    model = BrianBDRERouteCore(_config())
    first, second = model.bdre_projections
    first_ids = {id(parameter) for parameter in first.parameters()}
    second_ids = {id(parameter) for parameter in second.parameters()}
    assert first_ids.isdisjoint(second_ids)
    assert first.key_read.shape == (4, 8, 8)
    assert first.value_read.shape == (4, 8, 8)
    assert first.key_read.data_ptr() != first.value_read.data_ptr()


def test_bdre_exact_forward_backward_and_incremental_match() -> None:
    torch.manual_seed(4)
    model = BrianBDRERouteCore(_config())
    input_ids = torch.randint(0, 64, (2, 5))
    output = model(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
        loss_weights={"route": 0.1},
    )
    assert output["logits"].shape == (2, 5, 64)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.bdre_projections[0].key_write.weight.grad is not None
    assert torch.isfinite(model.bdre_projections[0].key_write.weight.grad).all()

    model.eval()
    with torch.no_grad():
        full = model(input_ids, route_mode="fixed", pseudo_policy="sequential")["logits"]
        state = None
        incremental = []
        for token in input_ids.unbind(dim=1):
            step = model.forward_incremental(
                token,
                state,
                route_mode="fixed",
                pseudo_policy="sequential",
            )
            state = step["incremental_state"]
            incremental.append(step["logits"])
    assert state is not None
    assert state.cache.tokens == input_ids.size(1)
    assert torch.allclose(full, torch.cat(incremental, dim=1), atol=1e-6, rtol=1e-6)


def test_bdre_stream_chunks_preserve_exact_forward_and_loss_values() -> None:
    torch.manual_seed(41)
    model = BrianBDRERouteCore(_config()).eval()
    input_ids = torch.randint(0, 64, (3, 7))
    full = model(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    route_targets = model.prepare_stream_route_targets(
        input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    state = None
    chunk_logits = []
    chunk_losses = []
    for start in range(0, input_ids.size(1), 2):
        end = min(input_ids.size(1), start + 2)
        chunk = input_ids[:, start:end]
        targets = torch.full_like(chunk, -100)
        valid = max(0, min(end, input_ids.size(1) - 1) - start)
        if valid:
            targets[:, :valid] = input_ids[:, start + 1 : start + 1 + valid]
        output = model.forward_stream_chunk(
            chunk,
            state,
            next_token_targets=targets,
            loss_token_count=input_ids.size(0) * (input_ids.size(1) - 1),
            include_auxiliary_losses=end == input_ids.size(1),
            route_targets=route_targets,
            route_mode="fixed",
            pseudo_policy="sequential",
        )
        chunk_logits.append(output["logits"])
        chunk_losses.append(output["loss"])
        state = output["incremental_state"].detached()

    assert state is not None
    assert state.cache.tokens == input_ids.size(1)
    assert torch.allclose(full["logits"], torch.cat(chunk_logits, dim=1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(full["loss"], torch.stack(chunk_losses).sum(), atol=3e-6, rtol=1e-6)


def test_bdre_stream_state_detach_allows_independent_chunk_backwards() -> None:
    torch.manual_seed(43)
    model = BrianBDRERouteCore(_config()).train()
    input_ids = torch.randint(0, 64, (2, 4))
    route_targets = model.prepare_stream_route_targets(
        input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    first = model.forward_stream_chunk(
        input_ids[:, :2],
        next_token_targets=input_ids[:, 1:3],
        loss_token_count=6,
        route_targets=route_targets,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    first["loss"].backward()
    state = first["incremental_state"].detached()
    assert state.cache.block_keys is not None and state.cache.block_keys.grad_fn is None
    assert state.pre[0].keys is not None and state.pre[0].keys.grad_fn is None

    final_targets = torch.full_like(input_ids[:, 2:], -100)
    final_targets[:, 0] = input_ids[:, 3]
    second = model.forward_stream_chunk(
        input_ids[:, 2:],
        state,
        next_token_targets=final_targets,
        loss_token_count=6,
        include_auxiliary_losses=True,
        route_targets=route_targets,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    second["loss"].backward()
    assert model.bdre_projections[0].key_write.weight.grad is not None
    assert torch.isfinite(model.bdre_projections[0].key_write.weight.grad).all()


def test_bdre_single_stream_chunk_matches_exact_gradients() -> None:
    torch.manual_seed(45)
    exact = BrianBDRERouteCore(_config()).train()
    stream = BrianBDRERouteCore(_config()).train()
    stream.load_state_dict(exact.state_dict())
    input_ids = torch.randint(0, 64, (2, 5))

    exact_output = exact(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    exact_output["loss"].backward()

    next_targets = torch.full_like(input_ids, -100)
    next_targets[:, :-1] = input_ids[:, 1:]
    stream_output = stream.forward_stream_chunk(
        input_ids,
        next_token_targets=next_targets,
        loss_token_count=input_ids.size(0) * (input_ids.size(1) - 1),
        include_auxiliary_losses=True,
        route_targets=stream.prepare_stream_route_targets(
            input_ids,
            route_mode="fixed",
            pseudo_policy="sequential",
        ),
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    stream_output["loss"].backward()

    assert torch.allclose(exact_output["loss"], stream_output["loss"], atol=2e-6, rtol=1e-6)
    for exact_parameter, stream_parameter in (
        (exact.token_embedding.weight, stream.token_embedding.weight),
        (exact.bdre_projections[0].key_write.weight, stream.bdre_projections[0].key_write.weight),
        (exact.lm_head.weight, stream.lm_head.weight),
    ):
        assert exact_parameter.grad is not None and stream_parameter.grad is not None
        assert torch.allclose(exact_parameter.grad, stream_parameter.grad, atol=2e-6, rtol=1e-5)


def test_bdre_grouped_host_dispatch_matches_legacy_cuda_scan() -> None:
    torch.manual_seed(47)
    legacy = BrianBDRERouteCore(_config()).eval()
    grouped = BrianBDRERouteCore(replace(_config(), dispatch_mode="grouped_host")).eval()
    grouped.load_state_dict(legacy.state_dict())
    input_ids = torch.randint(0, 64, (5, 6))
    with torch.no_grad():
        legacy_logits = legacy(input_ids, route_mode="fixed", pseudo_policy="sequential")["logits"]
        grouped_logits = grouped(input_ids, route_mode="fixed", pseudo_policy="sequential")["logits"]
    assert torch.allclose(legacy_logits, grouped_logits, atol=1e-6, rtol=1e-6)


def test_bdre_prefix_logits_are_suffix_invariant() -> None:
    torch.manual_seed(7)
    model = BrianBDRERouteCore(_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 22, 23, 24]])
    with torch.no_grad():
        first_logits = model(first, route_mode="fixed", pseudo_policy="sequential")["logits"]
        second_logits = model(second, route_mode="fixed", pseudo_policy="sequential")["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("attention_backend", ["per_query_reference", "shared_padded_explicit"])
def test_synchronous_prefix_is_chunk_boundary_and_incremental_invariant(attention_backend: str) -> None:
    torch.manual_seed(53)
    config = _synchronous_config(chunk_size=8, attention_backend=attention_backend)
    full = BrianBDRERouteCore(config).eval()
    streamed = BrianBDRERouteCore(config).eval()
    incremental = BrianBDRERouteCore(config).eval()
    streamed.load_state_dict(full.state_dict())
    incremental.load_state_dict(full.state_dict())
    input_ids = torch.randint(0, 64, (3, 7))

    with torch.no_grad():
        full_logits = full(input_ids, route_mode="fixed", pseudo_policy="sequential")["logits"]
        state = None
        stream_logits = []
        for start in range(0, input_ids.size(1), 2):
            output = streamed.forward_stream_chunk(
                input_ids[:, start : start + 2],
                state,
                route_mode="fixed",
                pseudo_policy="sequential",
            )
            state = output["incremental_state"]
            stream_logits.append(output["logits"])
        incremental_state = None
        incremental_logits = []
        for token in input_ids.unbind(dim=1):
            output = incremental.forward_incremental(
                token,
                incremental_state,
                route_mode="fixed",
                pseudo_policy="sequential",
            )
            incremental_state = output["incremental_state"]
            incremental_logits.append(output["logits"])

    assert state is not None and state.cache.step_keys is not None
    assert state.cache.step_keys.shape == (3, 7, 3, 2, 8)
    assert torch.allclose(full_logits, torch.cat(stream_logits, dim=1), atol=2e-5, rtol=2e-5)
    assert torch.allclose(full_logits, torch.cat(incremental_logits, dim=1), atol=2e-5, rtol=2e-5)


def test_synchronous_prefix_is_suffix_invariant() -> None:
    torch.manual_seed(59)
    model = BrianBDRERouteCore(_synchronous_config(chunk_size=8)).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 22, 23, 24]])
    with torch.no_grad():
        first_logits = model(first, route_mode="fixed", pseudo_policy="sequential")["logits"]
        second_logits = model(second, route_mode="fixed", pseudo_policy="sequential")["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=2e-5, rtol=2e-5)


def test_synchronous_prefix_forward_backward_is_finite() -> None:
    torch.manual_seed(61)
    model = BrianBDRERouteCore(_synchronous_config(chunk_size=4)).train()
    input_ids = torch.randint(0, 64, (2, 4))
    output = model(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.bdre_projections[0].key_write.weight.grad is not None
    assert torch.isfinite(model.bdre_projections[0].key_write.weight.grad).all()


def test_shared_padded_explicit_matches_per_query_reference_gradients() -> None:
    torch.manual_seed(67)
    reference = BrianBDRERouteCore(_synchronous_config(chunk_size=8)).train()
    optimized = BrianBDRERouteCore(
        _synchronous_config(chunk_size=8, attention_backend="shared_padded_explicit")
    ).train()
    optimized.load_state_dict(reference.state_dict())
    input_ids = torch.randint(0, 64, (3, 7))

    reference_output = reference(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    optimized_output = optimized(
        input_ids,
        targets=input_ids,
        route_mode="fixed",
        pseudo_policy="sequential",
    )
    reference_output["loss"].backward()
    optimized_output["loss"].backward()

    assert torch.allclose(reference_output["logits"], optimized_output["logits"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(reference_output["loss"], optimized_output["loss"], atol=2e-6, rtol=2e-6)
    for reference_parameter, optimized_parameter in (
        (reference.token_embedding.weight, optimized.token_embedding.weight),
        (reference.route_blocks[0].block.attn.qkv.weight, optimized.route_blocks[0].block.attn.qkv.weight),
        (reference.bdre_projections[0].key_read, optimized.bdre_projections[0].key_read),
        (reference.bdre_projections[0].value_read, optimized.bdre_projections[0].value_read),
    ):
        assert reference_parameter.grad is not None and optimized_parameter.grad is not None
        assert torch.allclose(reference_parameter.grad, optimized_parameter.grad, atol=2e-5, rtol=2e-5)


def test_synchronous_prefix_visualization_masks_future_writer_steps() -> None:
    model = BrianBDRERouteCore(_synchronous_config(chunk_size=4)).eval()
    with torch.no_grad():
        output = model(
            torch.randint(0, 64, (1, 3)),
            route_mode="fixed",
            pseudo_policy="sequential",
            collect_bdre_visualization=True,
        )

    payload = output["bdre_visualization"]
    key_weights = payload["reader_step_key_weights"]
    value_weights = payload["reader_step_value_weights"]
    assert key_weights.shape == (1, 3, 2, 3)
    assert value_weights.shape == (1, 3, 2, 3)
    assert torch.count_nonzero(key_weights[:, 0, :, 1:]) == 0
    assert torch.count_nonzero(value_weights[:, 0, :, 1:]) == 0
    assert torch.count_nonzero(key_weights[:, 1, :, 2:]) == 0
    assert torch.count_nonzero(value_weights[:, 1, :, 2:]) == 0


@pytest.mark.parametrize("self_kv_mode", ["bdre_prefix", "current_step", "none"])
def test_bdre_self_modes_are_finite(self_kv_mode: str) -> None:
    model = BrianBDRERouteCore(_config(self_kv_mode=self_kv_mode))
    input_ids = torch.randint(0, 64, (2, 4))
    output = model(input_ids, targets=input_ids, route_mode="fixed", pseudo_policy="sequential")
    assert torch.isfinite(output["loss"])


def test_reader_step_eager_lazy_and_dynamic_are_equivalent() -> None:
    torch.manual_seed(11)
    eager = BrianBDRERouteCore(_config(depth_mode="reader_step", reader_step_cache="eager")).eval()
    lazy = BrianBDRERouteCore(_config(depth_mode="reader_step", reader_step_cache="lazy")).eval()
    dynamic = BrianBDRERouteCore(_config(depth_mode="reader_step", reader_step_cache="dynamic")).eval()
    lazy.load_state_dict(eager.state_dict())
    dynamic.load_state_dict(eager.state_dict())
    input_ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        eager_output = eager(input_ids, route_mode="fixed", pseudo_policy="sequential")
        lazy_output = lazy(input_ids, route_mode="fixed", pseudo_policy="sequential")
        dynamic_output = dynamic(input_ids, route_mode="fixed", pseudo_policy="sequential")
    eager_logits = eager_output["logits"]
    lazy_logits = lazy_output["logits"]
    dynamic_logits = dynamic_output["logits"]
    assert torch.allclose(eager_logits, lazy_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(eager_logits, dynamic_logits, atol=1e-6, rtol=1e-6)
    assert lazy_output["routing_summary"]["bdre_lazy_cache_hit_rate"] > 0.0


def test_config_dispatch_is_additive_and_old_model_remains_legacy() -> None:
    bdre = build_model_from_config("configs/model/brian_tiny_bdre_rckv.yaml")
    legacy = build_model_from_config("configs/model/brian_tiny.yaml")
    assert isinstance(bdre, BrianBDRERouteCore)
    assert isinstance(legacy, BrianRouteCore)
    assert not isinstance(legacy, BrianBDRERouteCore)
    assert not hasattr(legacy, "bdre_projections")


def test_bdre_visualization_writes_geometry_and_reader_weights(tmp_path) -> None:
    model = BrianBDRERouteCore(_config()).eval()
    input_ids = torch.randint(0, 64, (2, 4))
    with torch.no_grad():
        output = model(
            input_ids,
            route_mode="fixed",
            pseudo_policy="sequential",
            collect_bdre_visualization=True,
        )
    html = make_bdre_cache_visualization_from_payload(
        output["bdre_visualization"],
        model,
        output_path=tmp_path / "bdre.html",
        step=3,
    )
    report = json.loads(html.with_suffix(".json").read_text(encoding="utf-8"))
    assert html.exists()
    assert report["overall_status"] == "pass"
    assert len(report["nodes"]) == 4
    assert len(report["key_weights"]) == 2


def test_reader_step_visualization_contains_each_configured_step() -> None:
    model = BrianBDRERouteCore(_config(depth_mode="reader_step", reader_step_cache="lazy")).eval()
    with torch.no_grad():
        output = model(
            torch.randint(0, 64, (1, 3)),
            route_mode="fixed",
            pseudo_policy="sequential",
            collect_bdre_visualization=True,
        )
    payload = output["bdre_visualization"]
    assert payload["reader_step_key_weights"].shape == (1, 3, 2, 3)
    assert payload["reader_step_value_weights"].shape == (1, 3, 2, 3)


def test_bdre_rejects_top2_without_touching_legacy_config() -> None:
    config = _config()
    with pytest.raises(ValueError, match="top_k=later_top_k=1"):
        replace(config, route=replace(config.route, later_top_k=2)).validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bdre_cuda_bf16_forward_backward_is_finite() -> None:
    model = BrianBDRERouteCore(_config()).cuda().train()
    input_ids = torch.randint(0, 64, (2, 5), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, targets=input_ids, route_mode="fixed", pseudo_policy="sequential")
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.bdre_projections[0].key_write.weight.grad is not None
    assert torch.isfinite(model.bdre_projections[0].key_write.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("attention_backend", ["per_query_reference", "shared_padded_explicit"])
def test_synchronous_prefix_cuda_bf16_forward_backward_is_finite(attention_backend: str) -> None:
    model = BrianBDRERouteCore(
        _synchronous_config(chunk_size=4, attention_backend=attention_backend)
    ).cuda().train()
    input_ids = torch.randint(0, 64, (2, 4), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, targets=input_ids, route_mode="fixed", pseudo_policy="sequential")
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.bdre_projections[0].key_write.weight.grad is not None
    assert torch.isfinite(model.bdre_projections[0].key_write.weight.grad).all()
