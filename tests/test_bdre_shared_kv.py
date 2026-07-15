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
        step_lambda=0.7 if depth_mode == "reader_step" else 0.0,
        reader_step_cache=reader_step_cache,
        self_kv_mode=self_kv_mode,
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


def test_bdre_prefix_logits_are_suffix_invariant() -> None:
    torch.manual_seed(7)
    model = BrianBDRERouteCore(_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 22, 23, 24]])
    with torch.no_grad():
        first_logits = model(first, route_mode="fixed", pseudo_policy="sequential")["logits"]
        second_logits = model(second, route_mode="fixed", pseudo_policy="sequential")["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6)


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
