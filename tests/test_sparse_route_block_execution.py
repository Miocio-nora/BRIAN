from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.model.baseline import BaselineConfig
from brian_sphere_llm.model.brian_model import BrianRouteConfig, BrianRouteCore
from brian_sphere_llm.memory.attention_global_cache import CanonicalAttentionGlobalKVCache


def _paired_models(route_block_execution: str = "sparse") -> tuple[BrianRouteCore, BrianRouteCore]:
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
    )
    full = BrianRouteCore(cfg)
    sparse = BrianRouteCore(replace(cfg, route_block_execution=route_block_execution))
    sparse.load_state_dict(full.state_dict())
    full.eval()
    sparse.eval()
    return full, sparse


def _position(model: BrianRouteCore, batch_size: int, seq_len: int) -> torch.Tensor:
    base = model.position_table.initial(batch_size, torch.device("cpu"))
    return base.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


@pytest.mark.parametrize("route_block_execution", ["sparse", "sparse_varlen", "grouped_dense"])
def test_sparse_route_block_execution_matches_full_sequence_top1(route_block_execution: str) -> None:
    torch.manual_seed(11)
    full, sparse = _paired_models(route_block_execution)
    hidden = torch.randn(2, 6, 64)
    position = _position(full, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, full.out_action, 0, 1],
            [2, 0, full.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("route_block_execution", ["sparse", "sparse_varlen", "grouped_dense"])
def test_sparse_route_block_execution_handles_empty_batch_rows_for_action(route_block_execution: str) -> None:
    torch.manual_seed(13)
    full, sparse = _paired_models(route_block_execution)
    hidden = torch.randn(3, 6, 64)
    position = _position(full, 3, 6)
    selected = torch.tensor(
        [
            [0, 0, 0, full.out_action, 0, 0],
            [1, 1, 1, full.out_action, 1, 1],
            [2, 0, full.out_action, 2, 0, 2],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("route_block_execution", ["sparse", "sparse_varlen", "grouped_dense"])
def test_sparse_route_block_execution_matches_full_sequence_weighted_fusion(route_block_execution: str) -> None:
    torch.manual_seed(17)
    full, sparse = _paired_models(route_block_execution)
    hidden = torch.randn(2, 6, 64)
    position = _position(full, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    top_actions = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 0], [0, full.out_action], [1, 0], [2, 1]],
            [[2, 1], [0, 2], [1, full.out_action], [2, 0], [0, 1], [1, 2]],
        ],
        dtype=torch.long,
    )
    top_weights = torch.tensor(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [1.0, 0.0], [0.8, 0.2], [0.55, 0.45]],
            [[0.65, 0.35], [0.9, 0.1], [1.0, 0.0], [0.75, 0.25], [0.6, 0.4], [0.5, 0.5]],
        ],
        dtype=hidden.dtype,
    )
    use_weighted_fusion = torch.tensor(
        [
            [True, False, True, True, False, True],
            [True, True, True, False, True, False],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("route_block_execution", ["sparse", "sparse_varlen"])
def test_sparse_route_block_execution_supports_bf16_autocast(route_block_execution: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA autocast coverage requires CUDA.")
    torch.manual_seed(23)
    _, sparse = _paired_models(route_block_execution)
    sparse = sparse.cuda()
    hidden = torch.randn(2, 6, 64, device="cuda")
    position = _position(sparse, 2, 6).cuda()
    selected = torch.tensor(
        [
            [0, 1, 2, sparse.out_action, 0, 1],
            [2, 0, sparse.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
        device="cuda",
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1, device="cuda")
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert actual.shape == hidden.shape
    assert torch.isfinite(actual).all()


def test_sparse_varlen_route_block_execution_cuda_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA backward coverage requires CUDA.")
    torch.manual_seed(31)
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        route_block_execution="sparse_varlen",
    )
    model = BrianRouteCore(cfg).cuda().train()
    input_ids = torch.randint(0, 64, (2, 8), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids)["logits"].float().mean()
    loss.backward()

    grad = model.route_blocks[0].block.attn.qkv.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_grouped_dense_route_block_execution_cuda_backward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA backward coverage requires CUDA.")
    torch.manual_seed(33)
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        route_block_execution="grouped_dense",
    )
    model = BrianRouteCore(cfg).cuda().train()
    input_ids = torch.randint(0, 64, (2, 8), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids)["logits"].float().mean()
    loss.backward()

    grad = model.route_blocks[0].block.attn.qkv.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_sparse_varlen_dense_backend_matches_full_sequence_top1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIAN_SPARSE_VARLEN_BACKEND", "dense")
    torch.manual_seed(37)
    full, sparse = _paired_models("sparse_varlen")
    hidden = torch.randn(2, 6, 64)
    position = _position(full, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, full.out_action, 0, 1],
            [2, 0, full.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = full._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)
        actual = sparse._apply_routed_blocks(hidden, position, selected, top_actions, top_weights, use_weighted_fusion)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_sparse_varlen_route_block_execution_is_suffix_invariant() -> None:
    torch.manual_seed(29)
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        route_block_execution="sparse_varlen",
    )
    model = BrianRouteCore(cfg).eval()
    prefix = torch.tensor([[3, 7, 11, 13]], dtype=torch.long)
    suffix_a = torch.tensor([[17, 19, 23, 29]], dtype=torch.long)
    suffix_b = torch.tensor([[31, 37, 41, 43]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(torch.cat([prefix, suffix_a], dim=1))["logits"]
        logits_b = model(torch.cat([prefix, suffix_b], dim=1))["logits"]

    assert torch.allclose(logits_a[:, : prefix.size(1)], logits_b[:, : prefix.size(1)], atol=1e-5, rtol=1e-5)


def test_pure_factorized_attention_global_kv_is_suffix_invariant_and_shares_writers() -> None:
    torch.manual_seed(41)
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        attention_global_kv=True,
        attention_global_kv_mode="pure_factorized",
        attention_global_code_dim=12,
        attention_global_sink_slots=0,
        attention_global_window_slots=0,
        attention_global_logit_bias_init=-2.0,
    )
    model = BrianRouteCore(cfg).eval()
    assert model.route_blocks[0].block.attn.global_key_write is model.route_blocks[1].block.attn.global_key_write
    assert model.route_blocks[0].block.attn.global_value_write is model.route_blocks[2].block.attn.global_value_write

    prefix = torch.tensor([[3, 7, 11, 13]], dtype=torch.long)
    suffix_a = torch.tensor([[17, 19, 23, 29]], dtype=torch.long)
    suffix_b = torch.tensor([[31, 37, 41, 43]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(torch.cat([prefix, suffix_a], dim=1))["logits"]
        logits_b = model(torch.cat([prefix, suffix_b], dim=1))["logits"]

    assert torch.allclose(logits_a[:, : prefix.size(1)], logits_b[:, : prefix.size(1)], atol=1e-5, rtol=1e-5)


def test_latest_attention_global_kv_cache_preserves_invalid_tokens() -> None:
    cache = CanonicalAttentionGlobalKVCache(0, 0, latest_token_only=True)
    state = cache.empty(
        batch_size=1,
        n_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sequence_length=3,
    )
    first_key = torch.tensor([[[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]]])
    first_value = first_key + 10.0
    first_valid = torch.tensor([[[True, True, True]]])
    state = cache.write(state, first_key, first_value, first_valid)

    second_key = torch.tensor([[[[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]]]]])
    second_value = second_key + 10.0
    second_valid = torch.tensor([[[False, True, False]]])
    state = cache.write(state, second_key, second_value, second_valid)

    assert state.slots == 1
    assert torch.equal(state.valid, first_valid)
    assert torch.allclose(state.keys[0, 0, 0], torch.tensor([[1.0, 1.0], [8.0, 8.0], [3.0, 3.0]]))
    assert torch.allclose(state.values[0, 0, 0], torch.tensor([[11.0, 11.0], [18.0, 18.0], [13.0, 13.0]]))


def _pure_factorized_cfg(*, attention_global_route_execution: str = "selected") -> BrianRouteConfig:
    return BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        attention_global_kv=True,
        attention_global_kv_mode="pure_factorized",
        attention_global_code_dim=12,
        attention_global_sink_slots=0,
        attention_global_window_slots=0,
        attention_global_logit_bias_init=-2.0,
        attention_global_route_execution=attention_global_route_execution,
    )


def _pure_factorized_state(model: BrianRouteCore, batch_size: int, seq_len: int):
    assert model.attention_global_cache is not None
    state = model.attention_global_cache.empty(
        batch_size=batch_size,
        n_heads=model._attention_global_cache_heads(),
        head_dim=model._attention_global_cache_dim(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        sequence_length=seq_len,
    )
    key = torch.randn(batch_size, 1, 1, seq_len, model._attention_global_cache_dim())
    value = torch.randn_like(key)
    valid = torch.ones(batch_size, 1, seq_len, dtype=torch.bool)
    return model.attention_global_cache.write(state, key, value, valid)


def _attention_global_route_info() -> dict[str, list[torch.Tensor]]:
    return {
        "attention_global_kv_logit_bias": [],
        "attention_global_kv_last_token_mass": [],
        "attention_global_kv_sink_last_token_mass": [],
        "attention_global_kv_window_last_token_mass": [],
    }


def _pure_factorized_full_sequence_reference(
    model: BrianRouteCore,
    hidden: torch.Tensor,
    position: torch.Tensor,
    selected: torch.Tensor,
    top_actions: torch.Tensor,
    top_weights: torch.Tensor,
    use_weighted_fusion: torch.Tensor,
    attention_global_state,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_hidden = hidden.clone()
    write_key, write_value, write_valid = model._empty_attention_global_write(hidden)
    block_position = model._block_position(position)
    for action, block in enumerate(model.route_blocks):
        top1_mask = (selected == action) & ~use_weighted_fusion
        if torch.any(top1_mask):
            output, key_summary, value_summary, _ = block(
                hidden,
                block_position,
                attention_global_state,
                return_attention_kv=True,
            )
            key_write = model._attention_global_write_tensor(key_summary).to(write_key.dtype)
            value_write = model._attention_global_write_tensor(value_summary).to(write_value.dtype)
            token_mask = top1_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            next_hidden = torch.where(top1_mask.unsqueeze(-1), output, next_hidden)
            write_key = torch.where(token_mask, key_write, write_key)
            write_value = torch.where(token_mask, value_write, write_value)
            write_valid = write_valid | top1_mask.unsqueeze(1)

    if torch.any(use_weighted_fusion):
        accum = torch.zeros_like(hidden)
        key_accum = torch.zeros_like(write_key)
        value_accum = torch.zeros_like(write_value)
        weight_sum = torch.zeros_like(selected, dtype=hidden.dtype)
        for action, block in enumerate(model.route_blocks):
            action_output = None
            key_write = None
            value_write = None
            for rank in range(top_actions.size(-1)):
                mask = use_weighted_fusion & (top_actions[..., rank] == action)
                if not torch.any(mask):
                    continue
                if action_output is None:
                    action_output, key_summary, value_summary, _ = block(
                        hidden,
                        block_position,
                        attention_global_state,
                        return_attention_kv=True,
                    )
                    key_write = model._attention_global_write_tensor(key_summary).to(key_accum.dtype)
                    value_write = model._attention_global_write_tensor(value_summary).to(value_accum.dtype)
                weight = top_weights[..., rank].to(hidden.dtype) * mask.to(hidden.dtype)
                accum = accum + action_output * weight.unsqueeze(-1)
                key_weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
                assert key_write is not None and value_write is not None
                key_accum = key_accum + key_write * key_weight
                value_accum = value_accum + value_write * key_weight
                weight_sum = weight_sum + weight
        weighted_mask = use_weighted_fusion & (weight_sum > 0)
        if torch.any(weighted_mask):
            denom = weight_sum.clamp_min(1e-9)
            weighted_hidden = accum / denom.unsqueeze(-1)
            write_denom = denom.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            weighted_key = key_accum / write_denom
            weighted_value = value_accum / write_denom
            token_mask = weighted_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            next_hidden = torch.where(weighted_mask.unsqueeze(-1), weighted_hidden, next_hidden)
            write_key = torch.where(token_mask, weighted_key, write_key)
            write_value = torch.where(token_mask, weighted_value, write_value)
            write_valid = write_valid | weighted_mask.unsqueeze(1)
    return next_hidden, write_key, write_value, write_valid


def test_pure_factorized_attention_global_selected_matches_full_sequence_reference() -> None:
    torch.manual_seed(43)
    model = BrianRouteCore(_pure_factorized_cfg()).eval()
    hidden = torch.randn(2, 6, 64)
    position = _position(model, 2, 6)
    state = _pure_factorized_state(model, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, model.out_action, 0, 1],
            [2, 0, model.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = _pure_factorized_full_sequence_reference(
            model, hidden, position, selected, top_actions, top_weights, use_weighted_fusion, state
        )
        actual = model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)


def test_pure_factorized_attention_global_selected_matches_full_sequence_weighted_fusion() -> None:
    torch.manual_seed(47)
    model = BrianRouteCore(_pure_factorized_cfg()).eval()
    hidden = torch.randn(2, 6, 64)
    position = _position(model, 2, 6)
    state = _pure_factorized_state(model, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    top_actions = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 0], [0, model.out_action], [1, 0], [2, 1]],
            [[2, 1], [0, 2], [1, model.out_action], [2, 0], [0, 1], [1, 2]],
        ],
        dtype=torch.long,
    )
    top_weights = torch.tensor(
        [
            [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [1.0, 0.0], [0.8, 0.2], [0.55, 0.45]],
            [[0.65, 0.35], [0.9, 0.1], [1.0, 0.0], [0.75, 0.25], [0.6, 0.4], [0.5, 0.5]],
        ],
        dtype=hidden.dtype,
    )
    use_weighted_fusion = torch.tensor(
        [
            [True, False, True, True, False, True],
            [True, True, True, False, True, False],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        expected = _pure_factorized_full_sequence_reference(
            model, hidden, position, selected, top_actions, top_weights, use_weighted_fusion, state
        )
        actual = model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)


def test_pure_factorized_attention_global_top1_fast_matches_selected() -> None:
    torch.manual_seed(49)
    selected_model = BrianRouteCore(_pure_factorized_cfg()).eval()
    fast_model = BrianRouteCore(_pure_factorized_cfg(attention_global_route_execution="top1_fast")).eval()
    fast_model.load_state_dict(selected_model.state_dict())
    hidden = torch.randn(2, 6, 64)
    position = _position(selected_model, 2, 6)
    state = _pure_factorized_state(selected_model, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, selected_model.out_action, 0, 1],
            [2, 0, selected_model.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = selected_model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )
        actual = fast_model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )

    assert torch.allclose(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[3], expected[3])
    valid = expected[3].unsqueeze(2).unsqueeze(-1).expand_as(expected[1])
    assert torch.allclose(actual[1][valid], expected[1][valid], atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual[2][valid], expected[2][valid], atol=1e-5, rtol=1e-5)


def test_pure_factorized_attention_global_grouped_selected_matches_selected() -> None:
    torch.manual_seed(51)
    selected_model = BrianRouteCore(_pure_factorized_cfg()).eval()
    grouped_model = BrianRouteCore(_pure_factorized_cfg(attention_global_route_execution="grouped_selected")).eval()
    grouped_model.load_state_dict(selected_model.state_dict())
    hidden = torch.randn(2, 6, 64)
    position = _position(selected_model, 2, 6)
    state = _pure_factorized_state(selected_model, 2, 6)
    selected = torch.tensor(
        [
            [0, 1, 2, selected_model.out_action, 0, 1],
            [2, 0, selected_model.out_action, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_actions = selected.unsqueeze(-1)
    top_weights = torch.ones(*selected.shape, 1)
    use_weighted_fusion = torch.zeros_like(selected, dtype=torch.bool)

    with torch.no_grad():
        expected = selected_model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )
        actual = grouped_model._apply_routed_blocks_with_attention_global(
            hidden,
            position,
            selected,
            top_actions,
            top_weights,
            use_weighted_fusion,
            state,
            _attention_global_route_info(),
        )

    assert torch.allclose(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[3], expected[3])
    valid = expected[3].unsqueeze(2).unsqueeze(-1).expand_as(expected[1])
    assert torch.allclose(actual[1][valid], expected[1][valid], atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual[2][valid], expected[2][valid], atol=1e-5, rtol=1e-5)


def test_summarize_routing_false_keeps_loss_fields_without_diagnostics() -> None:
    torch.manual_seed(39)
    cfg = BrianRouteConfig(
        base=BaselineConfig(vocab_size=64, context_length=8, layers=5, d_model=64, n_heads=4),
        pre_blocks=1,
        route_pool_blocks=3,
        post_blocks=1,
        block_position_dim=8,
        max_route_steps=2,
        route_block_execution="grouped_dense",
    )
    model = BrianRouteCore(cfg).train()
    input_ids = torch.randint(0, 64, (2, 8))

    output = model(
        input_ids,
        targets=input_ids,
        route_mode="scheduled",
        pseudo_policy="sequential",
        summarize_routing=False,
    )
    output["loss"].backward()

    route_info = output["route_info"]
    assert "routing_summary" not in output
    assert len(route_info["route_logits"]) == cfg.max_route_steps
    assert len(route_info["route_probs"]) == cfg.max_route_steps
    assert len(route_info["selected_actions"]) == cfg.max_route_steps
    assert len(route_info["location_distance"]) == cfg.max_route_steps
    assert route_info["topk_actions"] == []
    assert route_info["topk_weights"] == []
    assert route_info["position_norms"] == []
    components = output["loss_components"]
    for name, value in components.items():
        if name != "lm_loss":
            assert value.item() == 0.0


def test_sparse_route_block_execution_config_stats_and_validation() -> None:
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 24, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "route_block_execution": "sparse",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.route_block_execution == "sparse"
    assert model.model_stats()["route_block_execution"] == "sparse"
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "route_block_execution": "sparse_varlen",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.route_block_execution == "sparse_varlen"
    assert model.model_stats()["route_block_execution"] == "sparse_varlen"
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "route_block_execution": "grouped_dense",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.route_block_execution == "grouped_dense"
    assert model.model_stats()["route_block_execution"] == "grouped_dense"
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "attention_global_kv": True,
            "attention_global_kv_mode": "pure_factorized",
            "attention_global_code_dim": 12,
            "attention_global_sink_slots": 0,
            "attention_global_window_slots": 0,
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.attention_global_kv_mode == "pure_factorized"
    assert model.model_stats()["attention_global_kv_mode"] == "pure_factorized"
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "attention_global_kv": True,
            "attention_global_kv_mode": "pure_factorized",
            "attention_global_code_dim": 12,
            "attention_global_sink_slots": 0,
            "attention_global_window_slots": 0,
            "attention_global_route_execution": "top1_fast",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.attention_global_route_execution == "top1_fast"
    assert model.model_stats()["attention_global_route_execution"] == "top1_fast"
    cfg = BrianRouteConfig.from_dict(
        {
            "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
            "pre_blocks": 1,
            "route_pool_blocks": 2,
            "post_blocks": 1,
            "block_position_dim": 8,
            "max_route_steps": 2,
            "attention_global_kv": True,
            "attention_global_kv_mode": "pure_factorized",
            "attention_global_code_dim": 12,
            "attention_global_sink_slots": 0,
            "attention_global_window_slots": 0,
            "attention_global_route_execution": "grouped_selected",
        }
    )
    model = BrianRouteCore(cfg)

    assert cfg.attention_global_route_execution == "grouped_selected"
    assert model.model_stats()["attention_global_route_execution"] == "grouped_selected"
    with pytest.raises(ValueError, match="route_block_execution"):
        BrianRouteConfig.from_dict(
            {
                "base": {"vocab_size": 64, "context_length": 6, "layers": 4, "d_model": 64, "n_heads": 4},
                "pre_blocks": 1,
                "route_pool_blocks": 2,
                "post_blocks": 1,
                "block_position_dim": 8,
                "max_route_steps": 2,
                "route_block_execution": "fast",
            }
        )
