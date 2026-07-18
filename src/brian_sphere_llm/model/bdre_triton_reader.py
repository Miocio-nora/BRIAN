from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional import guard.
    torch = None

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - Triton is optional outside CUDA training.
    triton = None
    tl = None


BDRE_TRITON_FORWARD_NUM_WARPS = 2
BDRE_TRITON_FORWARD_NUM_STAGES = 2
BDRE_TRITON_DQ_NUM_WARPS = 2
BDRE_TRITON_DQ_NUM_STAGES = 3
BDRE_TRITON_DKV_NUM_WARPS = 2
BDRE_TRITON_DKV_NUM_STAGES = 2
BDRE_TRITON_DKV_BLOCK_M = 32
BDRE_TRITON_DKV_BLOCK_N = 64
BDRE_TRITON_DKV_NUM_CTAS = 1


@dataclass(frozen=True)
class BDRETritonReaderMetadata:
    group: torch.Tensor
    query_start: torch.Tensor
    query_count: torch.Tensor
    group_query_start: torch.Tensor
    group_query_count: torch.Tensor
    reader_query_start: torch.Tensor
    reader_query_count: torch.Tensor


def triton_reader_available() -> bool:
    return torch is not None and triton is not None


def build_compact_reader_metadata(
    *,
    sorted_actions: torch.Tensor,
    sorted_indexes: torch.Tensor,
    valid: torch.Tensor,
    batch: int,
    chunk: int,
    readers: int,
    block_m: int,
) -> BDRETritonReaderMetadata:
    """Build fixed-capacity compact query-block metadata without a host sync."""

    if sorted_actions.ndim != 1 or sorted_indexes.ndim != 1 or valid.ndim != 1:
        raise ValueError("Compact reader dispatch tensors must be one-dimensional.")
    if not (sorted_actions.numel() == sorted_indexes.numel() == valid.numel()):
        raise ValueError("Compact reader dispatch tensors must have equal lengths.")
    if block_m < 1:
        raise ValueError("block_m must be positive.")

    device = sorted_actions.device
    groups = readers * batch
    sorted_batches = torch.div(sorted_indexes, chunk, rounding_mode="floor")
    reader_batches = sorted_actions * batch + sorted_batches
    counts = torch.zeros(groups, dtype=torch.long, device=device).scatter_add(
        0,
        reader_batches,
        valid.to(dtype=torch.long),
    )
    starts = torch.cumsum(counts, dim=0) - counts
    blocks_per_group = torch.div(counts + block_m - 1, block_m, rounding_mode="floor")

    potential_blocks = (chunk + block_m - 1) // block_m
    group_grid = torch.arange(groups, device=device).view(-1, 1).expand(-1, potential_blocks)
    block_grid = torch.arange(potential_blocks, device=device).view(1, -1).expand(groups, -1)
    block_valid = block_grid < blocks_per_group.view(-1, 1)
    flat_valid = block_valid.reshape(-1)
    compact_rank = torch.cumsum(flat_valid.to(dtype=torch.long), dim=0) - 1

    max_compact_blocks = (sorted_actions.numel() + block_m - 1) // block_m + groups
    sentinel = torch.full_like(compact_rank, max_compact_blocks)
    target = torch.where(flat_valid, compact_rank, sentinel)
    flat_groups = group_grid.reshape(-1)
    flat_blocks = block_grid.reshape(-1)
    flat_starts = starts.index_select(0, flat_groups) + flat_blocks * block_m
    flat_counts = torch.clamp(
        counts.index_select(0, flat_groups) - flat_blocks * block_m,
        min=0,
        max=block_m,
    )

    metadata_size = max_compact_blocks + 1
    compact_group = torch.zeros(metadata_size, dtype=torch.long, device=device).scatter(
        0,
        target,
        flat_groups,
    )
    compact_start = torch.zeros(metadata_size, dtype=torch.long, device=device).scatter(
        0,
        target,
        flat_starts,
    )
    compact_count = torch.zeros(metadata_size, dtype=torch.long, device=device).scatter(
        0,
        target,
        torch.where(flat_valid, flat_counts, torch.zeros_like(flat_counts)),
    )
    reader_counts = counts.view(readers, batch).sum(dim=1)
    reader_starts = torch.cumsum(reader_counts, dim=0) - reader_counts
    return BDRETritonReaderMetadata(
        group=compact_group[:max_compact_blocks],
        query_start=compact_start[:max_compact_blocks],
        query_count=compact_count[:max_compact_blocks],
        group_query_start=starts,
        group_query_count=counts,
        reader_query_start=reader_starts,
        reader_query_count=reader_counts,
    )


if triton is not None:

    @triton.jit
    def _bdre_compact_reader_forward_kernel(
        query_ptr,
        key_code_ptr,
        value_code_ptr,
        key_read_ptr,
        value_read_ptr,
        cosine_ptr,
        sine_ptr,
        sorted_index_ptr,
        metadata_group_ptr,
        metadata_start_ptr,
        metadata_count_ptr,
        output_ptr,
        latent_ptr,
        lse_ptr,
        query_stride_n: tl.constexpr,
        query_stride_h: tl.constexpr,
        key_code_stride_g: tl.constexpr,
        key_code_stride_k: tl.constexpr,
        key_code_stride_h: tl.constexpr,
        value_code_stride_g: tl.constexpr,
        value_code_stride_k: tl.constexpr,
        value_code_stride_h: tl.constexpr,
        key_read_stride_r: tl.constexpr,
        key_read_stride_h: tl.constexpr,
        key_read_stride_k: tl.constexpr,
        value_read_stride_r: tl.constexpr,
        value_read_stride_h: tl.constexpr,
        value_read_stride_v: tl.constexpr,
        output_stride_n: tl.constexpr,
        output_stride_h: tl.constexpr,
        latent_stride_n: tl.constexpr,
        latent_stride_h: tl.constexpr,
        lse_stride_n: tl.constexpr,
        lse_stride_h: tl.constexpr,
        BATCH: tl.constexpr,
        CHUNK: tl.constexpr,
        START_POSITION: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        head_index = tl.program_id(1)
        group_index = tl.load(metadata_group_ptr + block_index)
        query_start = tl.load(metadata_start_ptr + block_index)
        query_count = tl.load(metadata_count_ptr + block_index)
        reader_index = group_index // BATCH

        offsets_m = tl.arange(0, BLOCK_M)
        offsets_half = tl.arange(0, HEAD_DIM // 2)
        offsets_key_dim = tl.arange(0, KEY_DIM)
        offsets_value_dim = tl.arange(0, VALUE_DIM)
        query_rows = query_start + offsets_m
        query_mask = offsets_m < query_count
        token_indexes = tl.load(sorted_index_ptr + query_rows, mask=query_mask, other=0)
        query_positions = START_POSITION + token_indexes % CHUNK

        query_base = query_ptr + query_rows[:, None] * query_stride_n
        query_base += head_index * query_stride_h
        query_first = tl.load(
            query_base + offsets_half[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_second = tl.load(
            query_base + (offsets_half + HEAD_DIM // 2)[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_cosine = tl.load(
            cosine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=query_mask[:, None],
            other=1.0,
        )
        query_sine = tl.load(
            sine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_rotary_first = (query_first * query_cosine - query_second * query_sine).to(
            tl.bfloat16
        )
        query_rotary_second = (query_second * query_cosine + query_first * query_sine).to(
            tl.bfloat16
        )

        key_read_base = key_read_ptr + reader_index * key_read_stride_r
        key_read_base += head_index * key_read_stride_h
        key_read_first = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + offsets_half[None, :]
        )
        key_read_second = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + (offsets_half + HEAD_DIM // 2)[None, :]
        )

        running_max = tl.where(query_mask, -float("inf"), 0.0)
        running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, VALUE_DIM), dtype=tl.float32)

        for key_start in range(0, KEY_LENGTH, BLOCK_N):
            offsets_n = key_start + tl.arange(0, BLOCK_N)
            key_mask = offsets_n < KEY_LENGTH
            key_codes = tl.load(
                key_code_ptr
                + group_index * key_code_stride_g
                + offsets_n[:, None] * key_code_stride_k
                + head_index * key_code_stride_h
                + offsets_key_dim[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            decoded_first = tl.dot(key_codes, key_read_first)
            decoded_second = tl.dot(key_codes, key_read_second)
            key_cosine = tl.load(
                cosine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=key_mask[:, None],
                other=1.0,
            )
            key_sine = tl.load(
                sine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            key_rotary_first = (
                decoded_first.to(tl.bfloat16) * key_cosine
                - decoded_second.to(tl.bfloat16) * key_sine
            ).to(tl.bfloat16)
            key_rotary_second = (
                decoded_second.to(tl.bfloat16) * key_cosine
                + decoded_first.to(tl.bfloat16) * key_sine
            ).to(tl.bfloat16)
            scores = tl.dot(query_rotary_first, tl.trans(key_rotary_first))
            scores += tl.dot(query_rotary_second, tl.trans(key_rotary_second))
            scores *= SCALE
            causal_mask = offsets_n[None, :] <= query_positions[:, None]
            score_mask = query_mask[:, None] & key_mask[None, :] & causal_mask
            scores = tl.where(score_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            new_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
            value_codes = tl.load(
                value_code_ptr
                + group_index * value_code_stride_g
                + offsets_n[:, None] * value_code_stride_k
                + head_index * value_code_stride_h
                + offsets_value_dim[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator *= previous_scale[:, None]
            accumulator += tl.dot(probabilities.to(tl.bfloat16), value_codes)
            running_max = new_max
            running_sum = new_sum

        safe_sum = tl.maximum(running_sum, 1.0)
        latent_value = accumulator / safe_sum[:, None]
        latent_bf16 = latent_value.to(tl.bfloat16)
        value_read = tl.load(
            value_read_ptr
            + reader_index * value_read_stride_r
            + head_index * value_read_stride_h
            + offsets_value_dim[:, None] * value_read_stride_v
            + tl.arange(0, HEAD_DIM)[None, :]
        )
        output = tl.dot(latent_bf16, value_read)

        output_base = output_ptr + query_rows[:, None] * output_stride_n
        output_base += head_index * output_stride_h
        tl.store(
            output_base + tl.arange(0, HEAD_DIM)[None, :],
            output,
            mask=query_mask[:, None],
        )
        latent_base = latent_ptr + query_rows[:, None] * latent_stride_n
        latent_base += head_index * latent_stride_h
        tl.store(
            latent_base + offsets_value_dim[None, :],
            latent_bf16,
            mask=query_mask[:, None],
        )
        tl.store(
            lse_ptr + query_rows * lse_stride_n + head_index * lse_stride_h,
            running_max + tl.log(safe_sum),
            mask=query_mask,
        )

    @triton.jit
    def _bdre_value_read_backward_input_kernel(
        grad_output_ptr,
        latent_ptr,
        value_read_ptr,
        metadata_group_ptr,
        metadata_start_ptr,
        metadata_count_ptr,
        grad_latent_ptr,
        delta_ptr,
        grad_output_stride_n: tl.constexpr,
        grad_output_stride_h: tl.constexpr,
        latent_stride_n: tl.constexpr,
        latent_stride_h: tl.constexpr,
        value_read_stride_r: tl.constexpr,
        value_read_stride_h: tl.constexpr,
        value_read_stride_v: tl.constexpr,
        grad_latent_stride_n: tl.constexpr,
        grad_latent_stride_h: tl.constexpr,
        delta_stride_n: tl.constexpr,
        delta_stride_h: tl.constexpr,
        BATCH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        head_index = tl.program_id(1)
        group_index = tl.load(metadata_group_ptr + block_index)
        query_start = tl.load(metadata_start_ptr + block_index)
        query_count = tl.load(metadata_count_ptr + block_index)
        reader_index = group_index // BATCH
        offsets_m = tl.arange(0, BLOCK_M)
        offsets_d = tl.arange(0, HEAD_DIM)
        offsets_v = tl.arange(0, VALUE_DIM)
        rows = query_start + offsets_m
        row_mask = offsets_m < query_count

        grad_output = tl.load(
            grad_output_ptr
            + rows[:, None] * grad_output_stride_n
            + head_index * grad_output_stride_h
            + offsets_d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        value_read = tl.load(
            value_read_ptr
            + reader_index * value_read_stride_r
            + head_index * value_read_stride_h
            + offsets_v[:, None] * value_read_stride_v
            + offsets_d[None, :]
        )
        grad_latent = tl.dot(grad_output, tl.trans(value_read)).to(tl.bfloat16)
        latent = tl.load(
            latent_ptr
            + rows[:, None] * latent_stride_n
            + head_index * latent_stride_h
            + offsets_v[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        delta = tl.sum(grad_latent.to(tl.float32) * latent.to(tl.float32), axis=1)
        tl.store(
            grad_latent_ptr
            + rows[:, None] * grad_latent_stride_n
            + head_index * grad_latent_stride_h
            + offsets_v[None, :],
            grad_latent,
            mask=row_mask[:, None],
        )
        tl.store(
            delta_ptr + rows * delta_stride_n + head_index * delta_stride_h,
            delta,
            mask=row_mask,
        )

    @triton.jit
    def _bdre_value_read_backward_weight_kernel(
        grad_output_ptr,
        latent_ptr,
        reader_start_ptr,
        reader_count_ptr,
        grad_value_read_ptr,
        grad_output_stride_n: tl.constexpr,
        grad_output_stride_h: tl.constexpr,
        latent_stride_n: tl.constexpr,
        latent_stride_h: tl.constexpr,
        grad_value_read_stride_r: tl.constexpr,
        grad_value_read_stride_h: tl.constexpr,
        grad_value_read_stride_v: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        reader_index = tl.program_id(0)
        head_index = tl.program_id(1)
        tile_index = tl.program_id(2)
        value_tiles = tl.cdiv(VALUE_DIM, BLOCK_V)
        value_tile = tile_index % value_tiles
        dim_tile = tile_index // value_tiles
        offsets_v = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
        offsets_d = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        query_start = tl.load(reader_start_ptr + reader_index)
        query_count = tl.load(reader_count_ptr + reader_index)
        accumulator = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

        for query_offset in tl.range(0, query_count, BLOCK_Q):
            offsets_q = query_offset + tl.arange(0, BLOCK_Q)
            rows = query_start + offsets_q
            query_mask = offsets_q < query_count
            latent = tl.load(
                latent_ptr
                + rows[None, :] * latent_stride_n
                + head_index * latent_stride_h
                + offsets_v[:, None],
                mask=(offsets_v[:, None] < VALUE_DIM) & query_mask[None, :],
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr
                + rows[:, None] * grad_output_stride_n
                + head_index * grad_output_stride_h
                + offsets_d[None, :],
                mask=query_mask[:, None] & (offsets_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            accumulator += tl.dot(latent, grad_output)

        tl.store(
            grad_value_read_ptr
            + reader_index * grad_value_read_stride_r
            + head_index * grad_value_read_stride_h
            + offsets_v[:, None] * grad_value_read_stride_v
            + offsets_d[None, :],
            accumulator,
            mask=(offsets_v[:, None] < VALUE_DIM) & (offsets_d[None, :] < HEAD_DIM),
        )

    @triton.jit
    def _bdre_attention_backward_query_kernel(
        query_ptr,
        key_code_ptr,
        value_code_ptr,
        key_read_ptr,
        cosine_ptr,
        sine_ptr,
        sorted_index_ptr,
        metadata_group_ptr,
        metadata_start_ptr,
        metadata_count_ptr,
        grad_latent_ptr,
        delta_ptr,
        lse_ptr,
        grad_query_ptr,
        query_stride_n: tl.constexpr,
        query_stride_h: tl.constexpr,
        key_code_stride_g: tl.constexpr,
        key_code_stride_k: tl.constexpr,
        key_code_stride_h: tl.constexpr,
        value_code_stride_g: tl.constexpr,
        value_code_stride_k: tl.constexpr,
        value_code_stride_h: tl.constexpr,
        key_read_stride_r: tl.constexpr,
        key_read_stride_h: tl.constexpr,
        key_read_stride_k: tl.constexpr,
        grad_latent_stride_n: tl.constexpr,
        grad_latent_stride_h: tl.constexpr,
        delta_stride_n: tl.constexpr,
        delta_stride_h: tl.constexpr,
        lse_stride_n: tl.constexpr,
        lse_stride_h: tl.constexpr,
        grad_query_stride_n: tl.constexpr,
        grad_query_stride_h: tl.constexpr,
        BATCH: tl.constexpr,
        CHUNK: tl.constexpr,
        START_POSITION: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        block_index = tl.program_id(0)
        head_index = tl.program_id(1)
        group_index = tl.load(metadata_group_ptr + block_index)
        query_start = tl.load(metadata_start_ptr + block_index)
        query_count = tl.load(metadata_count_ptr + block_index)
        reader_index = group_index // BATCH
        offsets_m = tl.arange(0, BLOCK_M)
        offsets_half = tl.arange(0, HEAD_DIM // 2)
        offsets_key_dim = tl.arange(0, KEY_DIM)
        offsets_value_dim = tl.arange(0, VALUE_DIM)
        rows = query_start + offsets_m
        query_mask = offsets_m < query_count
        token_indexes = tl.load(sorted_index_ptr + rows, mask=query_mask, other=0)
        query_positions = START_POSITION + token_indexes % CHUNK

        query_base = query_ptr + rows[:, None] * query_stride_n + head_index * query_stride_h
        query_first = tl.load(
            query_base + offsets_half[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_second = tl.load(
            query_base + (offsets_half + HEAD_DIM // 2)[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_cosine = tl.load(
            cosine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=query_mask[:, None],
            other=1.0,
        )
        query_sine = tl.load(
            sine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        query_rotary_first = (query_first * query_cosine - query_second * query_sine).to(
            tl.bfloat16
        )
        query_rotary_second = (query_second * query_cosine + query_first * query_sine).to(
            tl.bfloat16
        )
        grad_latent = tl.load(
            grad_latent_ptr
            + rows[:, None] * grad_latent_stride_n
            + head_index * grad_latent_stride_h
            + offsets_value_dim[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + rows * delta_stride_n + head_index * delta_stride_h,
            mask=query_mask,
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + rows * lse_stride_n + head_index * lse_stride_h,
            mask=query_mask,
            other=0.0,
        )
        key_read_base = key_read_ptr + reader_index * key_read_stride_r
        key_read_base += head_index * key_read_stride_h
        key_read_first = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + offsets_half[None, :]
        )
        key_read_second = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + (offsets_half + HEAD_DIM // 2)[None, :]
        )
        grad_query_rotary_first = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
        grad_query_rotary_second = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)

        for key_start in range(0, KEY_LENGTH, BLOCK_N):
            offsets_n = key_start + tl.arange(0, BLOCK_N)
            key_mask = offsets_n < KEY_LENGTH
            key_codes = tl.load(
                key_code_ptr
                + group_index * key_code_stride_g
                + offsets_n[:, None] * key_code_stride_k
                + head_index * key_code_stride_h
                + offsets_key_dim[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            decoded_first = tl.dot(key_codes, key_read_first)
            decoded_second = tl.dot(key_codes, key_read_second)
            key_cosine = tl.load(
                cosine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=key_mask[:, None],
                other=1.0,
            )
            key_sine = tl.load(
                sine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            key_rotary_first = (
                decoded_first.to(tl.bfloat16) * key_cosine
                - decoded_second.to(tl.bfloat16) * key_sine
            ).to(tl.bfloat16)
            key_rotary_second = (
                decoded_second.to(tl.bfloat16) * key_cosine
                + decoded_first.to(tl.bfloat16) * key_sine
            ).to(tl.bfloat16)
            scores = tl.dot(query_rotary_first, tl.trans(key_rotary_first))
            scores += tl.dot(query_rotary_second, tl.trans(key_rotary_second))
            scores *= SCALE
            score_mask = (
                query_mask[:, None]
                & key_mask[None, :]
                & (offsets_n[None, :] <= query_positions[:, None])
            )
            scores = tl.where(score_mask, scores, -float("inf"))
            probabilities = tl.exp(scores - lse[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            value_codes = tl.load(
                value_code_ptr
                + group_index * value_code_stride_g
                + offsets_n[:, None] * value_code_stride_k
                + head_index * value_code_stride_h
                + offsets_value_dim[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            grad_probability = tl.dot(grad_latent, tl.trans(value_codes))
            grad_score = probabilities * (grad_probability - delta[:, None]) * SCALE
            grad_score = grad_score.to(tl.bfloat16)
            grad_query_rotary_first += tl.dot(grad_score, key_rotary_first)
            grad_query_rotary_second += tl.dot(grad_score, key_rotary_second)

        grad_query_first = (
            grad_query_rotary_first * query_cosine
            + grad_query_rotary_second * query_sine
        )
        grad_query_second = (
            -grad_query_rotary_first * query_sine
            + grad_query_rotary_second * query_cosine
        )
        grad_query_base = (
            grad_query_ptr + rows[:, None] * grad_query_stride_n + head_index * grad_query_stride_h
        )
        tl.store(
            grad_query_base + offsets_half[None, :],
            grad_query_first,
            mask=query_mask[:, None],
        )
        tl.store(
            grad_query_base + (offsets_half + HEAD_DIM // 2)[None, :],
            grad_query_second,
            mask=query_mask[:, None],
        )

    @triton.jit
    def _bdre_attention_backward_key_value_kernel(
        query_ptr,
        key_code_ptr,
        value_code_ptr,
        key_read_ptr,
        cosine_ptr,
        sine_ptr,
        sorted_index_ptr,
        group_start_ptr,
        group_count_ptr,
        grad_latent_ptr,
        delta_ptr,
        lse_ptr,
        grad_decoded_key_ptr,
        grad_value_head_ptr,
        query_stride_n: tl.constexpr,
        query_stride_h: tl.constexpr,
        key_code_stride_g: tl.constexpr,
        key_code_stride_k: tl.constexpr,
        key_code_stride_h: tl.constexpr,
        value_code_stride_g: tl.constexpr,
        value_code_stride_k: tl.constexpr,
        value_code_stride_h: tl.constexpr,
        key_read_stride_r: tl.constexpr,
        key_read_stride_h: tl.constexpr,
        key_read_stride_k: tl.constexpr,
        grad_latent_stride_n: tl.constexpr,
        grad_latent_stride_h: tl.constexpr,
        delta_stride_n: tl.constexpr,
        delta_stride_h: tl.constexpr,
        lse_stride_n: tl.constexpr,
        lse_stride_h: tl.constexpr,
        grad_decoded_key_stride_g: tl.constexpr,
        grad_decoded_key_stride_h: tl.constexpr,
        grad_decoded_key_stride_k: tl.constexpr,
        grad_value_head_stride_g: tl.constexpr,
        grad_value_head_stride_h: tl.constexpr,
        grad_value_head_stride_k: tl.constexpr,
        BATCH: tl.constexpr,
        CHUNK: tl.constexpr,
        START_POSITION: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        group_index = tl.program_id(0)
        head_index = tl.program_id(1)
        key_block = tl.program_id(2)
        reader_index = group_index // BATCH
        offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        key_mask = offsets_n < KEY_LENGTH
        offsets_half = tl.arange(0, HEAD_DIM // 2)
        offsets_key_dim = tl.arange(0, KEY_DIM)
        offsets_value_dim = tl.arange(0, VALUE_DIM)

        key_codes = tl.load(
            key_code_ptr
            + group_index * key_code_stride_g
            + offsets_n[:, None] * key_code_stride_k
            + head_index * key_code_stride_h
            + offsets_key_dim[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        key_read_base = key_read_ptr + reader_index * key_read_stride_r
        key_read_base += head_index * key_read_stride_h
        key_read_first = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + offsets_half[None, :]
        )
        key_read_second = tl.load(
            key_read_base
            + offsets_key_dim[:, None] * key_read_stride_k
            + (offsets_half + HEAD_DIM // 2)[None, :]
        )
        decoded_first = tl.dot(key_codes, key_read_first)
        decoded_second = tl.dot(key_codes, key_read_second)
        key_cosine = tl.load(
            cosine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=key_mask[:, None],
            other=1.0,
        )
        key_sine = tl.load(
            sine_ptr + offsets_n[:, None] * HEAD_DIM + offsets_half[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        key_rotary_first = (
            decoded_first.to(tl.bfloat16) * key_cosine
            - decoded_second.to(tl.bfloat16) * key_sine
        ).to(tl.bfloat16)
        key_rotary_second = (
            decoded_second.to(tl.bfloat16) * key_cosine
            + decoded_first.to(tl.bfloat16) * key_sine
        ).to(tl.bfloat16)
        value_codes = tl.load(
            value_code_ptr
            + group_index * value_code_stride_g
            + offsets_n[:, None] * value_code_stride_k
            + head_index * value_code_stride_h
            + offsets_value_dim[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        grad_key_rotary_first = tl.zeros((BLOCK_N, HEAD_DIM // 2), dtype=tl.float32)
        grad_key_rotary_second = tl.zeros((BLOCK_N, HEAD_DIM // 2), dtype=tl.float32)
        grad_value = tl.zeros((BLOCK_N, VALUE_DIM), dtype=tl.float32)
        query_start = tl.load(group_start_ptr + group_index)
        query_count = tl.load(group_count_ptr + group_index)

        for query_offset in tl.range(0, query_count, BLOCK_M):
            offsets_m = query_offset + tl.arange(0, BLOCK_M)
            rows = query_start + offsets_m
            query_mask = offsets_m < query_count
            token_indexes = tl.load(sorted_index_ptr + rows, mask=query_mask, other=0)
            query_positions = START_POSITION + token_indexes % CHUNK
            query_base = query_ptr + rows[:, None] * query_stride_n
            query_base += head_index * query_stride_h
            query_first = tl.load(
                query_base + offsets_half[None, :],
                mask=query_mask[:, None],
                other=0.0,
            )
            query_second = tl.load(
                query_base + (offsets_half + HEAD_DIM // 2)[None, :],
                mask=query_mask[:, None],
                other=0.0,
            )
            query_cosine = tl.load(
                cosine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=query_mask[:, None],
                other=1.0,
            )
            query_sine = tl.load(
                sine_ptr + query_positions[:, None] * HEAD_DIM + offsets_half[None, :],
                mask=query_mask[:, None],
                other=0.0,
            )
            query_rotary_first = (
                query_first * query_cosine - query_second * query_sine
            ).to(tl.bfloat16)
            query_rotary_second = (
                query_second * query_cosine + query_first * query_sine
            ).to(tl.bfloat16)
            scores = tl.dot(query_rotary_first, tl.trans(key_rotary_first))
            scores += tl.dot(query_rotary_second, tl.trans(key_rotary_second))
            scores *= SCALE
            score_mask = (
                query_mask[:, None]
                & key_mask[None, :]
                & (offsets_n[None, :] <= query_positions[:, None])
            )
            scores = tl.where(score_mask, scores, -float("inf"))
            lse = tl.load(
                lse_ptr + rows * lse_stride_n + head_index * lse_stride_h,
                mask=query_mask,
                other=0.0,
            )
            probabilities = tl.exp(scores - lse[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            grad_latent = tl.load(
                grad_latent_ptr
                + rows[:, None] * grad_latent_stride_n
                + head_index * grad_latent_stride_h
                + offsets_value_dim[None, :],
                mask=query_mask[:, None],
                other=0.0,
            )
            delta = tl.load(
                delta_ptr + rows * delta_stride_n + head_index * delta_stride_h,
                mask=query_mask,
                other=0.0,
            )
            grad_probability = tl.dot(grad_latent, tl.trans(value_codes))
            grad_score = probabilities * (grad_probability - delta[:, None]) * SCALE
            grad_score_bf16 = grad_score.to(tl.bfloat16)
            grad_key_rotary_first += tl.dot(tl.trans(grad_score_bf16), query_rotary_first)
            grad_key_rotary_second += tl.dot(tl.trans(grad_score_bf16), query_rotary_second)
            grad_value += tl.dot(tl.trans(probabilities.to(tl.bfloat16)), grad_latent)

        grad_decoded_first = (
            grad_key_rotary_first * key_cosine + grad_key_rotary_second * key_sine
        )
        grad_decoded_second = (
            -grad_key_rotary_first * key_sine + grad_key_rotary_second * key_cosine
        )
        grad_decoded_base = (
            grad_decoded_key_ptr
            + group_index * grad_decoded_key_stride_g
            + head_index * grad_decoded_key_stride_h
            + offsets_n[:, None] * grad_decoded_key_stride_k
        )
        tl.store(
            grad_decoded_base + offsets_half[None, :],
            grad_decoded_first,
            mask=key_mask[:, None],
        )
        tl.store(
            grad_decoded_base + (offsets_half + HEAD_DIM // 2)[None, :],
            grad_decoded_second,
            mask=key_mask[:, None],
        )
        grad_value_base = (
            grad_value_head_ptr
            + group_index * grad_value_head_stride_g
            + head_index * grad_value_head_stride_h
            + offsets_n[:, None] * grad_value_head_stride_k
        )
        tl.store(
            grad_value_base + offsets_value_dim[None, :],
            grad_value,
            mask=key_mask[:, None],
        )

    @triton.jit
    def _bdre_key_code_backward_kernel(
        grad_decoded_ptr,
        key_read_ptr,
        grad_key_code_ptr,
        grad_decoded_stride_g: tl.constexpr,
        grad_decoded_stride_h: tl.constexpr,
        grad_decoded_stride_k: tl.constexpr,
        key_read_stride_r: tl.constexpr,
        key_read_stride_h: tl.constexpr,
        key_read_stride_k: tl.constexpr,
        grad_key_code_stride_g: tl.constexpr,
        grad_key_code_stride_k: tl.constexpr,
        BATCH: tl.constexpr,
        HEADS: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        group_index = tl.program_id(0)
        key_block = tl.program_id(1)
        reader_index = group_index // BATCH
        offsets_k = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        offsets_d = tl.arange(0, HEAD_DIM)
        offsets_r = tl.arange(0, KEY_DIM)
        key_mask = offsets_k < KEY_LENGTH
        accumulator = tl.zeros((BLOCK_K, KEY_DIM), dtype=tl.float32)

        for head_index in range(HEADS):
            grad_decoded = tl.load(
                grad_decoded_ptr
                + group_index * grad_decoded_stride_g
                + head_index * grad_decoded_stride_h
                + offsets_k[:, None] * grad_decoded_stride_k
                + offsets_d[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            key_read = tl.load(
                key_read_ptr
                + reader_index * key_read_stride_r
                + head_index * key_read_stride_h
                + offsets_r[:, None] * key_read_stride_k
                + offsets_d[None, :]
            )
            accumulator += tl.dot(grad_decoded, tl.trans(key_read))
        tl.store(
            grad_key_code_ptr
            + group_index * grad_key_code_stride_g
            + offsets_k[:, None] * grad_key_code_stride_k
            + offsets_r[None, :],
            accumulator,
            mask=key_mask[:, None],
        )

    @triton.jit
    def _bdre_per_head_key_code_backward_kernel(
        grad_decoded_ptr,
        key_read_ptr,
        grad_key_code_ptr,
        grad_decoded_stride_g: tl.constexpr,
        grad_decoded_stride_h: tl.constexpr,
        grad_decoded_stride_k: tl.constexpr,
        key_read_stride_r: tl.constexpr,
        key_read_stride_h: tl.constexpr,
        key_read_stride_k: tl.constexpr,
        grad_key_code_stride_g: tl.constexpr,
        grad_key_code_stride_k: tl.constexpr,
        grad_key_code_stride_h: tl.constexpr,
        BATCH: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        group_index = tl.program_id(0)
        head_index = tl.program_id(1)
        key_block = tl.program_id(2)
        reader_index = group_index // BATCH
        offsets_k = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        offsets_d = tl.arange(0, HEAD_DIM)
        offsets_r = tl.arange(0, KEY_DIM)
        key_mask = offsets_k < KEY_LENGTH
        grad_decoded = tl.load(
            grad_decoded_ptr
            + group_index * grad_decoded_stride_g
            + head_index * grad_decoded_stride_h
            + offsets_k[:, None] * grad_decoded_stride_k
            + offsets_d[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        key_read = tl.load(
            key_read_ptr
            + reader_index * key_read_stride_r
            + head_index * key_read_stride_h
            + offsets_r[:, None] * key_read_stride_k
            + offsets_d[None, :]
        )
        grad_key_code = tl.dot(grad_decoded, tl.trans(key_read))
        tl.store(
            grad_key_code_ptr
            + group_index * grad_key_code_stride_g
            + offsets_k[:, None] * grad_key_code_stride_k
            + head_index * grad_key_code_stride_h
            + offsets_r[None, :],
            grad_key_code,
            mask=key_mask[:, None],
        )

    @triton.jit
    def _bdre_key_read_backward_kernel(
        key_code_ptr,
        grad_decoded_ptr,
        grad_key_read_ptr,
        key_code_stride_g: tl.constexpr,
        key_code_stride_k: tl.constexpr,
        key_code_stride_h: tl.constexpr,
        grad_decoded_stride_g: tl.constexpr,
        grad_decoded_stride_h: tl.constexpr,
        grad_decoded_stride_k: tl.constexpr,
        grad_key_read_stride_r: tl.constexpr,
        grad_key_read_stride_h: tl.constexpr,
        grad_key_read_stride_k: tl.constexpr,
        BATCH: tl.constexpr,
        KEY_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEY_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        reader_index = tl.program_id(0)
        head_index = tl.program_id(1)
        tile_index = tl.program_id(2)
        key_tiles = tl.cdiv(KEY_DIM, BLOCK_R)
        key_tile = tile_index % key_tiles
        dim_tile = tile_index // key_tiles
        offsets_r = key_tile * BLOCK_R + tl.arange(0, BLOCK_R)
        offsets_d = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        accumulator = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        for batch_index in range(BATCH):
            group_index = reader_index * BATCH + batch_index
            for key_start in range(0, KEY_LENGTH, BLOCK_K):
                offsets_k = key_start + tl.arange(0, BLOCK_K)
                key_mask = offsets_k < KEY_LENGTH
                key_codes = tl.load(
                    key_code_ptr
                    + group_index * key_code_stride_g
                    + offsets_k[None, :] * key_code_stride_k
                    + head_index * key_code_stride_h
                    + offsets_r[:, None],
                    mask=key_mask[None, :] & (offsets_r[:, None] < KEY_DIM),
                    other=0.0,
                )
                grad_decoded = tl.load(
                    grad_decoded_ptr
                    + group_index * grad_decoded_stride_g
                    + head_index * grad_decoded_stride_h
                    + offsets_k[:, None] * grad_decoded_stride_k
                    + offsets_d[None, :],
                    mask=key_mask[:, None] & (offsets_d[None, :] < HEAD_DIM),
                    other=0.0,
                )
                accumulator += tl.dot(key_codes, grad_decoded)
        tl.store(
            grad_key_read_ptr
            + reader_index * grad_key_read_stride_r
            + head_index * grad_key_read_stride_h
            + offsets_r[:, None] * grad_key_read_stride_k
            + offsets_d[None, :],
            accumulator,
            mask=(offsets_r[:, None] < KEY_DIM) & (offsets_d[None, :] < HEAD_DIM),
        )


def triton_compact_reader_forward(
    query: torch.Tensor,
    key_codes: torch.Tensor,
    value_codes: torch.Tensor,
    key_read: torch.Tensor,
    value_read: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    sorted_indexes: torch.Tensor,
    metadata: BDRETritonReaderMetadata,
    *,
    batch: int,
    chunk: int,
    start_position: int,
    block_m: int = 32,
    block_n: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused compact RC-KV reader forward kernel."""

    if not triton_reader_available():
        raise RuntimeError("The Triton RC-KV reader requires PyTorch and Triton.")
    if not query.is_cuda or query.dtype != torch.bfloat16:
        raise ValueError("The Triton RC-KV reader currently requires CUDA BF16 queries.")
    if (
        query.ndim != 3
        or key_codes.ndim not in {4, 5}
        or value_codes.ndim != key_codes.ndim
    ):
        raise ValueError("Unexpected compact reader tensor rank.")
    readers, cache_batch, key_length = key_codes.shape[:3]
    per_head = key_codes.ndim == 5
    heads, head_dim = query.shape[1:]
    if cache_batch != batch or value_codes.shape[:3] != key_codes.shape[:3]:
        raise ValueError("Canonical Key/Value cache shapes do not match.")
    if per_head:
        if key_codes.size(3) != heads or value_codes.size(3) != heads:
            raise ValueError("Strict per-head caches must match the query head count.")
        if value_codes.shape[:4] != key_codes.shape[:4]:
            raise ValueError("Per-head canonical Key/Value cache shapes do not match.")
    key_dim = key_codes.size(-1)
    value_dim = value_codes.size(-1)
    if key_read.shape != (readers, heads, key_dim, head_dim):
        raise ValueError("Unexpected Key reader shape.")
    if value_read.shape != (readers, heads, value_dim, head_dim):
        raise ValueError("Unexpected Value reader shape.")
    if head_dim % 32 or key_dim not in {16, 32, 64} or value_dim not in {16, 32, 64}:
        raise ValueError(
            "The Triton reader requires head dimensions divisible by 32 and cache "
            "dimensions in {16,32,64}."
        )
    if block_m not in {16, 32, 64, 128} or block_n not in {16, 32, 64}:
        raise ValueError("Unsupported Triton reader block size.")

    if per_head:
        compact_key_codes = key_codes.contiguous().view(
            readers * batch,
            key_length,
            heads,
            key_dim,
        )
        compact_value_codes = value_codes.contiguous().view(
            readers * batch,
            key_length,
            heads,
            value_dim,
        )
        key_code_head_stride = compact_key_codes.stride(2)
        value_code_head_stride = compact_value_codes.stride(2)
    else:
        compact_key_codes = key_codes.contiguous().view(readers * batch, key_length, key_dim)
        compact_value_codes = value_codes.contiguous().view(
            readers * batch,
            key_length,
            value_dim,
        )
        key_code_head_stride = 0
        value_code_head_stride = 0
    cosine = cosine.reshape(-1, head_dim).contiguous()
    sine = sine.reshape(-1, head_dim).contiguous()
    output = torch.zeros_like(query)
    latent = torch.zeros(
        query.size(0),
        heads,
        value_dim,
        dtype=query.dtype,
        device=query.device,
    )
    lse = torch.zeros(
        query.size(0),
        heads,
        dtype=torch.float32,
        device=query.device,
    )
    grid = (metadata.group.numel(), heads)
    _bdre_compact_reader_forward_kernel[grid](
        query,
        compact_key_codes,
        compact_value_codes,
        key_read,
        value_read,
        cosine,
        sine,
        sorted_indexes,
        metadata.group,
        metadata.query_start,
        metadata.query_count,
        output,
        latent,
        lse,
        query.stride(0),
        query.stride(1),
        compact_key_codes.stride(0),
        compact_key_codes.stride(1),
        key_code_head_stride,
        compact_value_codes.stride(0),
        compact_value_codes.stride(1),
        value_code_head_stride,
        key_read.stride(0),
        key_read.stride(1),
        key_read.stride(2),
        value_read.stride(0),
        value_read.stride(1),
        value_read.stride(2),
        output.stride(0),
        output.stride(1),
        latent.stride(0),
        latent.stride(1),
        lse.stride(0),
        lse.stride(1),
        BATCH=batch,
        CHUNK=chunk,
        START_POSITION=start_position,
        KEY_LENGTH=key_length,
        HEAD_DIM=head_dim,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=BDRE_TRITON_FORWARD_NUM_WARPS,
        num_stages=BDRE_TRITON_FORWARD_NUM_STAGES,
    )
    return output, latent, lse


if torch is not None:

    class _BDRETritonCompactReaderFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            query: torch.Tensor,
            key_codes: torch.Tensor,
            value_codes: torch.Tensor,
            key_read: torch.Tensor,
            value_read: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
            sorted_indexes: torch.Tensor,
            metadata_group: torch.Tensor,
            metadata_start: torch.Tensor,
            metadata_count: torch.Tensor,
            group_query_start: torch.Tensor,
            group_query_count: torch.Tensor,
            reader_query_start: torch.Tensor,
            reader_query_count: torch.Tensor,
            batch: int,
            chunk: int,
            start_position: int,
            block_m: int,
            block_n: int,
        ) -> torch.Tensor:
            metadata = BDRETritonReaderMetadata(
                group=metadata_group,
                query_start=metadata_start,
                query_count=metadata_count,
                group_query_start=group_query_start,
                group_query_count=group_query_count,
                reader_query_start=reader_query_start,
                reader_query_count=reader_query_count,
            )
            output, latent, lse = triton_compact_reader_forward(
                query,
                key_codes,
                value_codes,
                key_read,
                value_read,
                cosine,
                sine,
                sorted_indexes,
                metadata,
                batch=batch,
                chunk=chunk,
                start_position=start_position,
                block_m=block_m,
                block_n=block_n,
            )
            ctx.save_for_backward(
                query,
                key_codes,
                value_codes,
                key_read,
                value_read,
                cosine,
                sine,
                sorted_indexes,
                metadata_group,
                metadata_start,
                metadata_count,
                group_query_start,
                group_query_count,
                reader_query_start,
                reader_query_count,
                latent,
                lse,
            )
            ctx.batch = batch
            ctx.chunk = chunk
            ctx.start_position = start_position
            ctx.block_m = block_m
            ctx.block_n = block_n
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            (
                query,
                key_codes,
                value_codes,
                key_read,
                value_read,
                cosine,
                sine,
                sorted_indexes,
                metadata_group,
                metadata_start,
                metadata_count,
                group_query_start,
                group_query_count,
                reader_query_start,
                reader_query_count,
                latent,
                lse,
            ) = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            readers, batch, key_length = key_codes.shape[:3]
            groups = readers * batch
            heads = query.size(1)
            head_dim = query.size(2)
            per_head = key_codes.ndim == 5
            key_dim = key_codes.size(-1)
            value_dim = value_codes.size(-1)
            if per_head:
                compact_key_codes = key_codes.contiguous().view(
                    groups,
                    key_length,
                    heads,
                    key_dim,
                )
                compact_value_codes = value_codes.contiguous().view(
                    groups,
                    key_length,
                    heads,
                    value_dim,
                )
                key_code_head_stride = compact_key_codes.stride(2)
                value_code_head_stride = compact_value_codes.stride(2)
            else:
                compact_key_codes = key_codes.contiguous().view(groups, key_length, key_dim)
                compact_value_codes = value_codes.contiguous().view(
                    groups,
                    key_length,
                    value_dim,
                )
                key_code_head_stride = 0
                value_code_head_stride = 0
            cosine = cosine.reshape(-1, head_dim).contiguous()
            sine = sine.reshape(-1, head_dim).contiguous()

            grad_latent = torch.zeros_like(latent)
            delta = torch.zeros(
                query.size(0),
                heads,
                dtype=torch.float32,
                device=query.device,
            )
            compact_grid = (metadata_group.numel(), heads)
            _bdre_value_read_backward_input_kernel[compact_grid](
                grad_output,
                latent,
                value_read,
                metadata_group,
                metadata_start,
                metadata_count,
                grad_latent,
                delta,
                grad_output.stride(0),
                grad_output.stride(1),
                latent.stride(0),
                latent.stride(1),
                value_read.stride(0),
                value_read.stride(1),
                value_read.stride(2),
                grad_latent.stride(0),
                grad_latent.stride(1),
                delta.stride(0),
                delta.stride(1),
                BATCH=ctx.batch,
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                BLOCK_M=ctx.block_m,
                num_warps=4,
                num_stages=2,
            )

            grad_value_read = torch.empty_like(value_read)
            value_tiles = math.ceil(value_dim / 16) * math.ceil(head_dim / 32)
            _bdre_value_read_backward_weight_kernel[(readers, heads, value_tiles)](
                grad_output,
                latent,
                reader_query_start,
                reader_query_count,
                grad_value_read,
                grad_output.stride(0),
                grad_output.stride(1),
                latent.stride(0),
                latent.stride(1),
                grad_value_read.stride(0),
                grad_value_read.stride(1),
                grad_value_read.stride(2),
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                BLOCK_Q=32,
                BLOCK_V=16,
                BLOCK_D=32,
                num_warps=4,
                num_stages=2,
            )

            grad_query = torch.zeros_like(query)
            _bdre_attention_backward_query_kernel[compact_grid](
                query,
                compact_key_codes,
                compact_value_codes,
                key_read,
                cosine,
                sine,
                sorted_indexes,
                metadata_group,
                metadata_start,
                metadata_count,
                grad_latent,
                delta,
                lse,
                grad_query,
                query.stride(0),
                query.stride(1),
                compact_key_codes.stride(0),
                compact_key_codes.stride(1),
                key_code_head_stride,
                compact_value_codes.stride(0),
                compact_value_codes.stride(1),
                value_code_head_stride,
                key_read.stride(0),
                key_read.stride(1),
                key_read.stride(2),
                grad_latent.stride(0),
                grad_latent.stride(1),
                delta.stride(0),
                delta.stride(1),
                lse.stride(0),
                lse.stride(1),
                grad_query.stride(0),
                grad_query.stride(1),
                BATCH=ctx.batch,
                CHUNK=ctx.chunk,
                START_POSITION=ctx.start_position,
                KEY_LENGTH=key_length,
                HEAD_DIM=head_dim,
                KEY_DIM=key_dim,
                VALUE_DIM=value_dim,
                BLOCK_M=ctx.block_m,
                BLOCK_N=ctx.block_n,
                SCALE=1.0 / math.sqrt(head_dim),
                num_warps=BDRE_TRITON_DQ_NUM_WARPS,
                num_stages=BDRE_TRITON_DQ_NUM_STAGES,
            )

            grad_decoded_key = torch.empty(
                groups,
                heads,
                key_length,
                head_dim,
                dtype=query.dtype,
                device=query.device,
            )
            grad_value_head = torch.empty(
                groups,
                heads,
                key_length,
                value_dim,
                dtype=query.dtype,
                device=query.device,
            )
            backward_key_block_m = BDRE_TRITON_DKV_BLOCK_M
            backward_key_block_n = BDRE_TRITON_DKV_BLOCK_N
            key_value_grid = (
                groups,
                heads,
                triton.cdiv(key_length, backward_key_block_n),
            )
            _bdre_attention_backward_key_value_kernel[key_value_grid](
                query,
                compact_key_codes,
                compact_value_codes,
                key_read,
                cosine,
                sine,
                sorted_indexes,
                group_query_start,
                group_query_count,
                grad_latent,
                delta,
                lse,
                grad_decoded_key,
                grad_value_head,
                query.stride(0),
                query.stride(1),
                compact_key_codes.stride(0),
                compact_key_codes.stride(1),
                key_code_head_stride,
                compact_value_codes.stride(0),
                compact_value_codes.stride(1),
                value_code_head_stride,
                key_read.stride(0),
                key_read.stride(1),
                key_read.stride(2),
                grad_latent.stride(0),
                grad_latent.stride(1),
                delta.stride(0),
                delta.stride(1),
                lse.stride(0),
                lse.stride(1),
                grad_decoded_key.stride(0),
                grad_decoded_key.stride(1),
                grad_decoded_key.stride(2),
                grad_value_head.stride(0),
                grad_value_head.stride(1),
                grad_value_head.stride(2),
                BATCH=ctx.batch,
                CHUNK=ctx.chunk,
                START_POSITION=ctx.start_position,
                KEY_LENGTH=key_length,
                HEAD_DIM=head_dim,
                KEY_DIM=key_dim,
                VALUE_DIM=value_dim,
                BLOCK_M=backward_key_block_m,
                BLOCK_N=backward_key_block_n,
                SCALE=1.0 / math.sqrt(head_dim),
                num_warps=BDRE_TRITON_DKV_NUM_WARPS,
                num_stages=BDRE_TRITON_DKV_NUM_STAGES,
                num_ctas=BDRE_TRITON_DKV_NUM_CTAS,
            )

            grad_key_codes_compact = torch.empty_like(compact_key_codes)
            key_code_block = 32
            if per_head:
                _bdre_per_head_key_code_backward_kernel[
                    (groups, heads, triton.cdiv(key_length, key_code_block))
                ](
                    grad_decoded_key,
                    key_read,
                    grad_key_codes_compact,
                    grad_decoded_key.stride(0),
                    grad_decoded_key.stride(1),
                    grad_decoded_key.stride(2),
                    key_read.stride(0),
                    key_read.stride(1),
                    key_read.stride(2),
                    grad_key_codes_compact.stride(0),
                    grad_key_codes_compact.stride(1),
                    grad_key_codes_compact.stride(2),
                    BATCH=batch,
                    KEY_LENGTH=key_length,
                    HEAD_DIM=head_dim,
                    KEY_DIM=key_dim,
                    BLOCK_K=key_code_block,
                    num_warps=2,
                    num_stages=2,
                )
            else:
                _bdre_key_code_backward_kernel[
                    (groups, triton.cdiv(key_length, key_code_block))
                ](
                    grad_decoded_key,
                    key_read,
                    grad_key_codes_compact,
                    grad_decoded_key.stride(0),
                    grad_decoded_key.stride(1),
                    grad_decoded_key.stride(2),
                    key_read.stride(0),
                    key_read.stride(1),
                    key_read.stride(2),
                    grad_key_codes_compact.stride(0),
                    grad_key_codes_compact.stride(1),
                    BATCH=batch,
                    HEADS=heads,
                    KEY_LENGTH=key_length,
                    HEAD_DIM=head_dim,
                    KEY_DIM=key_dim,
                    BLOCK_K=key_code_block,
                    num_warps=2,
                    num_stages=2,
                )
            grad_key_codes = grad_key_codes_compact.view_as(key_codes)
            grad_key_read = torch.empty_like(key_read)
            key_read_tiles = math.ceil(key_dim / 16) * math.ceil(head_dim / 32)
            _bdre_key_read_backward_kernel[(readers, heads, key_read_tiles)](
                compact_key_codes,
                grad_decoded_key,
                grad_key_read,
                compact_key_codes.stride(0),
                compact_key_codes.stride(1),
                key_code_head_stride,
                grad_decoded_key.stride(0),
                grad_decoded_key.stride(1),
                grad_decoded_key.stride(2),
                grad_key_read.stride(0),
                grad_key_read.stride(1),
                grad_key_read.stride(2),
                BATCH=batch,
                KEY_LENGTH=key_length,
                HEAD_DIM=head_dim,
                KEY_DIM=key_dim,
                BLOCK_K=64,
                BLOCK_R=16,
                BLOCK_D=32,
                num_warps=4,
                num_stages=2,
            )
            if per_head:
                grad_value_codes = grad_value_head.view(
                    readers,
                    batch,
                    heads,
                    key_length,
                    value_dim,
                ).permute(0, 1, 3, 2, 4).contiguous()
            else:
                grad_value_codes = grad_value_head.view(
                    readers,
                    batch,
                    heads,
                    key_length,
                    value_dim,
                ).sum(dim=2)
            return (
                grad_query,
                grad_key_codes,
                grad_value_codes,
                grad_key_read,
                grad_value_read,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


def triton_compact_reader(
    query: torch.Tensor,
    key_codes: torch.Tensor,
    value_codes: torch.Tensor,
    key_read: torch.Tensor,
    value_read: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    sorted_indexes: torch.Tensor,
    metadata: BDRETritonReaderMetadata,
    *,
    batch: int,
    chunk: int,
    start_position: int,
    block_m: int = 64,
    block_n: int = 32,
) -> torch.Tensor:
    """Run the differentiable fused compact RC-KV reader."""

    if torch is None:
        raise RuntimeError("The Triton RC-KV reader requires PyTorch.")
    return _BDRETritonCompactReaderFunction.apply(
        query,
        key_codes,
        value_codes,
        key_read,
        value_read,
        cosine,
        sine,
        sorted_indexes,
        metadata.group,
        metadata.query_start,
        metadata.query_count,
        metadata.group_query_start,
        metadata.group_query_count,
        metadata.reader_query_start,
        metadata.reader_query_count,
        batch,
        chunk,
        start_position,
        block_m,
        block_n,
    )
