# RC-KV Static Route-Step Execution Report

**Date:** 2026-07-16

**Branch:** `rc-kv-static-route-step`

**Model contract:** unchanged `BRIAN-R125-BDRE-RCKV-v1`

**Scope:** exact execution optimization for the accepted CPBC-DP C2048 path;
no formal long training was started

## 1. Result

The optimized path removes CPU-visible route-step dispatch, avoids materializing
full local K/V before canonical compression, and fuses the largest remaining
pointwise chains. It does not change RC-KV cache visibility, route decisions,
parameter ownership, current-token memory, or the CPBC-DP approximation.

At local BS16, sequence/chunk 2048, U1, BF16 on one B200:

| Candidate | Median token/s | Peak allocated | Change from accepted predecessor |
| --- | ---: | ---: | ---: |
| Grouped reader + online prefix | 63,078 | 66,938 MiB | reference |
| GPU static workspace only | 67,793 | 69,609 MiB | +7.5% |
| + compiled decoded-Key RoPE | 71,757 | 69,609 MiB | +13.8% |
| + precomposed writer projection | 74,011 | 67,458 MiB | +17.3% |
| + compiled route pointwise | **78,381** | **61,697 MiB** | **+24.3%** |

A same-card rerun of the predecessor measured 62,518 token/s and 66,938 MiB,
so the direct paired gain is 25.4% and peak allocation falls by 5,241 MiB.

The final two-B200 smoke preserves local BS16 and global BS32:

| Steady step | Global token/s | Step time |
| ---: | ---: | ---: |
| 2 | 150,009 | 0.4369 s |
| 3 | 152,584 | 0.4295 s |
| 4 | 150,951 | 0.4342 s |
| 5 | 149,568 | 0.4382 s |
| Mean | **150,778** | **0.4347 s** |

Per-rank peak allocation was 65,510 MiB. DDP efficiency relative to twice the
single-card median is 96.2%. The previous candidate averaged 128,294 token/s,
so this is a 17.5% DDP2 gain. Against the matched ordinary Transformer at
618,738 token/s, the measured system gap contracts from 4.82x to **4.10x**.
A pure 5B-token pass is about **9.21 hours** at steady throughput, excluding
compile, evaluation, checkpoint, W&B, and data stalls.

## 2. Exact Execution Changes

### 2.1 GPU-resident static reader workspace

`grouped_mm_gpu` now supports the exact shared-padded BlockMask backend when
one reader group covers all eight free blocks. Active actions are sorted on the
GPU. Valid queries for each `(reader, batch)` receive contiguous local ranks;
OUT rows are sorted after valid rows of the clamped final reader and occupy only
the remaining workspace tail. The valid-row BlockMask excludes those rows from
attention.

The workspace has a fixed worst-case capacity of `chunk` rows per
`(reader, batch)`. Tensor shapes therefore remain static across route steps,
and dispatch requires no selected-action synchronization to the CPU. This is
an execution layout change, not padded-token computation being made visible to
the model.

### 2.2 Precomposed writer projection

For one selected block, the staged writer path was

```text
[Q, K, V] = W_QKV x
cK = A_K K
cV = A_V V
```

Because both writer transforms are linear, the optimized path constructs

```text
W_fused = [W_Q; A_K W_K; A_V W_V]
[Q, cK, cV] = W_fused x
```

for each route block. Autograd still reaches `W_QKV`, `A_K`, and `A_V`; only
the redundant full-width local K/V materialization and two grouped GEMMs are
removed. The grouped-MM count in the final profile falls from 315 to 225.

### 2.3 Compiled elementwise paths

Two explicit opt-in paths use `torch.compile`:

- decoded-Key temporal RoPE;
- grouped RMSNorm and SwiGLU `silu(gate) * up` pointwise work.

Only the large elementwise regions are captured. Grouped GEMM and
FlexAttention remain outside these compiled helpers, avoiding the compiler
split failure observed when trying to capture the full reader.

The new execution controls are:

```yaml
execution:
  dispatch: grouped_mm_gpu
  decoded_key_rope: compiled
  writer_projection: precomposed
  route_pointwise: compiled
```

Defaults remain `grouped_mm`, `eager`, `staged`, and `eager`, so historical
configs retain their prior behavior.

## 3. Profiling

The complete forward/backward profile improves from 421.924 ms to 382.258 ms
self CUDA and reduces observed CUDA launches from 9,346 to 7,156. The final
profile still attributes substantial time to:

- FlexAttention backward;
- copies/scatters, about 61.6 ms;
- remaining elementwise work, about 42.0 ms;
- decoded-Key RoPE;
- grouped expert GEMM, about 19.8 ms across 225 calls.

The static workspace slightly increases reader-attention arithmetic relative
to compact host-built groups, but eliminating host synchronization and fusing
writer/pointwise work yields the larger end-to-end gain.

## 4. Correctness Acceptance

The candidate passed:

- static workspace uniqueness, compact valid ranks, and OUT-tail placement;
- full logits/loss and representative gradient comparison against the accepted
  host-grouped BlockMask path in CUDA BF16;
- CPBC-DP and CPBC-FB suffix invariance and streamed/full consistency for both
  host and GPU-static dispatch;
- a five-step two-rank train/backward/manual-gradient-sync/eval/checkpoint smoke;
- all 54 focused BDRE tests;
- all 616 repository tests.

The DDP smoke synchronized 561,232,420 gradient bytes in nine buckets across
135 globally used parameters, with no rank-local missing parameters.

Legacy evaluation on the same batch produced:

```text
single B200: validation_loss = 115.02975463867188
DDP2:        validation_loss = 115.02935028076172
relative difference          = 3.5e-6
```

Routing and cache telemetry differed only at comparable reduction-rounding
scale. Precomposed projections and compiled BF16 kernels change accumulation
order, so acceptance is tolerance-based rather than bitwise. Causal visibility
and model semantics are unchanged.

## 5. Configurations

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_gpu_static_blockmask_incremental.yaml
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_gpu_static_blockmask_incremental_ddp2_legacyval.yaml
configs/train/smoke_cpbc_r125_5b_dp_u1_c2048_grouped_mm_gpu_static_blockmask_incremental_ddp2_legacyval.yaml
```

The formal configuration keeps the established balanced training data,
legacy validation, local BS16/global BS32, C2048, U1, route schedule, and
checkpoint benchmark policy. Only the execution backend changes.

## 6. Conclusion

This stage is an accepted exact systems optimization, not RC-KV v2. It raises
the current CPBC-DP C2048 implementation to 78.4k token/s on one B200 and
150.8k token/s on two B200s while reducing memory. The remaining 4.10x gap is
still material and is not a proven hardware limit. Further work should target
FlexAttention backward and workspace copy/scatter traffic, but only under the
same logits/loss/gradient and causal-consistency acceptance contract.
