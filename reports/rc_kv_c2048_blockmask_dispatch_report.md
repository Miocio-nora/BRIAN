# RC-KV C2048 Exact BlockMask and Dispatch Optimization Report

**Date:** 2026-07-16

**Status:** implemented and accepted for the recommended C2048 path

**Branch:** `rc-kv-fused-exact`

## 1. Scope

This optimization remains within `BRIAN-R125-BDRE-RCKV-v1`. It does not change
the RC-KV cache definition, CPBC-DP visibility, route decisions, reader/writer
parameters, RoPE placement, loss terms, or global-batch contract. The previous
`shared_padded_flex` path remains available as the numerical reference.

The work targeted three costs observed at BS16/T2048:

1. dense causal `score_mod` evaluation inside FlexAttention;
2. separate reader-action launches and padded query work;
3. host synchronization while constructing routed action groups.

All profiling and acceptance used idle GPUs 2-3. Existing jobs on GPUs 0-1 and
4-7 were not modified.

## 2. Implemented Paths

### 2.1 Exact padded BlockMask

`shared_padded_flex_blockmask` constructs a FlexAttention `BlockMask` from each
packed query's absolute token position. The mask enforces

```text
key_position <= absolute_query_position
```

including partial 128-token tiles. Key decoding, decoded-Key RoPE, latent Value
attention, and the head-specific Value read are unchanged. The recommended
`execution.flex_kernel_variant: bwd32` sets the four FlexAttention backward tile
dimensions to 32; `auto` preserves PyTorch's default selection.

An unsafe-row kernel hint was explicitly rejected. It caused one NaN at the
1023/1024 tile boundary even though all inputs were finite. Removing that hint
restored exact masking and finite forward/backward behavior. It must not be
reintroduced without a new boundary proof and regression test.

### 2.2 Ragged multi-reader attention

`ragged_flex_blockmask` flattens routed queries and uses one exact mask over
reader, batch, and causal dimensions. It removes padded query slots and
per-action attention calls. However, the current FlexAttention layout expands
decoded K/V for all reader-batch pairs, increasing both memory traffic and peak
allocation. It is retained as an experimental backend, not as the default.

### 2.3 GPU-resident dispatch

`execution.dispatch: grouped_mm_gpu` keeps action sort, counts, offsets, and
valid-token bookkeeping on the GPU. OUT tokens use a fixed-capacity group slot
but are masked from cache writes and hidden-state updates, preserving semantics.
This backend is restricted to `ragged_flex_blockmask` and has direct
logit/loss/gradient equivalence coverage against compact host dispatch.

### 2.4 Benchmark observability

`scripts/benchmark_bdre_tbptt.py` now records individual loss components and
the selected Flex kernel variant. This catches throughput changes that hide a
different execution contract or non-finite auxiliary loss.

## 3. Correctness Acceptance

The final code passed:

- BF16 logits, total loss, and representative gradients against the previous
  score-mod reader for both new attention layouts;
- compact host dispatch versus fixed-capacity GPU dispatch;
- suffix invariance at the actual Flex kernel boundary;
- full-forward versus streamed execution for CPBC-DP and CPBC-FB;
- single-GPU versus DDP2 evaluation on the same `val_legacy` batch;
- the focused BDRE module: 47 tests;
- the complete repository suite: 609 tests.

Single-GPU and DDP2 evaluation both produced
`validation_loss = 115.02999114990234`; route and cache metrics also matched.
The DDP2 three-step smoke completed training, backward, manual gradient sync,
legacy evaluation, and checkpoint writing with global BS32.

PyTorch currently emits a deprecation warning for the internal
`create_block_mask(..., _compile=True)` compatibility flag. It is non-fatal and
is retained because changing mask compilation is a separate kernel-behavior
change that was not required for this acceptance.

## 4. B200 Results

Single-GPU measurements use R125 BF16, BS16, sequence/chunk length 2,048, U1,
one warmup, and complete forward/backward.

| Path | Dispatch | Median tok/s | Peak allocated | Decision |
| --- | --- | ---: | ---: | --- |
| Previous fused score-mod reader | host grouped-MM | 23,246 | 66,011 MiB | reference |
| Exact BlockMask, auto kernel | host grouped-MM | 35,264 | about 66 GiB | superseded |
| Exact BlockMask, `bwd32` | host grouped-MM | **48,956** | **66,016 MiB** | recommended |
| Ragged BlockMask | host grouped-MM | 25,536 | 83,000 MiB | experimental |
| Ragged BlockMask | GPU dispatch | 28,050 | 83,562 MiB | experimental |

The recommended path is 2.106x the previous C2048 implementation, a 110.6%
throughput increase with effectively unchanged peak allocation. GPU dispatch
improves its matching ragged implementation by about 9.8%, proving that the
host grouping cost is real, but the ragged K/V expansion remains more expensive
than padded per-reader execution.

The DDP2 smoke retained local BS16 and global BS32:

| Implementation | Compiled step 2 | Compiled step 3 | Two-step mean |
| --- | ---: | ---: | ---: |
| Previous C2048 | 45,443 tok/s | 45,648 tok/s | 45,546 tok/s |
| Exact BlockMask `bwd32` | 90,780 tok/s | 97,447 tok/s | **94,114 tok/s** |

Peak allocation was about 70.2 GiB per rank. Against the historical same-shape
baseline result of 618,738 global tok/s, the remaining gap is about 6.6x rather
than 13.6x. At the smoke's steady rate, 5B pure training tokens would take
about 14.8 hours, excluding evaluation, checkpoints, public benchmarks, W&B,
data stalls, and startup compilation.

## 5. Negative Results

- A FlashAttention-backed Flex path was not available in the current
  environment because the required CUTE FlashAttention library is absent.
- Combining every reader into one ragged K/V tensor raises memory to about
  83.6 GiB and loses throughput despite fewer launches.
- GPU-resident dispatch alone is not the dominant remaining optimization; its
  gain is visible only after paying the ragged reader cost.
- Declaring causal rows unconditionally safe is incorrect at partial tile
  boundaries and produced a real NaN.
- Larger backward tiles were not uniformly better. The calibrated 32-token
  tile outperformed auto, 16, 64, and 128-token candidates at C2048.

## 6. Prepared Configurations

Recommended:

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_flex_blockmask.yaml
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_flex_blockmask_ddp2_legacyval.yaml
configs/train/smoke_cpbc_r125_5b_dp_u1_c2048_grouped_mm_flex_blockmask_ddp2_legacyval.yaml
```

Experimental, for controlled profiling only:

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_ragged_flex.yaml
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_gpu_ragged_flex.yaml
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_ragged_flex_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_gpu_ragged_flex_ddp2_legacyval.yaml
```

## 7. Remaining Bottleneck

After removing dense score modification and calibrating backward tiles,
FlexAttention backward remains the dominant kernel family. The next exact
optimization should target a compact multi-reader kernel that decodes and
attends only selected reader K/V without either query padding or all-reader K/V
expansion. CUDA graph capture or Python-loop cleanup is secondary: it cannot
close the remaining 6.6x gap while attention backward dominates.
