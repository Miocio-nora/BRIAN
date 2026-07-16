# RC-KV Exact Fused Reader and Large-Chunk Acceleration Report

**Date:** 2026-07-16

**Status:** implemented and accepted for controlled experiments

**Branch:** `rc-kv-fused-exact`

## 1. Scope and Safety

This work accelerates the existing RC-KV/CPBC implementation without creating
a new model version. It does not change the cache definition, route policy,
reader/writer parameterization, RoPE placement, causal mask, or DP/FB Depth
Visibility Policy. The accepted grouped-MM backend remains available as the
reference.

Development used an isolated worktree based on commit `9cfbbd7`. The formal
jobs on GPUs 0-1 were not stopped or modified; profiling and acceptance used
only GPUs 2-3.

## 2. Implemented Optimizations

### 2.1 Fused reader graph

The `shared_padded_flex` backend compiles the following existing reader algebra
into one FlexAttention graph:

1. decode canonical Keys with each reader/head-specific `key_read` matrix;
2. apply temporal RoPE after Key decoding, as in the reference implementation;
3. apply the same absolute token-causal mask;
4. aggregate compressed Values in latent space;
5. apply each reader/head-specific `value_read` matrix.

It avoids materializing the full score and attention-weight tensors. The
explicit reader remains the CPU, small-dimension, and dropout fallback. The
mathematical operation is unchanged, although online softmax and BF16 kernel
accumulation are not bitwise identical to explicit FP32-softmax execution.

### 2.2 Bounded cross-reader batching

`execution.flex_reader_group_size` controls how many reader actions enter one
FlexAttention call. Every grouped reader retains independent K/V reader
matrices and cache slices. This is an execution-layout option, not parameter
sharing.

Reader batching is shape-sensitive:

- C512 and FB-C128 use group size 8 because launch reduction dominates;
- DP-C2048 uses group size 1 because cross-reader query padding increases the
  dominant attention workload.

### 2.3 Vectorized full-bank compilation

`execution.full_bank_compile: vectorized` computes all completed FB reader
depths and reader blocks in one tensor program. Position score, normalized
depth distance, late-step score, support mask, independent K/V temperatures,
and weighted K/V output are unchanged from the reader-depth loop.

### 2.4 DP chunk 2048

For identical route decisions, CPBC-DP cache/attention values are
chunk-boundary invariant, so C2048 preserves the DP visibility rule. It is not
training-dynamics equivalent to C512: with U1, the TBPTT gradient horizon grows
from 512 to 2,048 tokens and four backward groups become one. Scheduled routing,
noise, and random-route override also consume RNG in different tensor groupings,
so the realized stochastic paths need not match across chunk sizes.

## 3. Correctness Acceptance

The following checks passed:

- vectorized FB compiler output, K/V weights, and writer/position gradients
  against the depth loop;
- explicit versus fused CUDA logits, loss, and representative gradients within
  BF16 rounding scale;
- actual Flex kernel suffix invariance for DP and FB;
- full versus streamed forward consistency for DP and FB;
- complete focused BDRE test module;
- complete repository regression: 606 passed with 15 existing PyTorch warnings;
- DDP2 DP-C2048 and FB-C128 training smokes with two rank-local states and a
  shared checkpoint;
- single-GPU versus DDP2 legacy evaluation on the same batch.

For the last check, validation loss was `115.0287628` on one GPU and
`115.0287552` under DDP2, a difference of approximately `7.6e-6`. Routing and
cache metrics agreed.

The fused reader produced exactly equal reported loss in the controlled
explicit comparison. BF16 logits had mean absolute difference about `0.020`
and maximum difference `0.125`; sampled gradient maximum differences were at
most `0.013`.

## 4. B200 Measurements

All single-GPU rows are R125 BF16, BS16, sequence length 2,048, complete
forward/backward, one warmup, and five measured repeats.

| Configuration | Visibility | C | Reader group | Median tok/s | Peak allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| Accepted grouped-MM reference | DP | 512 | explicit | 15,167 | 120,947 MiB |
| Fused reader | DP | 512 | 1 | 17,507 | 24,054 MiB |
| Final fused reader | DP | 512 | 8 | 19,139 | 24,742 MiB |
| Final fused reader | DP | 2048 | 1 | 23,246 | 66,011 MiB |
| Historical CPBC-FB | FB | 128 | explicit | 882 | about 36.7 GiB |
| Fused + vectorized compiler | FB | 128 | 1 | 6,489 | 13,253 MiB |
| Final fused + vectorized compiler | FB | 128 | 8 | 9,013 | 13,460 MiB |

Relative to the accepted C512 grouped-MM reference, final C512 gains about
26.2% throughput while reducing allocated memory about 79.5%. DP-C2048 gains
about 53.3% throughput and still uses only about 64.5 GiB allocated on a B200.
Final FB-C128 is over 10x faster than the historical FB implementation.

The DP-C2048 DDP2 smoke preserved global BS32. After first-call compilation,
steps 2 and 3 reached 45,443 and 45,648 global tok/s with about 70.2 GiB peak
allocated per rank. This is about 1.94x the single-GPU rate and corresponds to
roughly 1.27 days for 5B pure training tokens, excluding evaluation,
checkpointing, public benchmarks, W&B, and startup compilation.

## 5. Negative Results Kept Explicit

Larger CPBC chunks do not yield linear speedup. Moving C512 to C2048 reduces
four backward groups to one, but full attention work remains and grows with
the query/key extent. The measured gain is about 21.5% over final fused C512,
not 4x.

Cross-reader batching is also not universally positive:

| DP-C2048 reader group | Median tok/s |
| ---: | ---: |
| 1 | 23,246 |
| 2 | 19,259 |
| 8 | 21,779 |

The final C2048 config therefore uses group size 1. Padding-heavy grouping is
not enabled merely to reduce call count.

## 6. Prepared Configurations

```text
configs/model/brian_r125_bdre_cpbc_dp_c512_grouped_mm_flex.yaml
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_flex.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_grouped_mm_flex.yaml
configs/train/cpbc_r125_5b_dp_u1_c512_grouped_mm_flex_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_flex_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u1_c128_grouped_mm_flex_ddp2_legacyval.yaml
```

Each has a bounded smoke counterpart. The legacy validation split and global
batch 32 contract are retained.

## 7. Remaining Bottleneck

At C2048, FlexAttention forward/backward is now the dominant GPU work. Host
action dispatch, index/copy kernels, grouped weight preparation, and per-step
Python control remain visible, but removing them cannot produce a 4x gain when
attention arithmetic dominates. FB-C128 still pays for 16 chunk/backward
groups and first-call compiler startup. Further work should be accepted only
with end-to-end measurements and the same causal/gradient/DDP contract.
