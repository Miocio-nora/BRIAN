# Strict Per-Head RC-KV Triton Optimization Report

**Status:** implemented, validated, and enabled in the prepared Q7 configs

**Date:** 2026-07-18

**Branch:** `rc-kv-per-head-cache`

## 1. Question

The initial strict per-head calibration compared a generic FlexAttention path
against the already fused shared-cache Triton path. That comparison mixed the
cache architecture with kernel maturity. This work separates those variables
and adds a differentiable strict per-head Triton reader.

## 2. Matched-Backend Baseline

Single B200, local batch 16, sequence 2048, CPBC-FB C128 U4, BF16 backward,
two warmups, and three measured repeats:

| Layout | Reader | Prefix compiler | Median token/s | Peak allocated |
| --- | --- | --- | ---: | ---: |
| shared-d32 | Flex | incremental exact | 12,157 | 39,412 MiB |
| per-head-d32 | Flex | incremental exact | 10,546 | 74,368 MiB |

With the backend matched, strict per-head is only 1.15x slower. The former
approximately 2x result was therefore dominated by comparing an unfused path
to the mature shared Triton path, not by projection FLOPs alone.

## 3. Implementation

The existing compact Triton reader now accepts either:

```text
shared:   [reader, batch, token, cache_dim]
per-head: [reader, batch, token, head, cache_dim]
```

For per-head input, every forward and backward kernel adds the selected query
head's cache stride. The Key/Value payload of head `h` is never addressed by a
program processing another head.

Backward handling is layout-specific:

- shared cache gradients still sum contributions from all reader heads;
- per-head cache gradients use a separate `(group, head, token tile)` kernel;
- per-head Value gradients are returned without the shared-head reduction;
- reader-matrix gradients read only the corresponding head payload.

The kernel supports `d_cache` 16, 32, and 64 with production head dimensions
divisible by 32. FP32 evaluation retains the existing exact Flex fallback.

The larger per-head payload also favors the vectorized `recompute` prefix
compiler over the online `incremental_exact` state. Their outputs and
gradients are mathematically equivalent; the difference is execution strategy.
The Q7 configs now use:

```yaml
execution:
  prefix_compile: recompute
  reader_kernel: triton_fused
```

The previous Flex/incremental model configs remain available as an oracle.

## 4. B200 Results

All rows use the same shape and measurement contract as Section 2.

| Layout | Execution | Median token/s | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: |
| shared-d32 | Triton + incremental | 20,341 | 15,710 MiB | 16,686 MiB |
| per-head-d16 | Flex + incremental | 11,940 | 54,344 MiB | 67,120 MiB |
| per-head-d16 | Triton + recompute | 16,590 | 31,913 MiB | 45,516 MiB |
| per-head-d32 | Flex + incremental | 10,546 | 74,368 MiB | 94,226 MiB |
| per-head-d32 | Triton + recompute | 14,135 | 51,621 MiB | 78,234 MiB |
| per-head-d64 | Flex + incremental | 8,481 | 114,870 MiB | 161,694 MiB |
| per-head-d64 | Triton + recompute | 10,642 | 90,858 MiB | 145,456 MiB |

Optimization gains over the matched per-head oracle:

| Variant | Throughput gain | Allocated-memory reduction |
| --- | ---: | ---: |
| d16 | 38.9% | 41.3% |
| d32 | 34.0% | 30.6% |
| d64 | 25.5% | 20.9% |

The parameter-matched d32 variant now reaches 69.5% of shared-d32 Triton
throughput. The lower-dimensional d16 variant reaches 81.6%.

## 5. DDP Acceptance

The optimized d32 smoke completed 20 optimizer steps on GPUs 2-3 with global
batch 32 and sequence length 2048:

```text
median global throughput: 27,537 token/s
maximum global throughput: 27,645 token/s
peak allocated per rank:   53,208 MiB
final finite loss:          42.6672
```

The first compilation step is excluded from the steady-state interpretation.
Both ranks completed with manual stateful gradient synchronization and no
missing parameters.

## 6. Correctness Acceptance

- per-head Triton forward/backward matches the static BlockMask reader for
  `d_cache=16,32,64` within the established BF16 tolerance;
- recompute and incremental-exact prefix compilers match for shared and
  per-head layouts, including representative gradients;
- the complete BDRE CUDA module passes 65 tests;
- two-process stateful DDP unit tests pass;
- Q7 and repository config inventory tests pass;
- the existing shared Triton production benchmark remains at approximately
  20.3k token/s, so the optional head stride did not regress shared execution.

## 7. Remaining Bottleneck

Profiling shows that the fused reader itself is no longer the dominant source
of the shared/per-head gap. The larger per-head compiled state causes repeated
cache `cat`, copy, fill, and elementwise work across route steps and TBPTT
chunks. This is consistent with the intrinsic payload ratios: d32 carries 12x
as many canonical cache scalars as shared-d32.

The next optimization boundary is a chunked or segmented cache bank that lets
the reader consume historical and current cache segments without repeatedly
materializing their concatenation. That change is larger than a reader-kernel
tune and should retain the current Triton path as its correctness baseline.
