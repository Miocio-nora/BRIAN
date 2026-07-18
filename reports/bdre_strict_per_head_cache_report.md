# BRIAN Strict Per-Head RC-KV Report

**Status:** implemented and validated as an additive experimental cache layout

**Date:** 2026-07-18

**Branch:** `rc-kv-per-head-cache`

## 1. Motivation

The original RC-KV writer concatenates the twelve local attention heads and
compresses them into one canonical Key and one canonical Value:

```text
K_local [12,64] -> concat [768] -> shared K cache [32]
V_local [12,64] -> concat [768] -> shared V cache [32]
```

Every reader head has an independent `32 -> 64` decoder, but all heads read
the same code. The joint reconstructed Key or Value therefore has rank at most
32 across all twelve heads. This is an intentional but aggressive bottleneck,
not a lossless cache transformation.

The strict per-head layout is an ablation that removes this cross-head
bottleneck without changing routing, position geometry, CPBC visibility,
compiler weights, route losses, data, or evaluation contracts.

## 2. Implemented Architecture

For writer block `b`, head `h`, token `i`, and route step `s`:

```text
cK[i,s,h] = K_local[i,s,h] A_K[b,h]
cV[i,s,h] = V_local[i,s,h] A_V[b,h]
```

with:

```text
A_K[b,h]: R^64 -> R^d_cache
A_V[b,h]: R^64 -> R^d_cache
```

The stored payload is:

```text
K cache: [batch, token, reader, head, d_cache]
V cache: [batch, token, reader, head, d_cache]
```

Compilation remains a weighted sum over route writer steps. The same
position/depth score is used for every head, but payloads are never mixed
between heads:

```text
mK[i,r,l,h] = Sum_s alphaK[i,r,l,s] cK[i,s,h]
mV[i,r,l,h] = Sum_s alphaV[i,r,l,s] cV[i,s,h]
```

Reader block `r` and head `h` own independent decoders:

```text
K_hat[i,r,l,h] = mK[i,r,l,h] B_K[r,h]
B_K[r,h]: R^d_cache -> R^64
```

Values retain the existing exact linear optimization:

```text
latentV[h] = Sum_i attention_weight[h,i] mV[i,h]
output[h] = latentV[h] B_V[r,h]
```

RoPE is applied after Key decoding, preserving the current explicit
position-dependent Key contract.

Head `h1` cannot directly read the cache payload of `h2`. Heads still interact
indirectly because they originate from the same hidden state and their outputs
are mixed by the attention output projection and subsequent blocks.

## 3. Configuration Contract

The additive switch is:

```yaml
bdre_cache_layout: shared   # existing default
bdre_cache_layout: per_head # strict head-isolated cache
```

The existing default is unchanged, so old model configurations and checkpoints
retain shared-cache semantics.

Prepared CPBC-FB C128 U4 variants:

```text
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_flex_incremental_d16.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_flex_incremental_d32.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_flex_incremental_d64.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_triton_recompute_d16.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_triton_recompute_d32.yaml
configs/model/brian_r125_bdre_cpbc_fb_c128_per_head_triton_recompute_d64.yaml

configs/train/q7_cpbc_r125_250m_fb_u4_c128_per_head_d16_ddp2_legacyval.yaml
configs/train/q7_cpbc_r125_250m_fb_u4_c128_per_head_d32_ddp2_legacyval.yaml
configs/train/q7_cpbc_r125_250m_fb_u4_c128_per_head_d64_ddp2_legacyval.yaml
```

All three training configurations preserve the Q2 250M data, legacy
validation, optimizer, seed, routing, benchmark, global batch 32, C128, U4,
and CPBC-FB contracts.

## 4. Capacity and Parameter Accounting

The following counts are for K and V together, per token and compiled reader
state. They exclude tensor metadata.

| Layout | Scalars/token | Versus shared-d32 | Versus standard KV | Model parameters |
| --- | ---: | ---: | ---: | ---: |
| shared-d32 | 64 | 1x | 1/24 | 140,308,105 |
| per-head-d16 | 384 | 6x | 1/4 | 139,914,889 |
| per-head-d32 | 768 | 12x | 1/2 | 140,308,105 |
| per-head-d64 | 1,536 | 24x | 1x | 141,094,537 |

`per-head-d32` is the clean parameter-matched architecture ablation: it has
exactly the same total parameter count as `shared-d32`. `per-head-d64` removes
the forced dimensional bottleneck because each head maps `64 -> 64`, although
training may still learn a non-invertible projection.

## 5. Execution and Optimization

Implemented paths:

- exact token-serial training/incremental inference;
- CPBC-DP and CPBC-FB reference execution;
- incremental-exact prefix compilation with a rank-specialized per-head graph;
- vectorized recompute compilation, selected by default for the larger payload;
- grouped-GPU route-step dispatch;
- precomposed strict per-head writer projection;
- static BlockMask FlexAttention reader;
- differentiable per-head Triton reader for cache dimensions 16, 32, and 64;
- stateful TBPTT and manual DDP gradient synchronization.

The precomposed writer preserves strict head ownership. For each head, only
that head's slice of the local K/V projection is composed with its writer
matrix. No dense all-head writer is introduced by the optimization.

The Triton reader accepts shared `[reader,batch,token,d]` and strict per-head
`[reader,batch,token,head,d]` codes. A cache head stride of zero preserves the
existing shared kernel; a real head stride enforces isolated per-head access.
The Flex path remains the small-shape and FP32 evaluation fallback.

## 6. Correctness Results

Validated properties:

- modifying head 2 input does not change head 1 cache code;
- vectorized per-head compilation equals independent compilation of every head;
- suffix invariance holds for strict per-head token-serial execution;
- CPBC-FB with `chunk=1` matches exact token-serial logits;
- gradients reach every independent writer-head tensor;
- static GPU BlockMask forward/backward matches the explicit reader within BF16 tolerance;
- per-head Triton forward/backward matches BlockMask for d16, d32, and d64;
- recompute and incremental-exact compilation agree for both cache layouts;
- two-process stateful TBPTT gradients match a merged global batch;
- existing shared execution remains green.

Verification summary:

```text
tests/test_bdre_shared_kv.py:       65 passed on CUDA
tests/test_stateful_ddp.py:          2 passed
Q7 configs + config inventory:      33 passed
Python compileall:                   passed
```

## 7. B200 Calibration

Synthetic, untrained, single-B200 backward calibration:

```text
local batch = 16
sequence = 2048
CPBC-FB C128 U4
BF16 autocast
warmup = 2
measured repeats = 3
```

| Layout | Execution | token/s | Slowdown vs shared | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| shared-d32 | Triton + incremental | 20,341 | 1.00x | 15,710 MiB | 16,686 MiB |
| per-head-d16 | Triton + recompute | 16,590 | 1.23x | 31,913 MiB | 45,516 MiB |
| per-head-d32 | Triton + recompute | 14,135 | 1.44x | 51,621 MiB | 78,234 MiB |
| per-head-d64 | Triton + recompute | 10,642 | 1.91x | 90,858 MiB | 145,456 MiB |

All three per-head variants completed a full local-BS16 backward on B200. The
optimized path improves throughput by 25.5-38.9% and cuts allocated memory by
20.9-41.3% relative to matched per-head Flex baselines. `d64` remains exposed
to allocator fragmentation. Detailed matched-backend and DDP results are in
[`bdre_per_head_triton_optimization_report.md`](./bdre_per_head_triton_optimization_report.md).

## 8. Checkpoint Boundary

Shared and strict per-head writer tensors have different shapes:

```text
shared key_write:   [d_cache, 768]
per-head key_write: [12, d_cache, 64]
```

An existing shared checkpoint must continue to load with
`bdre_cache_layout: shared`. It cannot be resumed as a strict per-head run
without an explicit conversion policy, and no lossy conversion is performed
implicitly. Q7 experiments must start as new runs.

## 9. Experimental Recommendation

For the first quality comparison:

1. use `per-head-d32` as the primary parameter-matched ablation;
2. use `per-head-d64` as the no-forced-compression upper bound if resources permit;
3. retain `per-head-d16` as the lower-memory efficiency point.

No 250M quality run has been started by this implementation task. PPL,
reasoning S600, public S600, routing diversity, and checkpoint trajectories
must decide whether the added head capacity is useful.
