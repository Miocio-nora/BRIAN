# BDRE Synchronous-Prefix Kernel Optimization Report

**Status:** first performance checkpoint validated; deeper optimization remains

**Date:** 2026-07-16

**Branch:** `bdre-synchronous-prefix-kernel-opt`

## 1. Scope

This branch optimizes the mathematically equivalent `synchronous_prefix`
attention path. It does not change the route rule, causal mask, compiled cache
definition, writer/reader parameters, or TBPTT detach boundary. The original
per-query implementation remains available as `per_query_reference`.

The new `shared_padded_explicit` backend removes the dominant duplication:

- selected queries are packed by `(reader block, route step, batch)`;
- one batch history is gathered and decoded once, rather than once per query;
- queries are padded only within each active batch and reader group;
- Key decoding uses one flattened `F.linear` over all heads;
- attention weights aggregate compressed Values before the per-head Value read;
- grouped-host dispatch prepares padding metadata from the existing host action
  copy, avoiding an additional device-to-host scalar synchronization per block.

## 2. Correctness Contract

Both backends use the same parameters and produce the same cache state. The
focused tests compare:

- full forward, streamed chunks, and one-token incremental forward;
- suffix invariance under the synchronous causal mask;
- logits and loss between the reference and optimized backends;
- gradients for model, writer, and reader parameters;
- BF16 forward/backward finiteness on CUDA.

The focused BDRE suite currently passes all 30 tests. On the R125 B200
calibration, identical seeds produce identical reported loss for the reference
and optimized backends.

## 3. Measurement Method

Measurements use one uncontended NVIDIA B200, BF16, one complete forward and
backward microbatch, and the same random input/routing seed. Optimizer,
evaluation, checkpoint, and W&B overhead are excluded.

The benchmark now supports `--warmup-steps` and `--repeats`. Results below use
one warmup followed by repeated measurements in the same process. This is
important: first-call CUDA/kernel setup made the old one-shot measurements
roughly two to three times slower than steady-state training iterations.

## 4. Reference A/B

Formal context length, batch 8, chunk 256, three measured repeats:

| Backend | Median tok/s | Mean tok/s | Peak allocated | Peak reserved | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `per_query_reference` | 1,345.44 | 1,344.90 | 111,353 MiB | 179,230 MiB | 114.644012 |
| `shared_padded_explicit` | 3,918.21 | 3,915.99 | 27,326 MiB | 52,352 MiB | 114.644012 |

At the same batch, sequence, chunk, input, and seed, the optimized backend is
approximately **2.91x faster**, reduces peak allocated memory by **75.5%**, and
reduces peak reserved memory by **70.8%**. The loss is exactly equal at the
reported precision.

## 5. Stable Single-B200 Candidate

Batch 12, sequence 2,048, chunk 512, one warmup plus five repeats:

| Metric | Result |
| --- | ---: |
| Median throughput | 6,410.08 tok/s |
| Mean throughput | 6,408.48 tok/s |
| Per-repeat range | 6,403.33-6,410.88 tok/s |
| Peak allocated | 85,244 MiB |
| Peak reserved | 132,346 MiB |
| Reported loss | 114.692329 |

The prepared config is:

```text
configs/train/bdre_rckv_r125_5b_synchronous_prefix_b12_c512_shared_explicit_legacyval.yaml
```

At the kernel-only steady-state rate, 5B tokens would take about 9.0 days on
one B200. Real training will be slower after optimizer, data loading,
evaluation, checkpoint, and W&B overhead. Chunk 512 also gives a longer
gradient horizon than chunk 128; the throughput result is mathematically
forward-equivalent but is not a training-dynamics equivalence claim.

## 6. Small Optimization Decisions

### Host packing retained

Five-repeat BS12/C512 means were 6,324.62 tok/s with device-side metadata
packing and 6,408.48 tok/s with host-side packing. The retained implementation
is 1.33% faster with unchanged loss and memory.

### Flattened Key read retained

A B200 microbenchmark showed the single flattened `F.linear` Key decode about
18% faster in forward and 4.8% faster in forward/backward than the equivalent
head-wise `einsum`. Focused model tests confirm matching gradients.

### Fused SDPA attempt removed

An experimental padded SDPA backend passed tiny correctness tests but took more
than 120 seconds for the BS8, sequence-512, chunk-128 R125 backward and was
stopped. The reference completed that workload in about 9.7 seconds without
warmup. The likely cause is a slow fallback for the unequal Key head dimension
and compressed Value dimension plus expanded padded tensors. The backend and
its configs were removed rather than retained as dead complexity.

## 7. Remaining Bottlenecks

The optimized path still launches many small route/block kernels and retains a
bursty GPU profile. A sampled long run reached high instantaneous utilization
but only about 30% average SM utilization. The next profiler-driven work should
focus on:

1. route-step host synchronization and Python dispatch;
2. repeated per-block QKV and writer projection launches;
3. padding/index-copy overhead for skewed reader groups;
4. compiler and route-group operations that can be batched without changing
   routing or cache semantics.

Existing FlexAttention and grouped sparse-padded experiments in the repository
were slower in backward and should not be reintroduced without new evidence.

## 8. Reproduction

Reference:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/bdre_rckv_r125_5b_synchronous_prefix_b8_c128_legacyval.yaml \
  --batch-size 8 --sequence-length 2048 --chunk-size 256 \
  --warmup-steps 1 --repeats 3
```

Optimized candidate:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/bdre_rckv_r125_5b_synchronous_prefix_b12_c512_shared_explicit_legacyval.yaml \
  --warmup-steps 1 --repeats 5
```
