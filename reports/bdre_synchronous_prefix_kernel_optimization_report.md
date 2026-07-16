# BDRE Synchronous-Prefix Kernel Optimization Report

**Status:** second performance checkpoint validated; deeper optimization remains

**Date:** 2026-07-16

**Branch:** `bdre-synchronous-prefix-kernel-opt`

**Performance follow-up:** the accepted grouped-MM backend and current B200
numbers are documented in
[`rc_kv_deep_acceleration_report.md`](./rc_kv_deep_acceleration_report.md).
The measurements below remain the pre-grouped-MM reference.

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

The focused BDRE suite currently passes all 31 tests in the performance
environment, and the complete repository suite passes in the primary
`brian-sphere` environment. On the R125 B200 calibration, identical seeds
produce identical reported loss for the reference and optimized backends.

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
| `per_query_reference` | 1,467.80 | 1,468.06 | 111,353 MiB | 179,230 MiB | 114.644012 |
| `shared_padded_explicit` | 4,695.64 | 4,685.06 | 27,333 MiB | 52,986 MiB | 114.644012 |

At the same batch, sequence, chunk, input, and seed, the optimized backend is
approximately **3.20x faster**, reduces peak allocated memory by **75.5%**, and
reduces peak reserved memory by **70.4%**. The loss is exactly equal at the
reported precision.

## 5. Stable Single-B200 Candidate

Batch 14, sequence 2,048, chunk 512, one warmup plus 20 repeats:

| Metric | Result |
| --- | ---: |
| Median throughput | 11,658.95 tok/s |
| Mean throughput | 11,633.61 tok/s |
| Per-repeat range | 11,457.65-11,737.98 tok/s |
| Peak allocated | 101,808 MiB |
| Peak reserved | 159,848 MiB |
| Reported loss | 114.653511 |

The prepared config is:

```text
configs/train/bdre_rckv_r125_5b_synchronous_prefix_b14_c512_shared_explicit_legacyval.yaml
```

At the kernel-only steady-state rate, 5B tokens would take about 5.0 days on
one B200. Real training will be slower after optimizer, data loading,
evaluation, checkpoint, and W&B overhead. Chunk 512 also gives a longer
gradient horizon than chunk 128; the throughput result is mathematically
forward-equivalent but is not a training-dynamics equivalence claim.

## 6. Small Optimization Decisions

### Advanced indexing cleanup retained

The remaining profiler hotspot was not attention. Position lookup in the cache
compiler used `normalized_positions[safe_blocks]`. Although the table has only
eight rows, its generic `IndexBackward/_index_put_impl_` took 65.6 ms, or 37%
of self CUDA time, for 15 calls in one 128-token chunk. Equivalent `F.embedding`
lookups now serve compiler writer/reader positions and routed block positions;
selected hidden/position gathers use `index_select`.

A direct repeated-ID gradient test confirms exact output and gradient equality
between the old indexing expression and `F.embedding`. The focused model suite
also preserves logits, loss, cache state, and parameter gradients.

For one profiled chunk, generic `IndexBackward` disappeared and self CUDA time
fell from 177.0 ms to 112.9 ms. At BS12/C512, median throughput rose from
6,410.08 to 10,880.33 tok/s, a 69.7% increase. The compiler change is shared by
both attention backends: the final reference/shared A/B table above was rerun
after this optimization.

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

### Batch 15 rejected as the default

Batch 15 reached 11,843 tok/s, only 1.7% above batch 14, while peak reserved
memory rose to 172,276 MiB and left about 10.8 GiB free on the B200. Batch 14
retains about 23.0 GiB and is the stable candidate.

## 7. Remaining Bottlenecks

The optimization changed the runtime profile materially. During the stable
portion of a 20-repeat batch-14 run, one-second samples averaged approximately
79% SM utilization and peaked at 96%, compared with roughly 30% before the
indexing cleanup. A one-chunk profiler still records about 28,600 kernel
launches; `copy_`, matrix multiplies, and elementwise multiplies are now the
largest CUDA categories. The next profiler-driven work should focus on:

1. route-step host synchronization and Python dispatch;
2. repeated per-block QKV and writer projection launches;
3. padding/index-copy overhead for skewed reader groups;
4. repeated dtype/copy operations and compiler bookkeeping that can be fused or
   cached without changing routing or cache semantics.

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
  --config configs/train/bdre_rckv_r125_5b_synchronous_prefix_b14_c512_shared_explicit_legacyval.yaml \
  --warmup-steps 1 --repeats 20
```
