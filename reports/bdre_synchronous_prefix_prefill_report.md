# BDRE Synchronous-Prefix Prefill Implementation Report

**Status:** implementation and correctness validation complete; suitable for
short-run ablations, not yet suitable for a formal 5B run

**Date:** 2026-07-16

**Branch:** `bdre-synchronous-prefix-prefill`

## 1. Decision

The token-serial BDRE backend remains the semantic and debugging reference, but
its measured 25-28 tok/s throughput is not a viable training path. This branch
adds an independent `synchronous_prefix` execution mode that parallelizes the
token dimension inside a chunk while keeping route depth serial.

The new mode implements the agreed cache definition:

- every token retains one cache for each `(reader_step, reader_block)` pair;
- the cache at reader step `l` contains only writer steps `s <= l`;
- a completed chunk persists those step-indexed caches;
- the current chunk constructs the same prefix cache on the fly;
- causal masking prevents token `i` from reading later tokens in its chunk;
- the optional route-step distance term is enabled and normalized;
- training detaches persistent state at chunk boundaries.

This is an additive backend. Existing Non-Global, hidden Global KV, attention
Global KV, pure-factorized Global KV, and exact token-serial BDRE paths are
unchanged.

## 2. Synchronous-Prefix Semantics

For token `i`, route step `s`, and selected writer block `b[i,s]`, the block
produces canonical writer codes:

```text
cK[i,s] = A_K[b[i,s]](K[i,s])
cV[i,s] = A_V[b[i,s]](V[i,s])
```

At reader step `l`, reader block `r` compiles only the route prefix:

```text
C_K[i,l,r] = sum_{s <= l} alpha_K(i,l,r,s) * cK[i,s]
C_V[i,l,r] = sum_{s <= l} alpha_V(i,l,r,s) * cV[i,s]
```

The default score includes block-position similarity and normalized route-step
distance:

```text
score(i,l,r,s) = tau * cosine(z_r, z_b[i,s])
                 - lambda * abs(l - s) / (max_route_steps - 1)
```

Defaults:

```yaml
bdre_depth_mode: synchronous_prefix
bdre_step_lambda: 0.25
bdre_normalize_step_distance: true
bdre_reader_step_cache: eager
bdre_self_kv_mode: bdre_prefix
execution:
  mode: synchronous_prefix
  chunk_size: 512  # model-side safety limit, not the training chunk default
```

The hard `s <= l` support mask is applied before softmax. Changing writer codes
from later route steps therefore cannot change an earlier reader-step cache.

## 3. Chunk Forward

For a chunk starting at absolute token position `t0`:

1. Fixed pre-blocks process all chunk tokens with conventional causal attention
   over their retained pre-block KV state.
2. At route step `l`, the router selects one action independently for every
   active token in `[batch, chunk]`.
3. Selected free blocks compute Q/K/V and canonical K/V writes for all selected
   tokens, grouped by block.
4. The compiler builds `C_K/C_V[:, l, :, :]` for every token and reader block
   from writer steps `s <= l`.
5. A selected reader attends to completed-token cache `C[:, l, reader]` plus
   the current chunk's on-the-fly cache at the same `l`. A lower-triangular
   token mask enforces causality.
6. Exited tokens stop changing while remaining tokens advance to route step
   `l + 1`.
7. All reader-step caches are persisted, fixed post-blocks process the chunk,
   and logits are emitted.

The persistent step-cache shape is:

```text
[batch, tokens, max_route_steps, free_blocks, canonical_dim]
```

It is not a `max_route_steps x max_route_steps` cache. The current reader step
and visible writer-prefix depth are tied to the same index `l`.

During stateful training, the current chunk is differentiable end to end. The
pre/post KV states and BDRE step caches are detached before the next chunk, so
the forward values are preserved while gradients are truncated at the chunk
boundary.

## 4. Implementation Surface

Core implementation:

```text
src/brian_sphere_llm/memory/bdre_shared_kv.py
src/brian_sphere_llm/model/bdre_model.py
src/brian_sphere_llm/train/trainer.py
```

Configs:

```text
configs/model/brian_tiny_bdre_rckv_synchronous_prefix.yaml
configs/model/brian_r125_bdre_rckv_synchronous_prefix.yaml
configs/train/stage5_bdre_tiny_synchronous_prefix_debug.yaml
configs/train/bdre_rckv_r125_5b_synchronous_prefix_b8_c128_legacyval.yaml
```

The accepted long-run single-GPU candidate uses local batch 8 and chunk 128.

## 5. Correctness Results

| Check | Result |
| --- | --- |
| Future writer-step intervention | earlier prefix cache unchanged |
| Prefix suffix-invariance | passed |
| Full synchronous forward vs streamed chunks | passed, `atol=rtol=2e-5` |
| Full synchronous forward vs one-token incremental | passed, `atol=rtol=2e-5` |
| Persistent cache shape | `[B,T,16,8,32]` for R125 Key and Value caches |
| CPU forward/backward | finite |
| B200 BF16 forward/backward | finite |
| Tiny 3-step train/eval/checkpoint smoke | passed |
| Synchronous reader-step visualization | passed; future-step weights exactly zero |
| BDRE focused suite | 27 passed |
| Full repository suite | 585 passed |

Chunk-boundary invariance applies to forward values under the new
synchronous-prefix cache definition. It does not claim equivalence with the old
token-serial backend, whose historical reader-step cache can be compiled from a
token's completed route. Chunk size also changes the gradient horizon because
state is detached between training chunks.

## 6. B200 Calibration

All measurements use BF16 on one uncontended NVIDIA B200 and include complete
forward and backward for one random-token microbatch. Optimizer, evaluation,
checkpoint, and W&B overhead are excluded.

| Backend | Batch | Sequence | Chunk | Throughput | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact token-serial | 32 | 128 | 8 | 28.18 tok/s | 3.28 GiB | 3.59 GiB |
| Synchronous prefix | 32 | 128 | 128 | 935.63 tok/s | 19.82 GiB | 20.29 GiB |
| Exact token-serial | 32 | 256 | 8 | 25.53 tok/s | 4.46 GiB | 4.86 GiB |
| Synchronous prefix | 32 | 256 | 256 | 1,626.20 tok/s | 63.78 GiB | 67.03 GiB |
| Synchronous prefix | 32 | 512 | 256 | 1,738.93 tok/s | 115.37 GiB | 142.34 GiB |
| Synchronous prefix | 8 | 2,048 | 128 | 447.15 tok/s | 55.12 GiB | 100.94 GiB |
| Synchronous prefix | 8 | 2,048 | 256 | 684.54 tok/s | 107.86 GiB | 177.41 GiB |

The same-length throughput ratios are approximately 33x at sequence 128 and
64x at sequence 256. These ratios compare two model semantics and should not be
reported as exact-kernel equivalence speedups.

At the formal 2048-token length, chunk 256 is 52% faster than chunk 128 but its
177.4 GiB reserved memory leaves essentially no operating margin on a 183 GiB
B200. Chunk 128 is the stable candidate.

A one-second sample over the active tail of the chunk-128 calibration observed
approximately 35% average SM utilization, 97% peak SM utilization, and 22% peak
memory-controller utilization. The alternating 90%+ bursts and zero-utilization
gaps confirm that host-side route grouping, block dispatch, and selected-reader
attention still leave substantial GPU bubbles.

## 7. Training Feasibility

At the measured 2048-token throughput:

```text
chunk 128: 5B / 447.15 tok/s = approximately 129 days on one GPU
chunk 256: 5B / 684.54 tok/s = approximately 85 days on one GPU
```

Actual wall time would be longer after optimizer, evaluation, checkpoint, and
benchmark overhead. The current backend is therefore acceptable for smoke
tests and bounded ablations, but not for a formal 5B run.

The stateful trainer currently rejects distributed execution. Multi-GPU scaling
must not be assumed from the table and needs a separate DDP implementation and
equivalence test.

## 8. Remaining Bottlenecks

The largest memory and throughput cost is selected-reader attention. For every
selected query, the current implementation decodes and materializes historical
Keys with shape approximately `[selected_queries, heads, history, head_dim]`.
This duplicates the same reader history across many queries.

The next engineering priorities are:

1. batch selected queries by `(reader_block, reader_step, batch)` so decoded
   reader history is shared rather than repeated per query;
2. retain the low-dimensional Value aggregation but fuse or cache decoded Key
   transforms where causality permits;
3. remove remaining host synchronization from action grouping;
4. add and validate stateful DDP only after the single-rank kernel is stable.

No formal routing ablation should be launched from this report alone. The next
ablation plan should account for both mechanism quality and the gradient-horizon
effect of chunk size, and should use a bounded token budget until throughput is
improved.

## 9. Commands

Tiny end-to-end smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/train.py \
  --config configs/train/stage5_bdre_tiny_synchronous_prefix_debug.yaml
```

Stable 2048-token calibration:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/bdre_rckv_r125_5b_synchronous_prefix_b8_c128_legacyval.yaml \
  --batch-size 8 --sequence-length 2048 --chunk-size 128
```

The prepared R125 config should be used for short-run experiments only until a
new ablation budget is approved.
