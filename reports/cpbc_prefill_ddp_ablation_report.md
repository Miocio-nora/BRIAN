# CPBC Approximate Prefill, Stateful DDP, and Ablation Report

**Status:** implementation and smoke validation complete; long-run quality
ablation prepared but not launched

**Date:** 2026-07-16

**Branch:** `bdre-synchronous-prefix-ddp`

## 1. Method Name

**Chunkwise Progressive Bank Completion (CPBC)** is an approximate prefill
procedure in which each chunk is processed in parallel across tokens, while the
per-token cache bank is progressively completed over recurrent steps.

CPBC keeps token causality strict. Full-bank visibility refers only to the route
depth axis and never allows a query to read a future token.

## 2. Depth Visibility Policy

At route depth `l`, a token in the active chunk has produced writer states only
for steps `s <= l`. Both policies therefore use the same depth-prefix cache for
the active chunk. They differ only after a chunk has completed.

### CPBC-DP: Depth-Prefix Visibility

For both active and completed chunks, reader depth `l` sees only writer states
`s <= l`:

```text
C[t, l, r] = compile(reader=r, reader_step=l, writers[t, s <= l])
```

This policy is invariant to chunk boundaries. Changing chunk size changes the
execution shape and gradient boundary, but not forward values under CPBC-DP.

### CPBC-FB: Full-Bank Depth Visibility

The active chunk still uses `s <= l`. When the chunk completes, CPBC-FB
recompiles every `(token, reader_step, reader_block)` cache from all valid
writer steps of that token:

```text
C[t, l, r] = compile(reader=r, reader_step=l, writers[t, all valid s])
```

The reader-step term still affects compile weights; full-bank means that later
writer steps are visible, not that route depth is ignored. Future chunks read
the completed full bank. With `chunk_size=1`, CPBC-FB matches the exact
token-by-token BDRE-Serial forward within the tested numerical tolerance.

Configuration:

```yaml
execution:
  mode: synchronous_prefix       # compatibility key for the CPBC backend
  depth_visibility_policy: full_bank  # full_bank or depth_prefix
```

## 3. Truncated Gradient Horizon

`stateful_tbptt.detach_interval_chunks` defines `U`, the number of consecutive
CPBC chunks retained in one autograd graph:

```yaml
stateful_tbptt:
  enabled: true
  chunk_size: 128
  detach_interval_chunks: 4
```

The trainer accumulates the normalized losses from up to `U` chunks, executes
one backward pass, then detaches pre-block KV, post-block KV, and CPBC cache
state. The last partial group is also backpropagated. For sequence length `T`
and chunk size `C`, the maximum gradient horizon is:

```text
min(T, C * U) tokens
```

`U` does not change forward values at fixed weights. It changes whether later
chunk losses can update hidden states and cache writers from preceding chunks.
This remains true for both CPBC-DP and CPBC-FB.

## 4. Stateful DDP

Stateful CPBC forwards now enter through the model's normal `forward`, so every
chunk passes through the DDP wrapper. Cache state remains rank-local. All
stateful forwards run under `DDP.no_sync()`, and gradients are synchronized once
after the complete optimizer-step accumulation.

The manual synchronization path first all-reduces a used-parameter mask. A
parameter used on only one rank contributes a zero gradient on other ranks;
parameters unused globally retain `grad=None`. Dense gradients are then reduced
in deterministic dtype/device buckets and averaged before clipping and the
optimizer update.

Stateful configs default to `ddp_broadcast_buffers: false` because fixed RoPE
buffers do not need to be rebroadcast for every chunk. The gradient bucket size
is configured by `stateful_tbptt.gradient_sync_bucket_mb`.

Configs may enforce both launch shape and global batch:

```yaml
batch_size: 16
gradient_accumulation_steps: 1
expected_world_size: 2
expected_global_batch_size: 32
```

The trainer rejects a launch that violates either contract.

CPBC-FB additionally requires model-side `execution.chunk_size` to equal
training-side `stateful_tbptt.chunk_size`. FB promotes banks at chunk
boundaries, so differing values would create a train/evaluation semantic
mismatch. The trainer validates this before allocating the model on a GPU.

## 5. Prepared Ablation

The four configs isolate one variable at a time:

| Run | Visibility | U | Chunk | DDP | Local BS | Global BS | Gradient horizon |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPBC-DP-U1 | depth prefix | 1 | 128 | 2 | 16 | 32 | 128 |
| CPBC-FB-U1 | full bank | 1 | 128 | 2 | 16 | 32 | 128 |
| CPBC-FB-U2 | full bank | 2 | 128 | 2 | 16 | 32 | 256 |
| CPBC-FB-U4 | full bank | 4 | 128 | 2 | 16 | 32 | 512 |

`CPBC-DP-U1` versus `CPBC-FB-U1` isolates the Depth Visibility Policy.
`CPBC-FB-U1/U2/U4` isolates the truncated gradient horizon. Data, seed,
optimizer, routing configuration, sequence length, chunk size, and global batch
remain fixed.

Prepared train configs:

```text
configs/train/cpbc_r125_5b_dp_u1_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u1_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u2_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml
```

Quality comparison must use legacy validation loss/PPL, the S600 reasoning and
public suites, teacher accuracy, reasoning exact, route entropy, path diversity,
block coverage, route length, and token-by-token inference. PPL alone is not an
acceptance criterion.

## 6. Correctness Validation

| Check | Result |
| --- | --- |
| CPBC-FB, chunk 1 vs BDRE-Serial | passed for reference and shared-explicit readers, `atol=rtol=2e-5` |
| CPBC-DP vs CPBC-FB active first chunk | identical within `2e-5` |
| CPBC-DP vs CPBC-FB after history exists | diverges as intended |
| CPBC-DP chunk-boundary invariance | passed |
| CPBC-FB full vs streamed at matching boundaries | passed |
| Suffix invariance / token causality | passed |
| U=1 vs U=2 forward loss | identical |
| U=1 vs U=2 gradients | differ as intended |
| CPU DDP vs merged global batch | loss, gradients, and update match |
| Rank-local unused parameter synchronization | passed |
| Tiny CUDA DDP2 CPBC-FB-U2 | 3 train/eval steps and rank checkpoints passed |
| Focused CPBC/TBPTT/DDP suite | passed |
| Full repository regression | 598 passed, 15 pre-existing warnings |

## 7. B200 Feasibility

The following one-shot runs used one B200 each, BF16, local batch 16, sequence
2,048, and chunk 128. Four processes ran concurrently, so timings are host
contention measurements and are not accepted throughput benchmarks.

| Variant | U | Peak allocated | Peak reserved | One-shot throughput |
| --- | ---: | ---: | ---: | ---: |
| CPBC-DP | 1 | 36.3 GiB | 58.0 GiB | 883 tok/s |
| CPBC-FB | 1 | 36.7 GiB | 58.6 GiB | 882 tok/s |
| CPBC-FB | 2 | 67.8 GiB | 100.9 GiB | 829 tok/s |
| CPBC-FB | 4 | 125.5 GiB | 152.3 GiB | 833 tok/s |

The highest-risk shape, CPBC-FB-U4, also passed a real two-B200 DDP optimizer
step with global batch 32:

| Metric | Result |
| --- | ---: |
| Global tokens per optimizer step | 65,536 |
| CPBC chunks per sequence | 16 |
| Backward groups | 4 |
| Gradient synchronization buckets | 9 |
| Peak allocated per rank | 126.5 GiB |
| Global train throughput | 1,896 tok/s |
| Train step time | 34.55 s |
| Eval/checkpoint/rank states | passed |

Chunk 128 is much slower than the previous batch-14/chunk-512 single-GPU
kernel calibration. It provides the requested smaller approximation window and
supports U up to 4, but the current throughput is not suitable for launching
four complete 5B runs. At 1,896 global tok/s, one 5B run would require about 31
days before benchmark and checkpoint overhead.

## 8. Decision

Implementation and smoke acceptance are complete. The four long-run configs are
prepared, but no quality conclusion exists until controlled checkpoints are
trained and evaluated. A bounded-token pilot should be approved before a full
ablation, and CPBC throughput needs more work if the intended budget is near 5B
tokens per arm.

## 9. Commands

Tiny CPBC-FB-U2 DDP validation:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/stage5_bdre_tiny_cpbc_fb_u2_ddp2_debug.yaml
```

R125 CPBC-FB-U4 one-step DDP smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/smoke_cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml
```

Single-rank memory calibration with explicit U:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml \
  --batch-size 16 --sequence-length 2048 --chunk-size 128 \
  --detach-interval-chunks 4
```
