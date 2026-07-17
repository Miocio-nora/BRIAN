# CPBC Approximate Prefill, Stateful DDP, and Ablation Report

**Status:** implementation and smoke validation complete; matched C128 250M
visibility and U1/U2/U4/U8 quality pilots complete; C128-FB-U4 selected for the
first formal 5B FB run

**Updated:** 2026-07-18

**Branch:** `bdre-synchronous-prefix-ddp`

**Performance follow-up:** the accepted grouped-MM backend raises the same
formal DDP2 shape to 28.6-30.6k global tok/s. See
[`rc_kv_deep_acceleration_report.md`](./rc_kv_deep_acceleration_report.md).
The tables below retain the original CPBC/DDP ablation measurements.

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

The matched plain-Transformer control uses the same balanced data, legacy
validation split, global batch, token budget, and checkpoint benchmark cadence:

```text
configs/train/baseline_r125_5b_balanced_ddp2_legacyval.yaml
```

This replaces the old unbalanced `r125_main_5b` Sbase as the primary plain
baseline for CPBC comparisons; the old result remains a historical reference.

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

### CPBC-DP Four-B200 Training Profile

CPBC-DP does not incur an intrinsic regression from the FB/U/DDP additions.
Controlled, warmed single-B200 measurements on the current branch give:

| Shape | Throughput | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| local BS 16, sequence 2,048, chunk 128, U1 | 3,878 tok/s mean | 36.3 GiB | 57.7 GiB |
| local BS 8, sequence 2,048, chunk 512, U1 | 8,901 tok/s mean | 47.8 GiB | 72.2 GiB |
| local BS 14, sequence 2,048, chunk 512, U1 | 11,617 tok/s mean | 99.4 GiB | 156.1 GiB |

The current-branch BS14/C512 result is within 0.4% of the historical 11,659
tok/s result. The C128 slowdown is therefore an execution-shape cost: it uses
16 chunks and 16 backward groups instead of four, increasing host dispatch,
cache compilation, and kernel-launch overhead.

The recommended throughput-oriented four-B200 DP config fixes global batch at
32 with local batch 8 per rank:

```text
configs/train/cpbc_r125_5b_dp_u1_c512_ddp4_legacyval.yaml
```

A real three-step DDP4 smoke produced the following steady-state measurements
after the one-time first-step warmup:

| Metric | Result |
| --- | ---: |
| Global batch / tokens per step | 32 / 65,536 |
| CPBC chunks / backward groups | 4 / 4 |
| Gradient horizon | 512 tokens |
| Gradient synchronization | 9 buckets, 561 MB |
| Peak allocated per rank | 51.4 GiB |
| Steady global throughput | 24,563-24,896 tok/s |
| Steady train step | 2.63-2.67 s |
| Rank-local checkpoint states | 4, passed |

At approximately 24.7k global tok/s, 5B tokens require about 2.35 days of pure
training. Evaluation, public benchmarks, checkpointing, and W&B should put a
formal run closer to 2.5-3 days. This C512 config is appropriate for a fast DP
training run. It is not a controlled replacement for the C128 DP arm in the
Depth Visibility ablation: comparing DP-C512 against FB-C128 would also change
chunk count and gradient horizon.

The resource-efficient default uses two B200s with local batch 16 while keeping
the same global batch and optimizer-step token count:

```text
configs/train/cpbc_r125_5b_dp_u1_c512_ddp2_legacyval.yaml
```

The BS16/C512 shape requires the expandable CUDA allocator to avoid allocator
fragmentation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

A real three-step DDP2 smoke passed train, legacy validation, gradient sync,
and both rank-local checkpoint states. Steady measurements exclude the first
warmup step:

| Shape | Steady global tok/s | Step time | Peak allocated/rank | 5B pure train |
| --- | ---: | ---: | ---: | ---: |
| DDP2, local BS16, C512, U1 | 15,996-16,555 | 3.96-4.10 s | 123.7 GiB | 3.56 days |
| DDP4, local BS8, C512, U1 | 24,563-24,896 | 2.63-2.67 s | 51.4 GiB | 2.35 days |

DDP2 retains about 66% of DDP4 throughput with half the GPUs. Its mean
throughput per GPU is approximately 8.14k tok/s versus 6.18k tok/s for DDP4,
making DDP2 about 32% more efficient per allocated GPU. DDP2 is therefore the
recommended default when GPU efficiency matters; DDP4 remains the option when
reducing wall time by roughly 1.2 days is worth occupying two additional GPUs.

## 8. Decision

Implementation, smoke acceptance, and the bounded 250M quality pilots are
complete. At matched C128 FB semantics, final reasoning exact is 27.67%,
16.00%, 37.67%, and 31.83% for U1, U2, U4, and U8 respectively. Public S600 is
effectively flat across the arms, while PPL differs by less than 2%.

U4 is selected for the first formal 5B FB run. It exceeds U8 by 5.83 reasoning
points while using 17.5 GiB instead of 29.5 GiB allocated per rank. U16 remains
held after smoke validation. The formal 5B launch must extend the optimized
Triton C128-FB-U1 configuration and override only
`stateful_tbptt.detach_interval_chunks: 4`; the older U4 configuration uses the
pre-Triton backend.

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

Throughput-oriented CPBC-DP training on four B200s:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train.py \
  --config configs/train/cpbc_r125_5b_dp_u1_c512_ddp4_legacyval.yaml
```

Resource-efficient CPBC-DP training on two B200s:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src:. \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/cpbc_r125_5b_dp_u1_c512_ddp2_legacyval.yaml
```

Single-rank memory calibration with explicit U:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml \
  --batch-size 16 --sequence-length 2048 --chunk-size 128 \
  --detach-interval-chunks 4
```

## 10. Prioritized Ablation Plan

### 10.1 Running Anchors

These runs establish the matched end-to-end reference, but they do not isolate
CPBC visibility, chunk size, or gradient horizon:

| ID | Model | Configuration | Status | Purpose |
| --- | --- | --- | --- | --- |
| A0 | plain Transformer | balanced 5B, DDP2, global BS 32, legacy val | running | matched architecture baseline |
| A1 | BDRE CPBC-DP | C512, U1, balanced 5B, DDP2, global BS 32, legacy val | running | current full BDRE anchor |

The A0/A1 comparison must report quality and route/cache behavior together
with throughput. It must not be described as a DP-versus-FB ablation.

### 10.2 P0: Execution Optimization Ablations

Run these as fixed-weight, warmed 20-100-step benchmarks after A0 releases
GPUs 2-3. They are engineering equivalence tests, not long quality runs.

| ID | Change from current explicit backend | Primary measurement |
| --- | --- | --- |
| E0 | current `shared_padded_explicit` reference | reference logits, loss, gradients, kernels, memory, tok/s |
| E1 | GPU-resident route packing/group construction | remove route-step host synchronization |
| E2 | grouped free-block projections/MLP execution | reduce small per-block launches |
| E3 | fused BDRE reader attention | remove decoded score/mask/softmax/einsum intermediates |
| E4 | E1 + E2 + E3 integrated | end-to-end speedup and peak memory |

Each optimized arm must match E0 on forward values and gradients within a
declared BF16 tolerance, and pass suffix-causality and streamed-state tests.
Chunk size may be benchmarked for speed, but it cannot be changed inside a
quality comparison and then attributed to another variable.

### 10.3 P1: Core CPBC Quality Ablations

Do not launch these at 5B per arm. First use a 250M-token pilot (3,815 optimizer
steps at 65,536 tokens/step) with checkpoints and the full evaluation contract
near one-third, two-thirds, and the end.

| ID | Controlled comparison | Question answered |
| --- | --- | --- |
| Q1 | DP-U1 vs FB-U1 at the same chunk size | Does completed-history full-bank visibility help? |
| Q2 | U1 vs U2 vs U4 under the Q1-winning visibility, same chunk | How much cross-chunk gradient horizon is useful? |
| Q3 | C128-U4 vs C512-U1 under one visibility | At a matched 512-token horizon, how much does prefill granularity matter? |
| Q4 | exact BDRE-Serial vs selected CPBC policy on a bounded short-context diagnostic | What approximation error remains relative to the serial oracle? |

Q1 is the first required quality ablation. Q2 is adaptive: reuse the winning
Q1 U1 arm, then add only U2 and U4. Q3 must not be interpreted as a pure U
ablation; it deliberately measures chunk-boundary approximation at a matched
maximum gradient horizon. Q4 remains a diagnostic because exact serial
training is not computationally viable at 5B.

### 10.4 P2: BDRE Semantic Ablations

Run these one at a time from the best P1 configuration. They must not be
expanded into a Cartesian sweep.

| Priority | Controlled comparison | Trigger / purpose |
| --- | --- | --- |
| S1 | `bdre_prefix` self K/V vs `current_step` self K/V | required; tests whether accumulated self memory is a shortcut |
| S2 | depth scoring off vs `reader_step` depth scoring | required; separates compiler depth scoring from CPBC visibility |
| S3 | compiler position term on vs off | required; tests whether position-conditioned fusion helps or dominates |
| S4 | compile-all vs hard top-k 8 | only after cache-weight entropy is healthy; tests sparsification |
| S5 | K/V temperatures 0.5/1.0 vs 1.0/1.0 | only if compiler entropy or writer domination is abnormal |

`self_kv_mode: none`, late-step bias, top-k 12/4, and temperature sweeps are
diagnostics only. They should not consume long-run budget without a preceding
metric showing the corresponding failure mode.

### 10.5 Fixed Contract and Promotion Rule

All quality arms use balanced training data, the legacy validation split,
sequence length 2,048, global batch 32, the same seed, optimizer, router
settings, token order, and evaluation points. The decision table must include:

```text
legacy val loss and PPL
reasoning S600 exact and teacher accuracy
public S600 average and task breakdown
route entropy, path diversity, block coverage, and route length
compiler entropy, writer mass, self-memory mass, and cache norms
throughput, GPU utilization, and peak CUDA memory
```

PPL alone cannot promote an arm. Promote only a 250M winner to 2B; promote to
5B only if the 2B checkpoints show a consistent public/reasoning improvement
without route or cache takeover. The existing A1 run remains the 5B anchor, so
there is no justification for a full 5B grid.

### 10.6 Deferred Existing Knobs

Slow noise, selective balance, coverage floor, self-recurrence cap, route
length limits, and disabled location bias already come from the accepted
route-core configuration. Do not re-ablate them during the CPBC/BDRE semantic
study. Ain/anchor combinations, generic top-1/top-2 comparisons, 16/32-block
scaling, and fine-grained block sizing remain separate future packages.
