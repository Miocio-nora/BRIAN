# RC-KV Grouped Reader and Incremental Prefix Report

**Date:** 2026-07-16

**Status:** implemented and retained as the accepted predecessor

**Branch:** `rc-kv-compact-reader`

**Successor:**
[`rc_kv_static_route_step_report.md`](./rc_kv_static_route_step_report.md)

## 1. Scope

This work remains inside `BRIAN-R125-BDRE-RCKV-v1`. It does not introduce a
new model version or change:

- RC-KV writer or reader parameter ownership;
- CPBC-DP cache visibility;
- route decisions, hard top-1 routing, or maximum route depth;
- causal token visibility or RoPE placement;
- the C2048/U1 gradient horizon;
- local BS16 or DDP2 global BS32;
- loss definitions or training data.

The accepted predecessor used one exact padded FlexAttention call per active
reader action and rebuilt each token's complete writer prefix at every route
step. This follow-up targets those two repeated costs.

## 2. Implemented Optimization

### 2.1 Bounded grouped-reader BlockMask

`execution.flex_reader_group_size: 8` batches up to eight independent reader
actions into one FlexAttention call. Reader and head dimensions are represented
as batched tensor axes; no projection is shared. Every reader still owns its
original Key decoder, Value reader, route block, and attention parameters.

Queries are padded only inside the bounded group. The exact mask requires both:

```text
query_slot_is_valid
key_position <= absolute_query_position
```

This prevents padded rows from becoming valid position-zero queries. In the
measured C2048 route, the change reduced observed reader-attention calls from
120 to 15 per complete training forward/backward.

Key decode plus decoded-Key RoPE and the final Value read remain eager. An
attempt to compile them together with FlexAttention triggered an Inductor
backward `CantSplit` failure. Compiling the attention kernel alone is stable and
faster, so the failing combined graph was not retained.

### 2.2 Algebraically exact online prefix compiler

The default `recompute` compiler remains available. The new explicit option is:

```yaml
execution:
  prefix_compile: incremental_exact
```

For synchronous depth-prefix compilation, every visible writer step `s`
satisfies `s <= l`, where `l` is the current reader step. Therefore:

```text
-lambda * abs(l - s) / scale
= -lambda * l / scale + lambda * s / scale
```

The first term is common to every writer visible at reader step `l` and cancels
inside softmax. The compiler can consequently update one stable softmax state
per token and reader when each writer arrives. It stores FP32 running maxima,
denominators, and weighted numerators independently for Key and Value
temperatures.

The result is algebraically equivalent to unrestricted full-prefix
recomputation. Floating-point accumulation order differs, so acceptance uses
explicit numerical tolerances rather than bitwise equality. The autograd graph
is retained through the recurrence; this is not a detached cache shortcut.

Hard compiler top-k is rejected with `incremental_exact`, because changing the
support set cannot be represented by this simple online recurrence. Detailed
compiler diagnostics and visualization require full writer weights, so those
modes automatically fall back to `recompute`.

### 2.3 Kernel calibration

The current kernel variant is:

```yaml
execution:
  flex_kernel_variant: bwd32_fwd32
```

It sets FlexAttention forward and backward M/N tiles to 32. Auto, forward-64,
forward-128, and mixed M/N candidates were measured and rejected for this
BS16/T2048 shape. This is a shape-specific implementation choice, not a model
change.

## 3. Correctness Acceptance

The implementation passed:

- online compiler outputs and gradients against full-prefix recomputation;
- full-model logits, loss, and representative writer/position gradients;
- BF16 grouped BlockMask logits, loss, and gradients against the score-mod
  reference;
- suffix invariance for CPBC-DP and CPBC-FB;
- full-forward versus streamed execution;
- single-process versus DDP2 legacy evaluation;
- a two-rank training, backward, gradient-sync, evaluation, and checkpoint
  smoke;
- 50 focused BDRE tests;
- all 612 repository tests.

Single-process and DDP2 evaluation of the same candidate and legacy batch both
produced:

```text
validation_loss = 115.0297622680664
```

Route and cache statistics also matched. DDP changes only batch partitioning
and performance, not model semantics.

## 4. Single-B200 Calibration

All rows use R125 BF16, local BS16, sequence/chunk length 2,048, U1, one warmup,
and a complete forward/backward step.

| Candidate | Median tok/s | Peak allocated | Relative to predecessor |
| --- | ---: | ---: | ---: |
| Group 1, recompute, `bwd32` | 48,913 | 66,018 MiB | 1.000x |
| Group 2, recompute, `bwd32` | 50,023 | 69,059 MiB | 1.023x |
| Group 4, recompute, `bwd32` | 51,760 | 69,057 MiB | 1.058x |
| Group 8, recompute, `bwd32` | 53,139 | 69,083 MiB | 1.086x |
| Group 8, online prefix, `bwd32` | 55,478 | 66,938 MiB | 1.134x |
| Group 8, online prefix, `bwd32_fwd32` | **63,078** | **66,938 MiB** | **1.290x** |

The stable predecessor's separately accepted median was 48,956 tok/s. Against
that value, the current candidate gains 28.8%. Against the original C2048
score-mod reader at 23,246 tok/s, it is 2.71x faster.

The final candidate's measured total loss was `114.6927109`, versus
`114.696228` for the earlier accepted BlockMask run. The small difference is
consistent with BF16 kernel accumulation order and remains within the tested
forward/gradient tolerances.

## 5. DDP2 Acceptance

The two-B200 smoke uses local BS16 on each rank, global BS32, 65,536 tokens per
optimizer step, C2048, U1, and no gradient accumulation.

| Steady step | Global tok/s | Step time | Peak allocated per rank |
| --- | ---: | ---: | ---: |
| 2 | 126,878 | 0.5165 s | 70,926 MiB |
| 3 | 129,710 | 0.5052 s | 70,791 MiB |
| Mean | **128,294** | - | - |

The predecessor averaged 94,114 tok/s, so the DDP2 gain is 36.3%. The matched
plain Transformer baseline averaged 618,738 tok/s:

```text
618,738 / 128,294 = 4.82x remaining throughput gap
```

At the measured steady rate, five billion pure training tokens require about
10.8 hours. This excludes compilation, evaluation, checkpoints, public
benchmarks, W&B, and data stalls.

## 6. Profile Change

| Metric | Predecessor | Current candidate |
| --- | ---: | ---: |
| Observed reader-attention calls | 120 | 15 |
| CUDA kernel launches | about 16,771 | about 9,346 |
| FlexAttention forward self CUDA | - | 30.1 ms |
| FlexAttention backward self CUDA | - | 67.2 ms |
| Total profiled self CUDA | - | 421.9 ms |
| Attention forward + backward share | dominant | about 23.1% |

The bottleneck is no longer only attention backward. The largest remaining
families include multiplication, copy, elementwise, grouped-MM, concatenation,
packing, and the repeated route loop. A further exact speedup therefore needs
to reduce intermediate materialization and launch fragmentation across a route
step, not only tune the attention kernel again.

## 7. Prepared Configurations

Current model configuration:

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_grouped_mm_flex_blockmask_group8_incremental.yaml
```

Formal DDP2 training configuration:

```text
configs/train/cpbc_r125_5b_dp_u1_c2048_grouped_mm_flex_blockmask_group8_incremental_ddp2_legacyval.yaml
```

Accepted smoke configuration:

```text
configs/train/smoke_cpbc_r125_5b_dp_u1_c2048_grouped_mm_flex_blockmask_group8_incremental_ddp2_legacyval.yaml
```

## 8. Limits and Next Engineering Target

- No formal long training or downstream quality comparison was started in this
  optimization pass.
- The measured 4.82x gap is not established as a hardware or mathematical
  limit. It is the current implementation gap.
- Online prefix compilation intentionally excludes hard compiler top-k.
- Reader groups still use bounded padding and host-created action groups.
- The final completed-token bank still performs one full compiler pass because
  it has a different, reader-step-independent output contract.
- Diagnostics fall back to full prefix recomputation, so diagnostic steps are
  slower than normal training steps.

The next exact optimization should preserve all current acceptance invariants
while fusing or reorganizing route-step packing, canonical writer updates,
reader projection materialization, and scatter/gather work. A new model version
is not justified unless its mathematical cache or routing definition changes.
