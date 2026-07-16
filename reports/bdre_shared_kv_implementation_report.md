# BRIAN RC-KV / CPBC Implementation Report

**Status:** RC-KV exact oracle, CPBC approximate prefill, stateful TBPTT, and
stateful DDP are implemented and validated

**Date:** 2026-07-16

**Working model:** `BRIAN-R125-BDRE-RCKV-v1`

**Current branch:** `bdre-synchronous-prefix-ddp`

**Academic report:**
[`BRIAN_RC_KV_Implementation_Report.tex`](./BRIAN_RC_KV_Implementation_Report.tex)

## 1. Executive Summary

This document is the implementation-grounded contract for BRIAN's
Reader-Compiled KV Cache (RC-KV). It supersedes the earlier design-only state
of `BRIAN_Free_Routing_Shared_KV_Cache.txt`.

Free routing removes the fixed layer identity used by an ordinary Transformer
KV cache. Different tokens can visit different blocks, revisit a block, and
exit at different route depths. RC-KV therefore separates memory into three
levels:

1. block/head-specific attention projections produce local K/V;
2. block-specific writer projections map local K/V into canonical latent
   spaces;
3. the BDRE compiler turns a token's writer route into a reader-specific cache
   object, which each reader block/head interprets with its own decoder.

"Shared KV" means that readers use a common canonical cache interface. It does
not mean that Q/K/V projections, writer canonicalizers, or reader decoders are
shared across blocks.

The implementation now has two distinct execution contracts:

- **BDRE-Serial:** exact token-by-token execution. It is the causal and
  incremental-inference oracle and remains permanently available.
- **Chunkwise Progressive Bank Completion (CPBC):** an approximate prefill
  procedure in which each chunk is processed in parallel across tokens while
  each token's cache bank is progressively completed over recurrent route
  steps.

CPBC makes training feasible, but it is not presented as mathematically
identical to serial execution. Its depth visibility policy is explicit:

- **CPBC-DP:** CPBC with Depth-Prefix Visibility.
- **CPBC-FB:** CPBC with Full-Bank Depth Visibility.

The system has passed causal, incremental, cache-policy, gradient, and DDP
correctness tests. The unresolved limitation is execution speed. The current
two-B200 CPBC-DP run reaches about `21.6k token/s`, while the matched baseline
reaches about `619k token/s`; this is an end-to-end gap of about `28.7x`.
These measurements do not establish a downstream quality conclusion.

Existing Non-Global, hidden-state Global KV, attention-summary Global KV, and
pure-factorized cache-only implementations remain available and unchanged.
RC-KV is additive and selected through a separate model configuration.

## 2. Implemented Surface

Primary implementation files:

```text
src/brian_sphere_llm/model/bdre_model.py
src/brian_sphere_llm/memory/bdre_shared_kv.py
src/brian_sphere_llm/routing/block_position.py
src/brian_sphere_llm/eval/bdre_cache_visualization.py
src/brian_sphere_llm/train/stage_runner.py
src/brian_sphere_llm/train/trainer.py
tests/test_bdre_shared_kv.py
tests/test_stateful_tbptt.py
tests/test_stateful_ddp.py
```

Core model configurations:

```text
configs/model/brian_r125_bdre_rckv_v1.yaml
configs/model/brian_r125_bdre_rckv_v1_tbptt.yaml
configs/model/brian_r125_bdre_rckv_synchronous_prefix.yaml
configs/model/brian_r125_bdre_rckv_synchronous_prefix_shared_explicit.yaml
configs/model/brian_r125_bdre_cpbc_dp_shared_explicit.yaml
configs/model/brian_r125_bdre_cpbc_dp_c512_shared_explicit.yaml
configs/model/brian_r125_bdre_cpbc_fb_shared_explicit.yaml
```

Current training and ablation configurations:

```text
configs/train/cpbc_r125_5b_dp_u1_c512_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_dp_u1_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u1_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u2_c128_ddp2_legacyval.yaml
configs/train/cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml
configs/train/bdre_rckv_r125_5b_ddp2_legacyval.yaml
configs/train/bdre_rckv_r125_5b_tbptt_bs32_legacyval.yaml
```

Related engineering reports remain useful for historical calibration details:

```text
reports/bdre_synchronous_prefix_prefill_report.md
reports/bdre_stateful_tbptt_implementation_report.md
reports/bdre_synchronous_prefix_kernel_optimization_report.md
reports/cpbc_prefill_ddp_ablation_report.md
reports/route_sphere_terminal_dashboard.md
```

## 3. Current R125 Scope

| Item | Implemented value |
| --- | ---: |
| Actual parameter count | `140,308,105` |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Fixed pre blocks | 2 |
| Free route blocks | 8 |
| Exit Block | 1 |
| Fixed post blocks | 2 |
| Maximum route steps | 16 |
| Route action | hard top-1 |
| Block position dimension | 64 |
| Canonical Key dimension | 32 |
| Canonical Value dimension | 32 |
| Current context length | 2048 |

RC-KV applies only to the eight free route blocks. Fixed pre/post blocks retain
standard per-layer autoregressive KV state because their layer identities are
stable. OUT is a route action and terminal position, but it does not write a
canonical cache object.

When a token exits, its hidden state is frozen. Other active tokens continue
routing; the exited token only occupies masked/padded positions until the
route loop ends.

## 4. Parameter Ownership and Writer Codes

For free block `b` and head `h`, all attention projections are independent:

```text
W_Q[b,h], W_K[b,h], W_V[b,h]
```

The code may pack them into combined PyTorch linear layers, but their
mathematical ownership remains block- and head-specific.

For token `i` at route step `s`, the normalized attention input is:

```text
x[i,s] = RMSNorm(hidden[i,s] + position_adapter(z[b[i,s]]))
q[i,s,h] = W_Q[b[i,s],h] x[i,s]
k[i,s,h] = W_K[b[i,s],h] x[i,s]
v[i,s,h] = W_V[b[i,s],h] x[i,s]
```

Before temporal RoPE, the selected writer block maps concatenated K and V
heads into separate canonical spaces:

```text
cK[i,s] = A_K[b[i,s]] Concat_h(k[i,s,h])
cV[i,s] = A_V[b[i,s]] Concat_h(v[i,s,h])
```

`A_K[b]` and `A_V[b]` are independent, bias-free, block-specific linear
maps. They are not shared across free blocks. Canonical coordinates are
learned jointly; no fixed external basis is assumed.

Each reader block and head also owns independent decoders:

```text
B_K[reader,h]: R^rK -> R^dh
B_V[reader,h]: R^rV -> R^dh
```

## 5. Spherical Position Geometry

The implementation maintains `IN + 8 internal blocks + OUT` learnable unit
position vectors. Initialization uses antipodal IN/OUT poles and a regular
simplex for the internal blocks in the orthogonal subspace:

```text
dot(z_IN, z_OUT) = -1
dot(z_IN, z_block) = dot(z_OUT, z_block) = 0
dot(z_block_i, z_block_j) = -1/7, i != j
```

A deterministic seeded orthogonal rotation is applied after construction so
the geometry is not tied to coordinate axes. Positions are normalized when
read and remain trainable.

- IN initializes router state and first-step position injection.
- Internal positions serve the router, block adapters, and BDRE compiler.
- OUT represents termination and is injected into the Exit Block.
- IN and OUT do not index reader cache entries.

A weak Gram-matrix loss with weight `0.001` discourages geometric collapse.
It does not define a nearest-neighbor path and is not a route-imitation loss.

## 6. BDRE Reader Compilation

For reader block `r`, reader step `l`, and historical writer step `s`, the
implemented score is:

```text
e[r,l,s] = tau_z * dot(z_reader[r], z_writer[s])
           - lambda_s * depth_distance(l,s)
           + gamma * (s + 1) / max_route_steps
```

The formal CPBC profile uses normalized depth distance:

```text
depth_distance(l,s) = abs(l - s) / (max_route_steps - 1)
```

The three score terms can be independently disabled:

- block-only compilation uses `lambda_s = 0` and `gamma = 0`;
- reader-step compilation enables `lambda_s > 0`;
- late-step bias enables `gamma > 0` and remains an ablation only.

Only valid writer steps enter the candidate set. Optional hard compile top-k
is applied before softmax. Key and Value use the same support but independent
temperatures:

```text
alphaK = softmax(e / T_K)
alphaV = softmax(e / T_V)
mK[i,r,l] = Sum_s alphaK[r,l,s] * cK[i,s]
mV[i,r,l] = Sum_s alphaV[r,l,s] * cV[i,s]
```

Current defaults are `T_K=0.5`, `T_V=1.0`, `tau_z=1.0`, and no hard compile
top-k. The current formal CPBC profile uses `lambda_s=0.25` and `gamma=0`.

### 6.1 Persistent State Variants

| Mode | Main K/V state | Approx. BF16 K+V per token | Semantics |
| --- | --- | ---: | --- |
| Block-only | `[B,T,R,rX]` | 1 KiB | one cache pair per completed token and reader block |
| Reader-step eager | `[B,T,L,R,rX]` | 16 KiB | compile every reader step when the token completes |
| Reader-step lazy | block cache + writer history + memo | grows on access | compile `(reader,step)` at first read and memoize |
| Reader-step dynamic | `[B,T,Smax,rX]` writer history | about 2 KiB | recompute fusion on every read; reference/cost mode |

CPBC requires eager, step-indexed banks. Exact reader-step execution supports
eager, lazy, and dynamic policies and verifies their numerical equivalence.

## 7. Reader Attention and Current-Token Memory

For a historical canonical object, the exact Key path decodes into each
reader-head coordinate system and then applies token-position RoPE:

```text
k_hat[i,reader,h] = B_K[reader,h](mK[i,reader])
q_rope[t,h] = RoPE(q[t,h], position=t)
k_rope[i,h] = RoPE(k_hat[i,reader,h], position=i)
```

Value aggregation uses a strictly equivalent low-dimensional path:

```text
value_latent[h] = Sum_i attention_weight[i,h] * mV[i,reader]
output_head[h] = B_V[reader,h](value_latent[h])
```

The Value decoder can therefore run after the weighted sum. The Key decoder
cannot currently be removed because each historical Key receives a different
RoPE rotation. Any latent-relative-RoPE optimization must first prove exact
equivalence against this decoded-Key reference.

The exact backend implements three current-token self-KV policies:

- `bdre_prefix`: compile the current token's writer prefix into one temporary
  self object; this is the default.
- `current_step`: expose only the current writer step's K/V.
- `none`: attend only to historical tokens `i < t`.

`bdre_prefix` does not expose every route step as a separate attention object.
Current-step K/V are generated from the pre-attention input, so the operation
does not create a circular dependency. CPBC currently requires
`bdre_prefix`.

## 8. Exact Execution: BDRE-Serial

BDRE-Serial is sequence-serial and batch-parallel. For token position `t`:

1. fixed pre blocks process token `t` using standard incremental KV;
2. position starts at IN and the router selects one internal block or OUT at
   each route step;
3. active tokens are grouped by selected reader block;
4. the reader uses completed caches from tokens `< t` plus the configured
   current-token self object;
5. each valid route step appends one canonical writer code;
6. after route completion, token `t` is compiled into persistent reader
   cache state;
7. the Exit Block and fixed post blocks process token `t`, producing LM logits.

`forward_incremental` directly advances and returns pre/post KV plus RC-KV
state. Teacher-forced exact `forward` calls the same incremental kernel for
each token and does not recompute prefixes. This backend is the correctness
oracle for causality, generation, and CPBC comparisons.

Exact execution is not a practical 5B-token training backend. On one B200,
measured R125 BF16 full forward/backward throughput was about `25-28 token/s`
for the calibrated sequence lengths.

## 9. CPBC Approximate Prefill

**Chunkwise Progressive Bank Completion (CPBC)** is an approximate prefill
procedure in which each chunk is processed in parallel across tokens, while
the per-token cache bank is progressively completed over recurrent steps.

For chunk size `C`, token positions are parallel on the token axis and serial
on route depth. At reader step `l`:

1. fixed pre blocks run standard causal attention over the chunk;
2. the router independently selects an action for every active `[batch,token]`;
3. tokens are grouped by selected reader block for Q/K/V, writer projection,
   reader attention, and FFN;
4. writer codes with valid step `s <= l` compile the current step bank;
5. each query attends to completed history plus current-chunk positions
   `i <= t` under a strict token-causal mask;
6. exited tokens freeze while remaining tokens advance to `l+1`;
7. completed chunk state is committed before the next chunk begins.

CPBC preserves token causality but changes when a completed route becomes
visible. It has two implemented depth visibility policies.

### 9.1 CPBC-DP: Depth-Prefix Visibility

Both active and completed tokens retain only writer support `s <= l` for
reader depth `l`:

```text
visible_DP(i,l) = {s | writer_valid[i,s] and s <= l}
```

This policy is invariant to changing chunk boundaries at fixed weights: a
reader depth never gains later writer steps merely because a token moved into
an earlier completed chunk.

### 9.2 CPBC-FB: Full-Bank Depth Visibility

The active chunk still uses the progressive prefix available at reader step
`l`. Once a chunk completes, every reader depth is recompiled from the token's
full valid writer route:

```text
visible_FB(i,l) = {s | writer_valid[i,s]}  for completed tokens
```

CPBC-FB is closer to completed-route serial semantics, and at `chunk_size=1`
it matches the BDRE-Serial completed-route reference within the tested BF16
tolerance. It is still chunk-boundary dependent for active-chunk history and
must not be described as exact token-serial prefill.

## 10. Stateful TBPTT and DDP

Let `C` be chunk size and `U` be the number of chunks retained in one autograd
graph before cache state is detached. The maximum gradient horizon is:

```text
H_grad = min(sequence_length, C * U)
```

Cache values survive a detach boundary, but later losses no longer assign
credit to writers before that boundary. Chunk losses in one group are summed
before backward. Optimizer parameters are not updated until all chunks in the
sequence have completed, so one sequence never observes two parameter
versions.

LM loss is normalized by the full sequence's valid next-token labels. The sum
of chunk losses therefore equals the full-sequence mean loss. At fixed model
weights, changing `U` does not change forward values; it changes the backward
graph and gradients.

DDP divides the batch only. Every rank owns independent RC-KV and pre/post KV
state. Stateful forwards execute under `no_sync()`, followed by one explicit
gradient synchronization per optimizer update:

1. all-reduce a parameter-used mask;
2. create zero gradients for parameters used globally but not on this rank;
3. bucket gradients by dtype/device;
4. all-reduce and divide by world size.

The current DDP2 profile synchronizes about `561 MB` of gradients in nine
`64 MB`-bounded buckets per update. CPU merged-global-batch tests verify loss,
gradient, and parameter-update equivalence.

## 11. Current Formal Training Profile

The exact oracle defaults and the current formal training profile are
different. The active 5B profile is resolved from:

```text
configs/model/brian_r125_bdre_cpbc_dp_c512_shared_explicit.yaml
configs/train/cpbc_r125_5b_dp_u1_c512_ddp2_legacyval.yaml
```

| Setting | Formal value |
| --- | --- |
| Execution | `synchronous_prefix` (CPBC) |
| Visibility | `depth_prefix` (CPBC-DP) |
| Attention backend | `shared_padded_explicit` |
| Dispatch | `grouped_host` |
| Chunk size `C` | 512 |
| TBPTT detach interval `U` | 1 |
| Hardware | 2 B200 GPUs |
| Local batch per rank | 16 |
| Global batch | 32 |
| Context length | 2048 |
| Tokens per optimizer update | 65,536 |
| Precision | BF16 |
| Learning rate | `3e-4` |
| Token budget | 5B |
| Validation compatibility | legacy validation profile |

Global batch is intentionally held at 32 when changing DDP world size. Batch
per rank changes to preserve this experiment-level control.

The current routing recipe uses:

```text
top-1 routing
maximum route steps: 16
minimum exit step: 4
exit ramp start: 12
force final OUT: enabled
self-recurrence consecutive cap: 2
router probability curriculum: 0 -> 1 over steps 0..1500
route imitation loss: 0
logit noise: 0.08 -> 0.008
random route override: 0.25 -> 0.03
```

The active objective is:

```text
L = L_LM
    + 0.0002 * L_cost
    + 0.02   * L_selected_balance
    + 0.05   * L_coverage
    + 0.02   * L_exit
    + 0.001  * L_position_geometry
```

Location bias/loss, plain balance, transition diversity, and route imitation
have zero weight. This routing recipe is an experiment control, not part of
the mathematical definition of RC-KV.

## 12. Implemented Options and Guardrails

| Mechanism | Implemented values | Formal value |
| --- | --- | --- |
| Execution | `token_by_token`, `synchronous_prefix` | `synchronous_prefix` |
| Depth mode | `none`, `reader_step`, `synchronous_prefix` | `synchronous_prefix` |
| Visibility | `depth_prefix`, `full_bank` | `depth_prefix` |
| Reader cache | `eager`, `lazy`, `dynamic` | `eager` |
| Self-KV | `bdre_prefix`, `current_step`, `none` | `bdre_prefix` |
| Reader attention | `per_query_reference`, `shared_padded_explicit` | `shared_padded_explicit` |
| Dispatch | `legacy_cuda_scan`, `grouped_host` | `grouped_host` |
| Compiler top-k | `null` or positive integer | `null` |
| Key/Value temperatures | positive and independent | `0.5 / 1.0` |
| Position/depth/late weights | non-negative | `1.0 / 0.25 / 0` |
| TBPTT detach interval | `U >= 1` | `1` |

Hard validation rules include:

- RC-KV v1 supports route top-1 only.
- CPBC requires `hard_exit=true`, `bdre_depth_mode=synchronous_prefix`, eager
  reader-step banks, and `bdre_prefix` self-KV.
- Full-bank visibility is legal only under CPBC.
- `shared_padded_explicit` is legal only under CPBC.
- Position mode must be spherical code with an independent IN position.
- RC-KV cannot be combined with legacy hidden Global KV, attention-summary
  Global KV, or parallel passing in the same model.
- Top-2 reader fusion is not implemented in RC-KV v1.

## 13. Correctness Evidence

| Check | Result |
| --- | --- |
| Exact full forward vs stateful incremental | maximum logits difference `0` |
| Exact streamed chunks vs unchunked exact forward | forward and loss agree |
| Reader-step eager/lazy/dynamic | agree at `atol=rtol=1e-6` |
| Prefix suffix-invariance | passed |
| Future-writer intervention | future tokens/steps do not affect prefix |
| CPBC-FB chunk 1 vs BDRE-Serial | reference and shared-explicit agree within `2e-5` |
| CPBC-DP chunk-boundary invariance | passed |
| CPBC full forward vs streamed chunks | agree within `2e-5` |
| CPBC full forward vs one-token incremental | agree within `2e-5` |
| `U=1` vs `U=2` | same forward loss; gradients differ as intended |
| Shared-padded-explicit vs per-query reference | logits, loss, cache, and gradients agree |
| CPU DDP vs merged global batch | loss, gradients, and updates agree |
| Rank-local unused parameter synchronization | passed |
| B200 BF16 forward/backward and DDP2 smoke | finite; checkpoint state passed |
| Full repository regression | `598 passed`, `15` pre-existing warnings |

These checks establish the implementation and causal interface. They do not
show that CPBC-DP, CPBC-FB, or RC-KV improves language modeling or public
benchmarks. Model selection must combine PPL, reasoning/public suites, route
entropy, path diversity, block coverage, compiler entropy, writer mass, and
cache norms.

## 14. Performance Measurements

| Backend | BS | Sequence | Chunk | Throughput | Peak allocated | Context |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Exact token-serial | 32 | 128 | 8 | 28.18 token/s | 3.28 GiB | oracle |
| Exact token-serial | 32 | 256 | 8 | 25.53 token/s | 4.46 GiB | oracle |
| CPBC per-query reference | 8 | 2048 | 256 | 1,467.80 token/s | 111,353 MiB | median |
| CPBC shared-padded-explicit | 8 | 2048 | 256 | 4,695.64 token/s | 27,333 MiB | median |
| CPBC shared-padded-explicit | 14 | 2048 | 512 | 11,658.95 token/s | 101,808 MiB | stable single-GPU candidate |
| Formal CPBC-DP DDP2 | 32 global | 2048 | 512 | 21,577 token/s | 55.6 GiB/rank | active-run snapshot |

The formal row is the latest-100-step snapshot captured on 2026-07-16, not an
isolated kernel benchmark. Median throughput was `21,646 token/s`, with a
`20,549-22,190 token/s` range. The matched Transformer baseline averaged
`618,738 token/s`, producing an end-to-end gap of about `28.7x`.

The two models do not execute the same number of block evaluations, so this is
not a pure attention-kernel ratio. The gap is nevertheless much larger than
route length alone can explain.

### 14.1 Remaining Hotspots

Profiling and code inspection identify the dominant costs:

1. each route step copies selected actions from GPU to CPU for host grouping;
2. routing scatters tokens across eight blocks, creating many small QKV,
   writer-projection, FFN, and index-copy kernels;
3. the exact reader path explicitly decodes Keys, builds causal masks, performs
   FP32 softmax, and launches multiple einsums;
4. one chunk still launches about `28,600` kernels, leaving dispatch bubbles;
5. `C=512,U=1,T=2048` executes four backward groups per optimizer step and then
   synchronizes about `561 MB` of gradients;
6. step-indexed banks, route-dependent padding, and variable group shapes add
   memory traffic.

The BDRE compiler arithmetic is usually a small fraction of wall time. The
next performance work should target GPU-resident grouping, grouped expert
GEMM, and fused reader attention while preserving logits, loss, cache, and
gradient equivalence.

## 15. Metrics and Visualization

Training/evaluation telemetry includes or is expected to preserve:

```text
bdre_compile_time_ms
bdre_compile_fraction
bdre_self_prefix_time_ms
bdre_self_prefix_fraction
bdre_compiled_state_count
bdre_lazy_cache_hit_rate
bdre_writer_steps_mean/min/max
bdre_key_weight_entropy
bdre_value_weight_entropy
bdre_selected_writer_steps
bdre_last_step_mass
bdre_position_gram_error
bdre_internal_position_min_angle
bdre_cache_memory_mb
```

Visualization covers spherical block positions, real sampled token routes,
writer-step trajectories, per-reader Key/Value compile weights, top-k support,
and last-step concentration. The detached terminal telemetry path feeds the
text-free animated Route Sphere under `tools/route_sphere_tui/` without keeping
training autograd graphs alive.

## 16. Controlled Ablation Order

The matched baseline and CPBC-DP C512/U1 5B runs are anchor runs, not a complete
visibility ablation. Mechanism tests should proceed in this order:

1. compare CPBC-DP and CPBC-FB at identical `C`, `U`, global batch, data, and
   token budget;
2. on the winning visibility policy, compare `U=1,2,4`;
3. compare C128/U4 with C512/U1 at the same 512-token gradient horizon;
4. from the winning setup, separately compare `bdre_prefix` vs `current_step`,
   depth score on/off, and compiler position on/off;
5. activate hard compile top-k or temperature ablations only if compiler
   entropy or writer domination becomes abnormal.

Initial arms should use 250M tokens and run legacy validation, reasoning S600,
and public S600 at roughly 1/3, 2/3, and full progress. Only configurations
that jointly preserve capability, route/cache stability, and performance
should advance to 2B. A Cartesian product of 5B runs is not justified.

## 17. Limitations and Guardrails

### 17.1 Training Speed

CPBC improves serial throughput by hundreds of times, but remains about one
order of magnitude or more behind the matched baseline. This is the primary
practical limitation and makes broad long-run ablations expensive.

### 17.2 CPBC Is an Explicit Approximation

CPBC-DP is chunk-boundary invariant but permanently restricts completed reader
depth `l` to writer prefix `s <= l`. CPBC-FB gives completed tokens full-route
visibility but remains progressively incomplete inside the active chunk.
Training PPL alone cannot choose between them.

### 17.3 TBPTT Truncation

The formal C512/U1 profile has a 512-token gradient horizon. Cache values remain
available across the sequence, but later chunks cannot assign gradient credit
to writers before the detach boundary. Increasing `U` raises activation memory
quickly; the quality benefit has not yet been measured.

### 17.4 Compression and Reader Takeover

`rK=rV=32` is strong compression relative to `d=768`. Writer canonicalizers
must learn a coherent shared interface, while position/depth score terms,
self-prefix memory, and reader decoders can create a memory shortcut. Compiler
entropy, writer mass, self-memory concentration, cache norms, and public
benchmarks must be diagnosed together.

### 17.5 RoPE and Routing Scope

Position-dependent RoPE currently prevents a fully latent Key-scoring path.
RC-KV v1 also supports only hard top-1 routing, eight free blocks, and 16
maximum route steps. These values are implemented controls, not established
scaling optima.

## 18. Current Conclusion

RC-KV now has a stable implementation contract rather than only a theoretical
proposal. Block/head-specific attention retains local parameter ownership;
canonical K/V latents provide a cross-route memory interface; BDRE compiles a
token's writer history into reader-specific objects; BDRE-Serial provides the
exact causal oracle; and CPBC provides a trainable approximate prefill with
explicit DP/FB visibility semantics. Stateful TBPTT and manual DDP gradient
synchronization preserve cache lifecycle and optimizer consistency.

Functionality, causality, incremental behavior, and distributed training have
been validated. The next engineering priority is reducing host synchronization,
small-kernel fragmentation, and explicit reader-attention overhead without
changing RC-KV/CPBC semantics. Quality claims must wait for the controlled
benchmark and ablation sequence above.
