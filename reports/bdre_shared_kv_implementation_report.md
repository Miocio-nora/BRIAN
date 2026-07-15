# BRIAN BDRE Reader-Compiled Shared KV Implementation Report

**Status:** exact-reference implementation complete; approximate prefill remains future work
**Date:** 2026-07-15
**Working model name:** `BRIAN-R125-BDRE-RCKV-v1`
**Base reference:** `BRIAN-R125 Global KV Cache-Only v1`

## 1. Purpose

This report defines the implementation contract for the next BRIAN Global KV
version: a Block-Dependent Relative Encoding (BDRE) reader-compiled shared KV
cache for free routing.

The design starts from
`reports/BRIAN_Free_Routing_Shared_KV_Cache.txt`, with the following approved
clarifications:

- reader caches are indexed by the eight free route blocks by default, not by
  a mandatory `reader_depth x reader_block` Cartesian product;
- the optional depth term uses the current inference token's route step, not a
  fixed depth attached to the historical token;
- all attention K/V projections are block- and head-specific;
- writer canonicalizers are block-specific and are not shared;
- the default current-token self cache is a single BDRE-compiled prefix object;
- exact token-by-token execution is the correctness reference;
- parallel or approximate prefill is a later implementation and must be
  measured against the exact reference;
- internal block positions use a spherical-code initialization rather than the
  existing open-arc initialization.

Existing Non-Global, hidden-state Global KV, attention-summary Global KV, and
pure-factorized cache-only implementations must remain available and unchanged.
The BDRE path is additive and selected by a separate model configuration.

## 1.1 Implementation Outcome

The exact `BRIAN-R125-BDRE-RCKV-v1` reference is implemented on branch
`bdre-reader-compiled-kv`. The implementation uses the existing
`brian_route_core` architecture dispatch plus `bdre_shared_kv: true`; legacy
configs still instantiate `BrianRouteCore`, while BDRE configs instantiate
`BrianBDRERouteCore`.

Primary implementation files:

```text
src/brian_sphere_llm/model/bdre_model.py
src/brian_sphere_llm/memory/bdre_shared_kv.py
src/brian_sphere_llm/routing/block_position.py
src/brian_sphere_llm/eval/bdre_cache_visualization.py
src/brian_sphere_llm/train/stage_runner.py
src/brian_sphere_llm/train/trainer.py
tests/test_bdre_shared_kv.py
```

Prepared configs:

```text
configs/model/brian_r125_bdre_rckv_v1.yaml
configs/model/brian_tiny_bdre_rckv.yaml
configs/train/bdre_rckv_r125_5b_ddp2_legacyval.yaml
configs/train/stage5_bdre_tiny_debug.yaml
configs/train/stage5_bdre_tiny_ddp2_debug.yaml
```

Implemented execution and state contracts:

- stateful one-token `forward_incremental` without prefix recomputation;
- token-serial teacher-forced training using the same incremental kernel;
- conventional per-layer incremental KV for fixed pre/post blocks;
- one persistent canonical K/V pair per completed token and free reader block;
- independent block/head Q/K/V, block-specific writer canonicalizers, and
  block/head-specific reader decoders;
- exact decoded-Key RoPE and exact low-dimensional Value aggregation;
- block-only compilation plus eager, persistent-lazy, and dynamic reader-step
  modes;
- `bdre_prefix`, `current_step`, and `none` self-KV modes;
- optional hard compile top-k and separate Key/Value temperatures;
- spherical IN/internal/OUT initialization and weak Gram loss;
- all required scalar diagnostics in train/eval logs;
- HTML/W&B visualization for position geometry, writer route, and per-reader
  Key/Value compile weights.

Validation completed on 2026-07-15:

| Check | Result |
| --- | --- |
| New BDRE unit/integration tests | 16 passed |
| Prefix suffix-invariance | passed |
| Full exact forward vs stateful incremental | max logits difference `0.0` |
| Reader-step eager/lazy/dynamic equivalence | passed at `atol=rtol=1e-6` |
| CUDA BF16 forward/backward | passed on B200 |
| Tiny 3-step train/eval/checkpoint smoke | passed |
| Tiny DDP2 train/eval/checkpoint smoke | passed on B200 GPUs 4-5 |
| Legacy position/loss/routing/global/sparse regressions | passed |
| R125 BF16 exact forward/backward smoke | passed |

The R125 smoke used batch 1, sequence length 4, and fixed sequential routing.
It produced eight valid writer steps per token, finite loss and gradients, about
`1.31 GiB` peak allocated CUDA memory, and about `2.47 s` wall time. This is a
correctness measurement only and must not be extrapolated to 5B training.

Approximate/wavefront prefill is intentionally not implemented. The exact
token-by-token implementation remains the oracle required by Section 16.

## 2. Model Scope

The first implementation targets BRIAN-R125:

| Item | Value |
| --- | ---: |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Fixed pre blocks | 2 |
| Free route blocks | 8 |
| Fixed post blocks | 2 |
| Maximum route steps | 16 |
| Routing | top-1 |
| Block position dimension | 64 |
| Canonical Key dimension | 32 |
| Canonical Value dimension | 32 |

BDRE shared KV applies only to the eight free route blocks. Fixed pre/post
blocks retain conventional per-layer autoregressive KV caches because their
layer identity is stable.

## 3. Parameter Ownership

No writer or reader attention matrix is shared across free blocks unless a
future ablation explicitly enables sharing.

For free block `b` and head `h`:

```text
W_Q[b,h], W_K[b,h], W_V[b,h]
```

are independent. They may remain packed in combined PyTorch linear weights,
but their mathematical and parameter ownership is per block and per head.

After concatenating all writer heads, each block has independent canonical
write projections:

```text
A_K[b]: R^(H*dh) -> R^rK
A_V[b]: R^(H*dh) -> R^rV
```

Each reader block and reader head has independent decoders:

```text
B_K[b,h]: R^rK -> R^dh
B_V[b,h]: R^rV -> R^dh
```

"Shared" refers to the canonical cache interface. It does not imply shared
projection parameters.

## 4. Writer-Side KV Construction

At route step `s`, token `i` selects writer block `b[i,s]`. K/V are generated
from the standard pre-norm attention input:

```text
x_attn[i,s] = RMSNorm(hidden[i,s] + position_adapter(z_current))
k[i,s,h] = W_K[b[i,s],h](x_attn[i,s])
v[i,s,h] = W_V[b[i,s],h](x_attn[i,s])
```

The canonical writer codes are computed before temporal RoPE:

```text
cK[i,s] = A_K[b[i,s]](Concat_h(k[i,s,h]))
cV[i,s] = A_V[b[i,s]](Concat_h(v[i,s,h]))
```

`cK` and `cV` remain mathematically separate. The implementation may fuse
`A_K[b] @ W_K[b]` and `A_V[b] @ W_V[b]` to avoid materializing concatenated
head tensors, provided equivalence tests pass.

During a token's route, step-level canonical codes are temporary:

```text
temporary_K: [batch, max_route_steps, rK]
temporary_V: [batch, max_route_steps, rV]
temporary_valid: [batch, max_route_steps]
temporary_writer_block: [batch, max_route_steps]
```

They are released after the token's persistent reader caches are compiled.

## 5. Position Geometry

### 5.1 Initialization

The BDRE version uses ten unit position vectors:

```text
IN + 8 internal free blocks + OUT
```

Initialization uses antipodal IN/OUT poles and an internal regular simplex in
the orthogonal subspace:

```text
z_IN dot z_OUT = -1
z_IN dot z_block = 0
z_OUT dot z_block = 0
z_block_i dot z_block_j = -1/7, i != j
```

A deterministic seeded orthogonal rotation is applied after construction so
the initialization is not tied to coordinate axes.

### 5.2 Semantics

- `IN` initializes the first router state and first block-position injection.
- The eight internal positions are used by the router, block adapters, and
  BDRE compilation.
- `OUT` represents the terminal state and is injected into the Exit Block.
- IN and OUT do not index reader-compiled KV caches.

All positions remain learnable and are normalized on use.

### 5.3 Geometry Regularization

A weak Gram-matrix loss prevents position collapse without prescribing a route
path:

```yaml
position_geometry:
  enabled: true
  weight: 0.001
  normalize: true
  internal_target: regular_simplex
  in_out_target: antipodal_orthogonal
```

This loss preserves broad spherical separation. It must not be used as a
nearest-neighbor routing objective.

## 6. BDRE Compilation

Let `z_reader` be the active reader block position and `z_writer[s]` the
position of the block used at writer step `s`.

### 6.1 Default: Block-Only

The default score has no route-step distance term:

```text
e[b,s] = tau_position * dot(z_reader[b], z_writer[s])
```

After a token completes its route, it is compiled into eight persistent cache
pairs:

```text
mK[token,b] = Sum_s alphaK[b,s] * cK[token,s]
mV[token,b] = Sum_s alphaV[b,s] * cV[token,s]
```

Persistent cache shapes are:

```text
compiled_K: [batch, sequence, 8, rK]
compiled_V: [batch, sequence, 8, rV]
compiled_valid: [batch, sequence]
```

### 6.2 Optional: Current Reader Step

The depth-aware option uses the current inference token's route step `l_t`:

```text
e[l_t,b,s]
    = -lambda_step * abs(l_t - s)
      + tau_position * dot(z_reader[b], z_writer[s])
```

This mode may use:

```yaml
reader_step_cache: eager
reader_step_cache: lazy
reader_step_cache: dynamic
```

- `eager` compiles all `8 x max_route_steps` reader states when a token route
  completes.
- `lazy` compiles a `(reader_block, reader_step)` state on first access and
  memoizes it; this is the default cache policy when reader-step scoring is
  enabled.
- `dynamic` retains historical step-level writer codes and recomputes fusion
  at every read. It exists for cost and correctness measurement, not as the
  expected production path.

### 6.3 Optional: Late-Step Bias

The block-only cache can add a reader-independent writer-step prior:

```text
e[b,s]
    = tau_position * dot(z_reader[b], z_writer[s])
      + late_step_weight * s / S
```

This keeps one cache per reader block but may collapse compilation toward the
last writer steps. It is implemented only as an ablation and is disabled by
default.

### 6.4 Candidate Selection and Temperatures

Hard top-k is optional:

```yaml
compile_top_k: null  # use every valid writer step; default
compile_top_k: 16
compile_top_k: 12
compile_top_k: 8
compile_top_k: 4
```

Key and Value use the same candidate set and separate softmax temperatures:

```text
alphaK = softmax(e / T_K)
alphaV = softmax(e / T_V)
T_K <= T_V
```

Initial defaults are:

```yaml
tau_position: 1.0
key_temperature: 0.5
value_temperature: 1.0
compile_top_k: null
```

Temperature and top-k values remain explicit ablation parameters.

## 7. Current-Token Self KV

The default self mode is `bdre_prefix`.

At current token route step `s`:

1. Compute the current step's standard pre-RoPE K/V and canonical `cK_s/cV_s`.
2. Append them to the current token's temporary route prefix.
3. Compile steps `1..s` with the current reader block and active BDRE scoring
   mode.
4. Produce exactly one temporary self K/V object.
5. Attend to historical compiled caches plus this one self object.

This avoids exposing earlier route steps as separate attention objects while
retaining an online summary of the current token's route history. There is no
circular dependency because current-step K/V are generated from the attention
input before attention output is computed.

Supported modes:

```yaml
self_kv_mode: bdre_prefix   # default
self_kv_mode: current_step  # standard current-step self K/V ablation
self_kv_mode: none          # strict historical i < t ablation
```

Self-prefix compilation uses the same top-k, temperatures, and scoring mode as
historical cache compilation unless a diagnostic override is explicitly set.

## 8. Reader Attention and RoPE

For historical token `i < t`, active reader block `b`, and head `h`:

```text
k_hat[i,b,h] = B_K[b,h](mK[i,b])
v_hat[i,b,h] = B_V[b,h](mV[i,b])
```

The exact reference applies temporal RoPE after Key decoding:

```text
q_rope[t,b,h] = RoPE(W_Q[b,h](x_reader), position=t)
k_rope[i,b,h] = RoPE(k_hat[i,b,h], position=i)
```

Attention remains causal and includes historical tokens plus the single
current-token self object selected by `self_kv_mode`.

### 8.1 Low-Dimensional Optimization

Value decoding is optimized directly and exactly:

```text
value_latent = Sum_i attention_weight[i] * mV[i,b]
output_head = B_V[b,h](value_latent)
```

Key scoring without temporal position transforms also satisfies:

```text
q dot (B_K mK) == (B_K^T q) dot mK
```

However, current BRIAN uses position-dependent RoPE on both query and decoded
Key. A single latent query cannot be reused for every historical token because
each Key position has a different rotation. Therefore:

- `explicit_decode_rope` is the default exact Key path;
- a future `latent_relative_rope` path may be enabled only after deriving an
  exact formulation and passing explicit-vs-latent equivalence tests;
- the naive position-free low-dimensional Key equation must not be used when
  RoPE is active.

## 9. Exact Token-by-Token Execution

The first implementation is sequence-serial and batch-parallel.

For token position `t`:

```text
embedding(t)
  -> incremental fixed pre blocks with standard per-layer KV
  -> initialize position at IN
  -> run complete free route for token t
       -> read compiled caches of tokens < t
       -> include one current self object
       -> collect temporary writer cK/cV
  -> compile token t into persistent reader caches
  -> Exit Block
  -> incremental fixed post blocks with standard per-layer KV
  -> LM logits(t)
```

Training processes teacher-forced tokens in order, stacks per-token logits,
and applies the standard shifted causal LM loss. DDP parallelizes batches, not
sequence positions.

The exact implementation must expose a stateful incremental API rather than
re-running the complete prefix for each generated token.

## 10. Routing and Training Defaults

The BDRE version inherits the corrected Global v1 routing controls:

```yaml
routing:
  top_k: 1
  later_top_k: 1
  max_route_steps: 16
  hard_exit: true
  min_exit_step: 4
  exit_ramp_start: 12
  force_final_exit: true
  self_recur_max_consecutive: 2
  logit_noise_std: 0.08
  logit_noise_decay_steps: 50000
  logit_noise_min_std: 0.008
  random_route_probability: 0.25
  random_route_decay_steps: 50000
  random_route_min_probability: 0.03
```

The corrected selected-balance, coverage-floor, weak cost, and exit-boundary
losses remain enabled. The route-position location loss is explicitly disabled:
position geometry affects BDRE compilation but does not prescribe a route path.
Route imitation remains disabled after the existing short execution curriculum.

Top-2 and weighted route fusion are out of scope for BDRE v1.

## 11. Default Configuration Contract

```yaml
model_name: brian_r125_bdre_rckv_v1
architecture: brian_route_core

route_pool_blocks: 8
max_route_steps: 16
top_k: 1
later_top_k: 1

block_position_dim: 64
block_position_mode: spherical_code
independent_input_position: true

bdre_shared_kv: true
bdre_key_dim: 32
bdre_value_dim: 32

bdre_depth_mode: none
bdre_step_lambda: 0.0
bdre_position_tau: 1.0
bdre_late_step_weight: 0.0

bdre_compile_top_k: null
bdre_key_temperature: 0.5
bdre_value_temperature: 1.0
bdre_reader_step_cache: lazy

bdre_self_kv_mode: bdre_prefix
bdre_key_read_mode: explicit_decode_rope
bdre_value_read_mode: latent_aggregate

position_geometry:
  enabled: true
  weight: 0.001
  normalize: true
  internal_target: regular_simplex
  in_out_target: antipodal_orthogonal

execution:
  mode: token_by_token
```

`bdre_reader_step_cache` is dormant while `bdre_depth_mode: none`.

## 12. Complexity Budget

Assume `S=16`, `B_route=8`, `rK=rV=32`, and BF16.

### 12.1 Persistent Cache

| Mode | Elements/token | Bytes/token | 2048-token context |
| --- | ---: | ---: | ---: |
| Block-only | `8 x 64 = 512` | 1 KB | about 2 MB |
| Block x reader-step | `8 x 16 x 64 = 8192` | 16 KB | about 32 MB |
| Retained writer-step latent | `16 x 64 = 1024` | 2 KB | about 4 MB |

These estimates exclude allocator overhead, validity masks, gradients, and
fixed pre/post caches.

### 12.2 BDRE Self Prefix

Across a complete 16-step route:

```text
Sum(1..16) = 136 writer candidates
position scoring: 136 x 64 = 8,704 MAC
K/V weighted sums: 136 x (32+32) = 8,704 MAC
total: about 17,408 MAC per token route
temporary storage: about 2 KB per active token
```

This arithmetic cost is small relative to route-block attention and FFN. The
measured cost may still be affected by small kernels, softmax launches, and
temporary tensor layout, so dedicated B200 profiling is required.

### 12.3 Dynamic Reader-Step Fusion

For context length `N=2048`:

```text
N x S x (rK+rV)
= 2048 x 16 x 64
= about 2.1M latent MAC per reader step
```

At 16 reader steps this is about 33.6M MAC per generated token, before reader
attention and block compute. This mode must be benchmarked against eager and
lazy compiled reader-step caches.

## 13. Implementation Plan

### Phase A: Additive Model and Configuration

- Add BDRE config parsing and validation without changing existing config
  defaults.
- Add spherical-code IN/internal/OUT initialization.
- Add geometry metrics and weak geometry loss.
- Add independent writer canonicalizers and reader/head decoders.

### Phase B: Cache and Compiler

- Add temporary per-token writer-step storage.
- Add block-only compiler.
- Add optional hard top-k and separate K/V temperatures.
- Add reader-step eager, lazy, and dynamic modes.
- Add late-step-bias ablation.
- Add `bdre_prefix`, `current_step`, and `none` self modes.

### Phase C: Exact Incremental Forward

- Add standard incremental KV state for fixed pre/post blocks.
- Add stateful token-step API for the route core.
- Add exact explicit Key decode + RoPE.
- Add latent Value aggregation.
- Add token-by-token training forward and generation integration.

### Phase D: Performance and Training Integration

- Preallocate temporary writer tensors.
- Vectorize BDRE scoring and prefix fusion across batch.
- Add timing and memory metrics.
- Integrate W&B route/cache visualizations.
- Add DDP batch-parallel smoke training.

### Phase E: Approximate Prefill

- Implement approximate or wavefront prefill only after the exact reference
  passes correctness and training smoke tests.
- Keep exact token-by-token mode permanently available as the oracle.

## 14. Required Metrics

Training and evaluation logs must include:

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

Visualization must expose:

- IN, eight internal blocks, and OUT on the sphere;
- writer route steps and their block positions;
- per-reader-block BDRE weights over writer steps;
- Key and Value weight differences;
- optional reader-step-conditioned weight changes;
- top-k support and last-step concentration.

## 15. Correctness Tests

The implementation is not accepted without:

1. Shape and lifecycle tests for temporary and persistent caches.
2. Proof that completed tokens expose one object per reader block, not one
   object per writer step or writer head.
3. Per-block/head parameter ownership tests proving no unintended sharing.
4. Causal suffix-invariance tests.
5. Incremental generation consistency tests.
6. Proof that current self BDRE uses only route prefix `1..s`.
7. Proof that future route steps and future tokens are invisible.
8. Block-only compile equivalence between eager and lazy execution.
9. Reader-step equivalence among eager, lazy, and dynamic modes.
10. `compile_top_k: null` equivalence to explicit all-step aggregation.
11. Hard top-k support and gradient-finiteness tests.
12. Separate Key/Value temperature tests.
13. Explicit Value decode versus latent Value aggregation equivalence.
14. Explicit Key+RoPE reference tests.
15. Spherical initialization Gram-target tests.
16. CUDA BF16 forward/backward finite tests.
17. DDP train/eval smoke tests.
18. Regression tests proving existing BRIAN model configs remain unchanged.

## 16. Exact vs Approximate Prefill Acceptance

Approximate prefill must be compared with exact token-by-token execution using:

```text
logits max/mean absolute difference
next-token prediction agreement
route-action agreement
route-length agreement
compiled-cache cosine similarity and relative error
validation loss and perplexity
reasoning S600
public S600
throughput
peak CUDA memory
```

Approximate prefill cannot replace the exact default based only on throughput.
Its quality and route behavior must remain within explicitly reported error
bounds.

## 17. Initial Ablation Matrix

| ID | Depth mode | Self mode | Top-k | Cache policy | Purpose |
| --- | --- | --- | --- | --- | --- |
| B0 | none | bdre_prefix | all | block eager | default BDRE |
| B1 | none | current_step | all | block eager | self-prefix value |
| B2 | none | none | all | block eager | strict historical-only |
| D0 | reader_step | bdre_prefix | all | lazy | current-step distance |
| D1 | reader_step | bdre_prefix | all | eager | lazy/eager equivalence |
| D2 | reader_step | bdre_prefix | all | dynamic | dynamic cost reference |
| L0 | late_step_bias | bdre_prefix | all | block eager | late-step collapse risk |
| K0 | none | bdre_prefix | 12 | block eager | mild hard top-k |
| K1 | none | bdre_prefix | 8 | block eager | medium hard top-k |
| K2 | none | bdre_prefix | 4 | block eager | aggressive hard top-k |

No large sweep should run before B0 passes exact correctness, CUDA backward,
short training, and cache visualization checks.

## 18. Risks and Guardrails

- Token-by-token training will be much slower than the current full-sequence
  route forward. It is accepted as the exact reference, not assumed to be the
  final performance path.
- Reader-step caches can grow to 128 cache pairs per token; lazy compilation
  and measured state occupancy are required.
- Late-step bias can collapse all compiled caches toward final writer states;
  last-step mass and compiler entropy must be monitored.
- `bdre_prefix` can create a within-token memory shortcut; compare it with
  `current_step` and track self-attention concentration.
- Hard top-k is non-smooth and routes gradients only through selected writer
  steps.
- Independent writer maps must still learn a coherent canonical interface.
- Position initialization alone does not prevent later geometry collapse;
  normalization, weak separation loss, and visualization are required.
- The position-free low-dimensional Key equation is not exact under the current
  decoded-Key RoPE contract.

## 19. Final Approved Defaults

```text
Model: BRIAN-R125-BDRE-RCKV-v1
Execution: exact token-by-token
Free blocks: 8
Routing: top-1
Max route steps: 16

Writer K/V: independent per block and head
Writer A_K/A_V: independent per block
Reader B_K/B_V: independent per block and head

Persistent cache index: reader block only
Depth-step term: implemented, disabled
Late-step bias: implemented, disabled
Reader-step cache policy: lazy when enabled

Compile top-k: all steps
Key temperature: 0.5
Value temperature: 1.0
Self KV: BDRE route prefix

Key read: explicit decode + RoPE
Value read: latent aggregation + one decode

Position initialization:
  IN/OUT antipodal poles
  eight internal blocks as orthogonal-subspace regular simplex
Position geometry loss: enabled, weak
Route-position location loss: disabled

Approximate prefill: disabled until exact-reference validation
```
