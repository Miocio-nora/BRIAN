# BRIAN-Sphere-LLM Project Plan

**Repository:** `BRIAN-Sphere-LLM`  
**Project name:** **BRIAN-Sphere-LLM**  
**Short name:** **BRIAN-Sphere** / **BRIAN**  
**Python package:** `brian_sphere_llm`  
**Expanded name:** **Block-Routed Inference with Adaptive Navigation over a Latent Operator Sphere**  
**Status:** Long-range research and engineering plan  
**Owner:** YMH-Latent-Sphere project  
**Last updated:** 2026-06-13

---

## 0. Executive Summary

BRIAN-Sphere-LLM is a research project to turn the fixed-depth computation chain of a Transformer into a learnable latent computation graph.

A standard decoder-only Transformer executes a fixed sequence of blocks:

```text
input → B1 → B2 → ... → BL → output
```

BRIAN-Sphere-LLM replaces part of the middle stack with a routeable block pool:

```text
input → pre-blocks → router-controlled latent block path → output block → post-blocks / LM head
```

The core hypothesis is:

> A language model can learn not only how to map tokens to tokens, but also how to organize its own internal computation path in latent space.

The project combines the following technical ideas:

1. Latent-space reasoning.
2. Router-controlled block selection.
3. Output block as a terminal routing action.
4. Block-position state / embedding as a computation-space coordinate.
5. Optional shared canonical global KV memory.
6. Optional parallel latent passing / latent beam search.

The first formal research goal is **not** to beat large public models. The first goal is to prove that the route-core system is trainable, measurable, and controllable at small scale.

---

## 1. Core Concept

### 1.1 What the project is really testing

The project tests whether the fixed physical depth of a Transformer can be replaced by a learned latent computation path.

Instead of forcing every input through the same middle-layer sequence, BRIAN-Sphere-LLM allows the model to choose:

```text
B3 → B5 → B5 → B2 → B7 → OUT
```

or:

```text
B2 → B3 → OUT
```

or:

```text
B4 → B4 → B6 → B8 → OUT
```

depending on the current latent state.

### 1.2 Minimal system

The minimal routed system maintains two states:

\[
S_r = (H_r, P_r)
\]

where:

- \(H_r\): content hidden state.
- \(P_r\): block-position / operator-position state.

The router selects an action:

\[
a_r \in \{B_1, B_2, ..., B_m, OUT\}
\]

If the selected action is an internal block:

\[
H_{r+1} = B_{a_r}(H_r, P_r)
\]

\[
P_{r+1} = E_{a_r}
\]

If the selected action is `OUT`:

\[
logits = Output(H_r, P_r)
\]

This minimal form is the core of the project. Global KV memory and parallel passing are extensions, not prerequisites.

---

## 2. Design Principles

### 2.1 Do not train the full system from scratch at once

The complete system has too many free variables:

- free router;
- block-position state;
- output action;
- top-k activation;
- global KV memory;
- parallel passing;
- balance / cost / location losses.

Training all of these simultaneously from zero is likely to fail or become impossible to diagnose.

The correct order is:

```text
fixed path → pseudo routing → scheduled free routing → output action → global KV → parallel passing
```

### 2.2 Route core first, memory second, parallelism third

The route-core system must be validated before adding global KV.

The recommended order:

1. Router + block-position + output action.
2. Local/global KV memory.
3. Parallel passing.

### 2.3 Always compare against fixed Transformer baselines

Every routed model must be compared against fixed-sequence baselines under at least three views:

1. Same parameter count.
2. Similar training FLOPs.
3. Similar active block evaluations per token.

A routed model that wins only by doing much more compute is not yet evidence for the architecture.

---

## 3. Architecture Roadmap

### 3.1 Model family names

| Scale | Name | Purpose |
|---|---:|---|
| Tiny smoke / debug | `BRIAN-R30` / `BRIAN-R60` | pipeline and training sanity |
| Small research model | `BRIAN-R125` | first serious route-core validation |
| Medium research model | `BRIAN-R350` | scaling trend validation |
| Large research model | `BRIAN-R1B` | serious validation after smaller success |

---

## 4. Recommended First Architecture: BRIAN-R125

### 4.1 Base model

Use a LLaMA-like decoder-only Transformer.

Recommended configuration:

| Item | Suggested value |
|---|---:|
| Parameter range | 110M–150M |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| FFN | SwiGLU / gated MLP |
| Norm | RMSNorm |
| Token position | RoPE |
| Vocab | 32k tokenizer |
| Context | 2k initially, optionally 4k |
| Route pool | middle 8 blocks |
| Pre blocks | 2 |
| Post blocks | 2 |
| Route actions | 8 internal blocks + 1 OUT |
| Max latent route steps | 4–8 |
| Initial routing | top-1 |
| Later routing | top-2 weighted fusion |
| Block-position dimension | 64 or 128 |
| Global KV | off for first route-core stage |
| Parallel passing | off |

Architecture split:

```text
pre:   B1, B2
pool:  B3, B4, B5, B6, B7, B8, B9, B10
post:  B11, B12
```

The router action space is:

```text
{B3, B4, B5, B6, B7, B8, B9, B10, OUT}
```

---

## 5. Later Architectures

### 5.1 BRIAN-R350

Recommended only after `BRIAN-R125` passes route-core success criteria.

| Item | Suggested value |
|---|---:|
| Parameter range | 300M–400M |
| Layers | 24 |
| Hidden size | 960 configured, 1024 optional |
| Attention heads | 16 |
| Pre blocks | 4 |
| Route pool | 16 |
| Post blocks | 4 |
| Route actions | 16 internal + OUT |
| Max route steps | 6–12 |
| Top-k | 1 → 2 |
| Global KV | stage 2 |
| Context | 4k → 8k |

### 5.2 BRIAN-R1B

Only start after `BRIAN-R350` shows at least one meaningful advantage over baselines.

| Item | Suggested value |
|---|---:|
| Parameter range | 0.8B–1.3B |
| Layers | 32 |
| Hidden size | 1536 or 2048 |
| Route pool | 20–24 blocks |
| Max route steps | 8–16 |
| Top-k | 2 |
| Global KV | on |
| Parallel passing | experimental only |
| Context | 8k → 16k |

---

## 6. Block-Position State

### 6.1 Purpose

The block-position state is not token position. It is a computation-space coordinate.

It tells the router and the next block:

- which region of block/operator space the latent state is currently in;
- which block produced or is associated with the current state;
- which nearby operators are geometrically plausible next choices;
- where the output action is located relative to internal blocks.

### 6.2 State representation

Maintain a separate position state:

\[
P_r \in \mathbb{R}^{d_p}
\]

Do not initially mix position directly and permanently into the content hidden state. Instead:

- router receives both pooled content and position;
- blocks receive position through a controlled adapter or side-channel;
- position evolves according to routed action.

Router input:

\[
Router(\text{Pool}(H_r), P_r)
\]

Block input:

\[
B_i(H_r, P_r)
\]

### 6.3 Position initialization

Initialize action positions from the original block order:

```text
IN → B1 → B2 → ... → Bm → OUT
```

Use circular or open-arc sinusoidal coordinates:

\[
E_i = [\cos\theta_i, \sin\theta_i, \cos2\theta_i, \sin2\theta_i, ...]
\]

Recommended first version: **open arc**, not full closed circle.

Reason: a full circle may place `IN` and `OUT` geometrically close, making premature exit easier.

### 6.4 Position update

For top-1 routing:

\[
P_{r+1} = E_{a_r}
\]

For top-k routing:

\[
P_{r+1} = \text{Norm}\left(\sum_{a \in TopK} \alpha_a E_a\right)
\]

If the router chooses `30% B1 + 70% B2`, then the position state should move to:

\[
P_{r+1} = \text{Norm}(0.3E_{B1} + 0.7E_{B2})
\]

### 6.5 Location bias and location loss

The router may receive a location bias:

\[
\ell_a = \ell_a + \beta \cdot \text{sim}(P_r, E_a)
\]

The location loss can encourage geometrically coherent routing:

\[
L_{loc}=\sum_a \pi(a)d(P_r,E_a)^2
\]

Training schedule:

```text
stronger location bias/loss early
weaker location bias/loss later
```

The goal is to stabilize early training without preventing long-range routing later.

---

## 7. Router Design

### 7.1 Router role

The router is not a standard MoE router.

A standard MoE router usually asks:

```text
Which expert should process this token?
```

BRIAN-Sphere-LLM's router asks:

```text
Which internal operator should the latent state pass through next, or should it exit now?
```

It is closer to a latent computation policy:

\[
\pi(a \mid H_r, P_r)
\]

where:

\[
a \in \{B_1, ..., B_m, OUT\}
\]

### 7.2 Output block as terminal action

`OUT` is an action in the same action space as internal blocks.

If:

```text
top1_action == OUT
```

then the model exits the latent routing loop.

`OUT` should not merely be a scalar halt head. It is a terminal operator that converts the current latent state into a representation suitable for post-blocks / LM head.

### 7.3 Top-k behavior

Start with top-1 routing.

After stability:

```text
top-k = 2
```

Initial top-k implementation should use weighted fusion, not independent branches:

\[
H_{r+1}=\sum_{a\in TopK}\alpha_aB_a(H_r,P_r)
\]

Independent parallel passing is postponed.

---

## 8. Router Training Strategy

### 8.1 Why free router training is risky

A free router from step zero is likely to fail because:

- the action space is discrete;
- block semantics are not adapted to dynamic routing;
- `OUT` can cause early-exit collapse or never-exit behavior;
- LM loss gives weak and delayed credit assignment to routing choices;
- global KV, if enabled too early, adds another unstable pathway.

### 8.2 Pseudo routing curriculum

The router should first learn stable pseudo routes.

#### Pseudo action types

```text
advance: B_i → B_{i+1}
skip:    B_i → B_{i+2}
recur:   B_i → B_i
exit:    B_i → OUT
```

These correspond to a controlled combination of skip-style dynamic depth and recurrent computation.

### 8.3 Difficulty-conditioned pseudo routing

Use the fixed baseline model's per-sample or per-token loss as a rough difficulty signal.

Let:

\[
d(x)=CE_{baseline}(x)
\]

Then define route length or pseudo route pattern by difficulty bin:

| Difficulty bin | Pseudo behavior |
|---|---|
| Easy / low baseline CE | more skip, earlier `OUT` |
| Medium | mostly original sequence |
| Hard / high baseline CE | more recurrent steps, later `OUT` |

This teaches the router a useful prior:

```text
simple examples should use less internal computation;
harder examples should use more internal computation.
```

---

## 9. Training Stages

### Stage 0: Fixed Transformer baseline

Train a normal decoder-only baseline.

Purpose:

- provide a strong comparison point;
- provide difficulty scores for pseudo routing;
- provide teacher logits / hidden states if distillation is needed.

Required models:

```text
BRIAN-B125 baseline
BRIAN-B350 baseline, later
```

### Stage 1: Fixed route wrapper

Convert the middle blocks into a route pool but force original order:

```text
B1 → B2 → ... → Bm → OUT
```

The router is trained to imitate the fixed path, but does not control the forward path yet.

Success criteria:

```text
validation loss within 1–3% of baseline
route imitation accuracy > 98%
no hidden-state numerical instability
```

### Stage 2: Pseudo skip/recurrent routing

Forward path follows pseudo route.

Router learns imitation.

Loss:

\[
L = L_{LM} + \lambda_{route}L_{route} + \lambda_{loc}L_{loc} + \lambda_{balance}L_{balance} + \lambda_{cost}L_{cost}
\]

Initial recommended weights:

```text
lambda_route   = 1.0
lambda_loc     = 0.05–0.1
lambda_balance = 0.01
lambda_cost    = 0.001–0.01
```

### Stage 3: Scheduled free routing

Gradually let the router control the forward path.

Example schedule:

```text
phase 1: 90% pseudo, 10% router
phase 2: 70% pseudo, 30% router
phase 3: 50% pseudo, 50% router
phase 4: 20% pseudo, 80% router
phase 5: 0% pseudo, 100% router
```

Decrease route imitation weight over time:

```text
lambda_route: 1.0 → 0.5 → 0.2 → 0.05
```

Success criteria:

```text
free routing does not collapse validation loss
average active route steps are controllable
OUT action is neither always early nor never used
block usage does not collapse to one block
```

### Stage 4: Real output action

Enable hard terminal behavior:

```text
if top1_action == OUT:
    exit latent loop
```

At this stage, cost loss becomes important.

The desired behavior:

```text
easy samples exit earlier
hard samples use more route steps
```

### Stage 5: Global KV memory

Only after route-core stability, add global KV.

First version:

```text
local KV + global canonical compressed memory
window cache + attention sink
```

Do not add complex memory tiers or heavy-hitter policies in the first global-KV stage.

### Stage 6: Parallel passing

Only after route-core and global KV are stable.

Parallel passing is treated as latent beam search:

\[
S_r^{(n)} \rightarrow \{S_{r+1}^{(n,1)}, S_{r+1}^{(n,2)}, ...\}
\]

Branch score:

\[
w_{r+1}=w_r+\log\pi(a)-\lambda_{cost}
\]

Must include pruning:

```text
keep top-B branches
```

Parallel passing is not part of the first serious experiment.

---

## 10. Global KV Memory Plan

### 10.1 Purpose

Dynamic block paths make raw layer-specific KV cache hard to share.

The global KV pool should not store raw K/V from arbitrary blocks. It should store canonical compressed memory codes:

\[
C = \{c_1, c_2, ..., c_N\}
\]

Each block/head has adapters:

```text
local → global: compression/write
global → local: read/decompression
```

### 10.2 Local/global split

Memory consists of:

```text
local KV: block/head-private, normal attention pathway
global KV: shared canonical compressed memory
```

### 10.3 First global cache policy

Use:

```text
M_global = M_sink ∪ M_window
```

- `M_sink`: first few memory slots, retained permanently.
- `M_window`: recent compressed global codes, fixed-size sliding window.

Do not use complex 3-tier memory in the first version.

### 10.4 First global-KV questions

The global KV stage should answer:

1. Does global memory get used at all?
2. Does it improve long-context / memory-constrained tasks?
3. Does it reduce effective KV memory under equal quality?
4. Does sink retention help stability?
5. Are per-block or per-head adapters necessary?

---

## 11. Parallel Passing Plan

### 11.1 Meaning

Top-k weighted fusion keeps one latent state.

Parallel passing keeps multiple latent branches:

```text
state → branch 1 through B2
      → branch 2 through B5
      → branch 3 through B7
```

This is equivalent to latent beam search over the block graph.

### 11.2 When to start

Do not implement or train parallel passing until:

- route-core is stable;
- output action works;
- cost loss controls compute;
- at least one `BRIAN-R350` run shows meaningful benefit.

### 11.3 Required constraints

Parallel passing must include:

- branch score decay;
- branch pruning;
- branch cost loss;
- shared base memory plus branch delta memory;
- strict maximum branch count.

Recommended initial branch count:

```text
B = 2
```

Do not start with large beams.

---

## 12. Losses

Total training loss:

\[
L = L_{LM} + \lambda_{route}L_{route} + \lambda_{balance}L_{balance} + \lambda_{cost}L_{cost} + \lambda_{loc}L_{loc}
\]

### 12.1 LM loss

Standard next-token cross-entropy.

For early experiments, it is acceptable to train/evaluate with last-token prediction for causal correctness and simpler routing diagnostics.

### 12.2 Route imitation loss

Cross-entropy or KL between router distribution and pseudo-route target.

\[
L_{route}=CE(\pi(a|H,P), a_{pseudo})
\]

### 12.3 Balance loss

Prevents internal block collapse.

Apply only to internal blocks, not to `OUT`.

### 12.4 Cost loss

Penalizes excessive internal computation.

\[
L_{cost}=E[\text{active internal block evaluations}]
\]

### 12.5 Location loss

Encourages route actions to be geometrically coherent with the current block-position state.

Use strongly early, weakly later.

### 12.6 Output-weighted CE warning

Avoid naive:

\[
L = \pi_{OUT} \cdot CE
\]

unless \(\pi_{OUT}\) is stop-gradient.

Otherwise, the router can learn to reduce output probability to avoid CE on hard examples.

For multi-output or multi-branch later stages, prefer mixture likelihood:

\[
L_{out} = -\log \sum_o \pi(o)\exp(-CE_o)
\]

---

## 13. Dataset Plan

### 13.1 General pretraining data

Recommended sources:

- FineWeb-Edu subset.
- FineWeb subset.
- TinyStories for low-cost small-model language sanity.
- Synthetic routing / reasoning curriculum.
- Optional small code / structured text mixture at 350M+.

### 13.2 125M data mixture

Target tokens:

```text
2B–5B tokens
```

Suggested mixture:

```text
70% FineWeb-Edu subset
20% TinyStories / simple narrative data
10% synthetic reasoning / routing curriculum
```

Purpose:

- train a real LM signal;
- keep small-model language behavior stable;
- provide enough synthetic structure for route training.

### 13.3 350M data mixture

Target tokens:

```text
10B–30B tokens
```

Suggested mixture:

```text
80% FineWeb-Edu / FineWeb subset
10% synthetic reasoning
5% code / structured text
5% math / QA-style data
```

### 13.4 1B data mixture

Target tokens:

```text
50B–100B tokens
```

Purpose:

- test scaling trend;
- not necessarily train a compute-optimal general LM;
- focus on architecture validation.

### 13.5 Evaluation datasets

Use multiple classes of evaluation:

| Category | Suggested datasets / tasks | Purpose |
|---|---|---|
| Language modeling | held-out FineWeb-Edu / validation corpus | general LM quality |
| Math reasoning | GSM8K, synthetic arithmetic chains | latent reasoning behavior |
| Symbolic reasoning | synthetic multi-step tasks, BBH-style tasks | path-dependent computation |
| Long context | RULER, LongBench, needle/multi-hop tracing | global KV validation |
| Compute adaptivity | difficulty-binned validation set | easy samples exit earlier, hard samples compute more |

---

## 14. Experiment Matrix

### 14.1 Route-core ablations

| ID | Experiment | Purpose |
|---|---|---|
| A0 | fixed Transformer baseline | main baseline |
| A1 | fixed route wrapper | verify wrapper does not break baseline |
| A2 | recurrent-only | compare with Universal-Transformer-like recurrence |
| A3 | skip-only | compare with dynamic depth / skip behavior |
| A4 | router without block-position | test need for position state |
| A5 | router with block-position | main route-core model |
| A6 | output action disabled | isolate terminal action effect |
| A7 | output action enabled | main terminal-action model |
| A8 | top-1 routing | stable routing baseline |
| A9 | top-2 weighted routing | test expressivity vs compute |

### 14.2 Block-position ablations

| ID | Experiment | Purpose |
|---|---|---|
| P0 | no position state | baseline |
| P1 | random position initialization | test learned geometry only |
| P2 | open-arc initialization | main proposal |
| P3 | full circular initialization | compare geometry |
| P4 | position only to router | test router-only effect |
| P5 | position to router + blocks | main proposal |
| P6 | no location bias | test geometric bias |
| P7 | no location loss | test regularization |
| P8 | direct position-hidden addition | check content pollution |
| P9 | separate position state | main proposal |

### 14.3 Global KV ablations

| ID | Experiment | Purpose |
|---|---|---|
| K0 | local KV only | memory baseline |
| K1 | local + global uncompressed | verify global path |
| K2 | local + global compressed | main proposal |
| K3 | global without sink | test sink importance |
| K4 | global with sink | main proposal |
| K5 | window size sweep | memory-quality curve |
| K6 | per-block adapter | low-cost adapter baseline |
| K7 | per-head adapter | expressivity test |
| K8 | per-block + low-rank head delta | likely main later version |

### 14.4 Parallel passing ablations

Only after route-core and global KV stability.

| ID | Experiment | Purpose |
|---|---|---|
| PP0 | top-k weighted fusion | single-state baseline |
| PP1 | independent branches, beam=2 | first parallel passing |
| PP2 | beam=4 | capacity test |
| PP3 | branch cost off | verify need for cost |
| PP4 | branch cost on | main proposal |
| PP5 | output if top-1 OUT | clean terminal rule |
| PP6 | output if OUT in top-k | contrast; likely unstable |

---

## 15. Metrics

### 15.1 Standard LM metrics

- validation loss;
- perplexity;
- downstream task accuracy;
- benchmark score.

### 15.2 Routing metrics

These are mandatory.

| Metric | Meaning |
|---|---|
| route entropy | whether routing collapses |
| block load entropy | whether internal blocks are used |
| average route steps | compute cost |
| exit step distribution | output action behavior |
| route path diversity | whether model finds diverse paths |
| recurrent ratio | how often blocks repeat |
| skip ratio | how often model jumps forward |
| location distance | geometric coherence |
| difficulty-step correlation | whether hard samples compute more |
| OUT probability by difficulty | exit behavior sanity |

The most important diagnostic:

\[
corr(CE_{baseline}, route\_steps)
\]

A positive correlation suggests adaptive internal computation.

### 15.3 Compute metrics

- active block evaluations per token;
- tokens/sec;
- estimated FLOPs/token;
- GPU memory;
- KV bytes/token;
- latency/token;
- train step time;
- inference time with and without hard exit.

### 15.4 Global KV metrics

- global attention mass;
- sink attention mass;
- local/global read ratio;
- global cache window utilization;
- memory-constrained long-context accuracy;
- performance vs global window size;
- performance vs KV bytes/token.

---

## 16. Go / No-Go Criteria

### 16.1 BRIAN-R125 go criteria

Proceed from 125M route-core to 350M only if:

```text
fixed route wrapper loss is within 1–3% of baseline
router imitation accuracy > 95%
free routing does not collapse validation loss
average route steps can be controlled by cost loss
block load does not collapse to one internal block
block-position ablation shows measurable difference
OUT action is neither always early nor never used
```

If these fail, do not scale up. Fix route training or position design first.

### 16.2 BRIAN-R350 go criteria

Proceed to 1B/global serious validation only if:

```text
same active compute: routed model is not worse than baseline
reasoning or synthetic multi-step tasks improve vs baseline
difficulty-step correlation is clearly positive
OUT action reduces compute on easy samples
global KV, if tested, improves memory-constrained long-context tasks
```

### 16.3 BRIAN-R1B go criteria

A 1B run is successful only if at least one core advantage is stable:

```text
better compute-adjusted perplexity
better reasoning accuracy
better long-context memory efficiency
less visible CoT needed for similar reasoning performance
```

and:

```text
routing does not collapse
KV memory remains controlled
inference latency remains acceptable
```

---

## 17. Resource and Cost Estimates

### 17.1 Estimation philosophy

Use GPU-hours as the primary cost unit. Provider prices change frequently, so concrete dollar estimates should be refreshed before each major run.

A rough training compute estimate:

\[
C \approx 6ND \cdot \gamma
\]

where:

- \(N\): active parameter scale;
- \(D\): training tokens;
- \(\gamma\): routing compute multiplier.

The multiplier \(\gamma\) increases with:

- extra route steps;
- top-k block execution;
- global KV read/write;
- parallel passing.

### 17.2 Current public cloud pricing anchors

As of this plan's update date, public H100 pricing varies significantly by provider and instance type. Treat the following as anchors, not guarantees:

- RunPod lists H100 SXM from about `$3.29/hr`.
- Lambda cloud / cluster pricing varies by cluster size and availability.
- AWS P5 H100 instances are typically much more expensive on on-demand pricing, but reserved/capacity-block pricing can differ.

Before any expensive run, update these numbers from the provider pages in the references section.

### 17.3 GPU-hour planning table

| Stage | Model | Tokens | Goal | Estimated GPU-hours / run | Approx cost at $3–$7/GPU-hour |
|---|---:|---:|---|---:|---:|
| Phase A | 30M–60M | 0.5B–1B | pipeline sanity | 20–80 | $60–$560 |
| Phase B | 125M | 2B–5B | route-core validation | 100–300 | $300–$2,100 |
| Phase C | 350M | 10B–30B | scaling trend | 500–1,500 | $1,500–$10,500 |
| Phase D | 1B | 50B–100B | serious validation | 2,000–6,000 | $6,000–$42,000 |

These estimates are planning-level numbers. Add buffer for failed runs, evals, checkpointing, data loading, and dynamic-routing overhead.

### 17.4 Program-level budget tiers

#### Lean program

Goal: decide whether the architecture is worth continuing.

Expected experiments:

```text
125M baseline
125M fixed route
125M router + position main
3–4 key ablations
1–2 350M main/scaling runs
```

Estimated total:

```text
1,500–4,000 GPU-hours
$5k–$28k GPU cost
4–8 weeks
```

#### Serious internal research program

Goal: produce credible internal technical conclusion.

Expected experiments:

```text
125M complete route-core ablation
350M route-core ablation
global KV phase
reasoning and long-context evaluation
```

Estimated total:

```text
6,000–15,000 GPU-hours
$20k–$100k GPU cost
8–14 weeks
```

#### Publishable program

Goal: produce external-quality paper / technical report / release.

Expected experiments:

```text
125M multi-seed
350M multi-seed
1B main runs
global KV full ablations
parallel passing beta
full benchmark suite
```

Estimated total:

```text
20,000–60,000 GPU-hours
$60k–$400k GPU cost
4–6 months
```

---

## 18. Timeline

Assuming:

```text
1 research lead
2 research engineers
0.5 infra/data engineer
```

### Month 1: Baseline and fixed route

Deliverables:

```text
BRIAN-B125 baseline
route wrapper
fixed route equivalence
router imitation logs
position-state initial ablation
```

Decision:

```text
Does route wrapping preserve baseline behavior?
```

### Month 2: Free router and output action

Deliverables:

```text
scheduled routing
output action
cost/balance/location losses
125M route-core ablation
```

Decision:

```text
Is router + block-position trainable?
```

### Month 3: 350M scaling

Deliverables:

```text
BRIAN-B350 baseline
BRIAN-R350 main route-core model
difficulty-conditioned pseudo routing
compute-quality curve
reasoning eval
```

Decision:

```text
Does the behavior scale beyond toy size?
```

### Month 4: Global KV

Deliverables:

```text
local/global KV adapters
window + sink global cache
long-context eval
KV memory/performance curve
```

Decision:

```text
Does canonical global memory help under memory constraints?
```

### Month 5–6: Larger run and parallel passing beta

Deliverables:

```text
BRIAN-R1B, if justified
parallel passing beta
visible CoT reduction evaluation
complete technical report
```

Decision:

```text
Is the architecture worth continued scale-up?
```

---

## 19. Engineering Milestones for Codex / Implementation Agent

This section is intentionally macro-level. Code-level implementation details should be planned separately.

### M0: Repository skeleton

Expected repo structure:

```text
BRIAN-Sphere-LLM/
  README.md
  PROJECT_PLAN.md
  brian_sphere_llm/
    model/
    routing/
    position/
    memory/
    losses/
    data/
    eval/
    train/
    configs/
  experiments/
  reports/
```

### M1: Baseline training path

Requirements:

```text
train fixed decoder-only baseline
save checkpoints
compute validation loss
produce baseline difficulty scores
```

### M2: Fixed route wrapper

Requirements:

```text
split pre / route pool / post
force original route path
verify loss parity with baseline
log routed hidden norms and route state
```

### M3: Router imitation

Requirements:

```text
pseudo route generator
route imitation loss
route accuracy metrics
route entropy / block load logging
```

### M4: Scheduled free routing

Requirements:

```text
pseudo/router mixing schedule
top-1 routing
top-2 weighted fusion later
cost/balance/location losses
```

### M5: Output action

Requirements:

```text
OUT as terminal action
exit-step histogram
forced max-step exit fallback
compute-adjusted eval
```

### M6: Global KV

Requirements:

```text
canonical global code pool
window + sink policy
local/global adapters
global attention diagnostics
```

### M7: Parallel passing beta

Requirements:

```text
beam state
branch score
branch prune
branch cost
memory delta policy
```

---

## 20. Risk Register

### Risk 1: Router collapse

Symptoms:

```text
always selects same block
always follows original sequence
always exits early
never exits
```

Mitigation:

```text
extend pseudo-route imitation
increase balance loss
lower cost loss if early exit
increase cost loss if never exit
delay hard OUT behavior
increase location bias early
```

### Risk 2: Block-position state has no effect

Symptoms:

```text
no-position ablation equals main model
position state becomes constant
location distance has no structure
```

Mitigation:

```text
feed position to both router and blocks
increase position adapter strength gradually
add location loss
compare random vs open-arc initialization
check whether normalization layers erase position signal
```

### Risk 3: Free routing degrades LM loss

Symptoms:

```text
fixed route works
free route causes validation loss spike
```

Mitigation:

```text
slower scheduled routing
teacher/logit distillation from fixed baseline
top-1 before top-2
fixed min/max route steps
lower router temperature gradually
```

### Risk 4: Global KV becomes noise

Symptoms:

```text
global attention mass near zero
global on/off no difference
global cache worsens loss
```

Mitigation:

```text
initialize global read gate near zero
train global adapters after route core
limit write frequency
use simple window + sink first
start with per-block adapters before full per-head adapters
```

### Risk 5: Parallel passing explodes cost

Symptoms:

```text
branch count grows
memory usage grows uncontrollably
branch credit assignment unstable
```

Mitigation:

```text
postpone until route core works
beam <= 2 initially
branch score decay
branch cost loss
shared base memory + branch delta memory
```

---

## 21. First Formal Experiment Package

### Package A: BRIAN-R125 route core

Experiments:

```text
A0: fixed Transformer baseline
A1: fixed route wrapper
A2: router imitation, original sequential path
A3: router imitation, skip/recur pseudo route
A4: free router + block-position
A5: no block-position ablation
A6: no output action ablation
A7: no location loss ablation
```

Data:

```text
2B–5B tokens
FineWeb-Edu subset + TinyStories + synthetic routing
```

Goal:

```text
prove router + block-position + output action can train stably
```

Expected cost:

```text
$5k–$20k GPU cost, depending on provider and number of reruns
```

### Package B: BRIAN-R350 scaling

Experiments:

```text
B0: 350M fixed baseline
B1: 350M routed main
B2: 350M routed no-position
B3: 350M routed no-output-action
B4: 350M routed difficulty-conditioned route
```

Data:

```text
10B–30B tokens
FineWeb-Edu/FineWeb subset + reasoning mixture
```

Goal:

```text
show that route-core behavior scales beyond toy size
```

Expected cost:

```text
$15k–$80k GPU cost
```

### Package C: Global KV

Experiments:

```text
C0: local KV only
C1: local + global uncompressed
C2: local + global compressed
C3: global no sink
C4: global with sink
C5: window size sweep
C6: per-block adapter
C7: shared per-head low-rank delta
C8: per-block + per-head low-rank delta
```

Evaluation:

```text
RULER
LongBench subset
synthetic multi-hop memory
long arithmetic/program traces
```

Goal:

```text
prove canonical global KV has measurable memory or long-context value
```

Expected cost:

```text
$10k–$60k GPU cost
```

---

## 22. What Not To Do Early

Do not start with:

```text
7B+ model
full parallel passing
complex 3-tier KV memory
per-head full matrix global adapters
RL-style router training
large-scale SFT/RLHF
GSM8K SOTA chasing
```

These are later-stage extensions.

Early success is defined by route stability, position usefulness, compute controllability, and preservation of LM quality.

---

## 23. Project Decision Tree

```text
1. Can fixed route wrapper match baseline?
   No → fix architecture wrapping.
   Yes → continue.

2. Can router imitate pseudo routes?
   No → fix router input, position state, losses.
   Yes → continue.

3. Can scheduled free routing avoid collapse?
   No → extend pseudo routing, add distillation, reduce freedom.
   Yes → continue.

4. Does block-position matter?
   No → redesign position state or question geometry hypothesis.
   Yes → continue.

5. Does output action control compute?
   No → adjust cost loss / exit curriculum.
   Yes → continue.

6. Does 350M show scaling trend?
   No → do not go 1B.
   Yes → test global KV.

7. Does global KV help under memory constraints?
   No → keep local route-core as main project.
   Yes → consider 1B and parallel passing.
```

---

## 24. Reference Links

These links are included for planning context and should be refreshed before formal publication or major budget decisions.

### Data and benchmarks

- FineWeb-Edu dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb technical report: https://arxiv.org/abs/2406.17557
- TinyStories paper: https://arxiv.org/abs/2305.07759
- GSM8K dataset: https://github.com/openai/grade-school-math
- GSM8K Hugging Face page: https://huggingface.co/datasets/openai/gsm8k
- RULER benchmark: https://github.com/NVIDIA/RULER
- RULER paper: https://arxiv.org/abs/2404.06654
- LongBench paper: https://arxiv.org/abs/2308.14508
- LongBench repository: https://github.com/THUDM/LongBench

### Training scale and compute

- Chinchilla / Training Compute-Optimal Large Language Models: https://arxiv.org/abs/2203.15556

### Cloud pricing references

- RunPod pricing: https://www.runpod.io/pricing
- RunPod H100 SXM page: https://www.runpod.io/gpu-models/h100-sxm
- Lambda cloud: https://lambda.ai/
- AWS P5 instances: https://aws.amazon.com/ec2/instance-types/p5/

---

## 25. Current Project Priority

As of 2026-06-13, Package A on `r125_main_2b` has completed for A0-A7. The
current priority is Package A analysis and the route-core go/no-go decision,
not Global KV or parallel passing.

The completed validation package was:

```text
BRIAN-R125 route-core validation
```

Specifically:

1. Train fixed baseline.
2. Implement fixed route wrapper.
3. Train router imitation on sequential and skip/recur pseudo routes.
4. Enable scheduled free routing.
5. Validate block-position state.
6. Validate output action.

All A0-A7 runs reached the 2B-token target (`step=30518` at `batch_size: 32`),
kept only `checkpoint_latest`, and produced routing reports. Local generated
summary reports live under:

```text
experiments/generated/route_core_r125_2b_package/
```

The next decision should compare Package A evidence against the R125 go
criteria before choosing a 5B R125 follow-up, R350 scaling, or additional
route-core fixes. Only after that should the project add Global KV memory.

---

## 26. One-Sentence Project Definition

**BRIAN-Sphere-LLM is a latent routing Transformer framework that learns to navigate a block/operator space with explicit computation-position state, terminal output actions, and optional shared canonical memory, replacing fixed middle-layer depth with adaptive internal computation paths.**
