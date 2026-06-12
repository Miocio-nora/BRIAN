# BRIAN-Sphere-LLM — Codex Guidance for Data, Code, and Training Deployment

**Project:** BRIAN-Sphere-LLM  
**Document role:** Long-range engineering guidance for Codex and implementation agents  
**Date:** 2026-06-11  
**Status:** v0.1 planning guidance

---

## 0. Executive summary

BRIAN-Sphere-LLM should be implemented as a staged research system, not as a one-shot full architecture. The first engineering goal is to produce a stable, measurable, reproducible training stack that can answer this question:

> Can a decoder-only Transformer replace a fixed middle-layer sequence with a latent routed block pool, while using a block-position state and an output block as a terminal action?

The first milestone should **not** include parallel passing or the full global KV system. Those come after the route core is stable.

Recommended first serious target:

```text
Model:      BRIAN-R125
Scale:      ~125M parameters
Hardware:   1× H100 80GB is enough
Data:       2B–5B tokens for main run; 100M–500M tokens for smoke/pilot
Context:    2k initially; 4k after stable
Routing:    middle 8 blocks, max_route_steps 4–8, top-1 first, top-2 later
Global KV:  off in route-core phase
Parallel:   off
```

Recommended second target:

```text
Model:      BRIAN-R350
Scale:      ~350M parameters
Hardware:   1× H100 for pilots; 4×–8× H100 recommended for serious runs
Data:       10B–30B tokens
Global KV:  add only after route core succeeds
```

Recommended third target:

```text
Model:      BRIAN-R1B
Scale:      ~1B parameters
Hardware:   1× H100 can fit debug/short runs with checkpointing;
            8× H100 minimum recommended for serious pretraining;
            16×–32× H100 if wall-clock matters.
Data:       50B–100B tokens for serious validation
```

---

## 1. Hardware answer: can H100 handle 125M and 1B?

### 1.1 Short answer

Yes, but distinguish **memory feasibility** from **project practicality**.

| Scale | 1× H100 80GB | Recommended for serious training | Notes |
|---|---:|---:|---|
| 125M | Yes, comfortably | 1× H100 | Single GPU is enough for route-core experiments and ablations. |
| 350M | Yes for pilots | 4×–8× H100 | Single GPU works, but ablation throughput becomes slow. |
| 1B | Yes for debug/short training if using BF16 + activation checkpointing + small microbatch | 8× H100 minimum; 16×–32× preferred | Memory can fit, but serious pretraining is wall-clock limited. |

### 1.2 Practical interpretation

- **125M:** H100 is more than enough. The bottleneck will be data pipeline correctness, routing stability, and ablation count.
- **350M:** One H100 can be used for prototype runs, but serious multi-run research should use at least 4× H100.
- **1B:** One H100 can run the model, but should be treated as a debug setup. For 50B–100B token pretraining, use 8× H100 or larger.
- Dynamic routing reduces effective throughput. Expect BRIAN models to be slower than a vanilla Transformer with the same parameter count until sparse dispatch and routing kernels are optimized.

### 1.3 Cost anchor

Use this as planning guidance, not as a fixed quote:

```text
RunPod H100 SXM 80GB public on-demand anchor: ~$3.29 / GPU-hour
Lambda H100 SXM 80GB public on-demand anchor: ~$3.99 / GPU-hour
Conservative planning range: $3.3–$7.0 / GPU-hour
```

Approximate daily GPU cost:

| Cluster | $3.3/GPU-hour | $4.0/GPU-hour | $7.0/GPU-hour |
|---|---:|---:|---:|
| 1× H100, 24h | ~$79 | ~$96 | ~$168 |
| 4× H100, 24h | ~$317 | ~$384 | ~$672 |
| 8× H100, 24h | ~$634 | ~$768 | ~$1,344 |
| 16× H100, 24h | ~$1,267 | ~$1,536 | ~$2,688 |

---

## 2. Engineering philosophy for Codex

Codex should treat this repository as a research training system. Prioritize correctness, observability, reproducibility, and stage gates over cleverness.

### 2.1 Non-negotiable principles

1. **Every architecture change must have a baseline and an ablation.**
2. **Every routed forward pass must log route behavior.**
3. **No full global KV before route core is stable.**
4. **No parallel passing before top-k weighted routing is stable.**
5. **No 1B training before 125M and 350M give interpretable results.**
6. **Every dataset shard must be reproducible from a manifest.**
7. **Every run must be resumable from checkpoints.**
8. **Every major config must be represented as a YAML file, not hard-coded.**

### 2.2 First implementation priority

The first deliverable is not model quality. It is a repeatable experimental loop:

```text
data subset → tokenization → baseline training → fixed-route wrapper → router imitation → scheduled free routing → metrics report
```

---

## 3. Repository structure

Recommended repo root:

```text
BRIAN-Sphere-LLM/
  README.md
  PROJECT_PLAN.md
  CODEX_GUIDANCE.md
  pyproject.toml
  requirements.txt

  configs/
    data/
      r125_smoke.yaml
      r125_main_2b.yaml
      r350_main_10b.yaml
      r1b_pilot.yaml
    model/
      baseline_125m.yaml
      brian_r125.yaml
      brian_r350.yaml
      brian_r1b.yaml
    train/
      stage0_baseline.yaml
      stage1_fixed_route.yaml
      stage2_router_imitation.yaml
      stage3_pseudo_skip_recur.yaml
      stage3_scheduled_free_routing.yaml
      stage4_scheduled_free_routing.yaml
      stage4_output_action.yaml
      stage5_output_action.yaml
      stage5_global_kv.yaml
      stage6_parallel_passing.yaml
      stage7_parallel_passing.yaml
    eval/
      lm_eval.yaml
      routing_eval.yaml
      reasoning_eval.yaml
      long_context_eval.yaml

  src/
    brian_sphere_llm/
      __init__.py
      data/
        manifest.py
        download.py
        filter.py
        tokenize.py
        pack.py
        dataloader.py
        synthetic_routing.py
      model/
        llama_backbone.py
        baseline.py
        brian_model.py
        route_block.py
        exit_block.py
      routing/
        router.py
        block_position.py
        pseudo_policy.py
        schedule.py
        metrics.py
      memory/
        global_cache.py
        read_adapter.py
        write_adapter.py
      losses/
        lm_loss.py
        route_loss.py
        balance_loss.py
        cost_loss.py
        location_loss.py
      train/
        trainer.py
        checkpoint.py
        stage_runner.py
      eval/
        perplexity.py
        routing_report.py
        reasoning.py
        long_context.py
      utils/
        logging.py
        seed.py
        distributed.py
        config.py

  scripts/
    prepare_tokenizer.py
    prepare_data.py
    train.py
    eval.py
    make_routing_report.py
    estimate_compute.py

  tests/
    test_data_manifest.py
    test_tokenization_pack.py
    test_baseline_forward.py
    test_fixed_route_equivalence.py
    test_router_shapes.py
    test_position_update.py
    test_loss_terms.py
    test_checkpoint_resume.py

  runs/
    .gitkeep

  data/
    README.md
    manifests/
    raw/
    processed/
    tokenized/
    shards/
```

---

## 4. Dataset plan

### 4.1 Do not download full web-scale datasets initially

FineWeb and FineWeb-Edu are much larger than needed for early experiments. Codex should implement streaming/subset preparation first.

Initial data targets:

| Dataset recipe | Tokens | Purpose |
|---|---:|---|
| `r125_tiny_smoke` | 10M–50M | Dataloader, tokenizer, training loop smoke |
| `r125_smoke` | 100M–500M | First baseline and fixed-route checks |
| `r125_main_2b` | 2B | First serious route-core validation |
| `r125_main_5b` | 5B | Stronger 125M result if 2B looks promising |
| `r350_main_10b` | 10B | First 350M trend check |
| `r350_main_30b` | 30B | Stronger 350M run |
| `r1b_pilot_10b` | 10B | 1B architecture pilot only |
| `r1b_main_50b` | 50B | Serious 1B validation |

### 4.2 Recommended data mixture

#### R125 main mixture

```text
FineWeb-Edu subset:          70%
TinyStories / simple text:   10%
Synthetic routing tasks:     10%
Math / symbolic / QA text:   5%
Code / structured text:      5%
```

#### R350 main mixture

```text
FineWeb-Edu / FineWeb:       80%
Synthetic routing tasks:     5%–10%
Math / symbolic / QA text:   5%
Code / structured text:      5%
```

#### R1B main mixture

```text
FineWeb-Edu / FineWeb:       85%–90%
Synthetic routing tasks:     2%–5%
Math / symbolic / QA text:   5%
Code / structured text:      5%
```

Synthetic routing proportion should decrease with scale. It is a training scaffold, not the final target distribution.

---

## 5. Data pipeline requirements

### 5.1 Required artifacts

Every prepared dataset must produce:

```text
data/manifests/<recipe_name>.jsonl
  Each row:
    sample_id
    source_dataset
    source_url_or_id
    split
    token_count
    byte_count
    sha256_text
    sha256_tokens
    license
    path
    mixture_tag
    created_at
```

Tokenized output:

```text
data/tokenized/<recipe_name>/
  tokenizer.json
  tokenizer_config.json
  train.bin or train_*.bin
  train.idx or train_*.idx
  val.bin
  val.idx
  manifest.jsonl
  stats.json
```

Stats file must include:

```text
num_documents
num_tokens_train
num_tokens_val
avg_tokens_per_doc
sequence_length
vocab_size
source_mixture_realized
sha256_manifest
```

### 5.2 Tokenizer guidance

Use one of two approaches:

#### Option A: train project tokenizer

Recommended for from-scratch clean research.

```text
Algorithm: BPE or unigram
Vocab:     32k
Training sample: 10GB–50GB representative text
Special tokens:
  <bos>
  <eos>
  <pad>
  <unk>
  <route_sink>    optional, not for initial LM if unnecessary
```

Pros:

```text
clean licensing
fully reproducible
no dependency on external model tokenizer
```

Cons:

```text
slightly harder comparability
```

#### Option B: use public existing tokenizer

Acceptable for speed if license allows.

Codex must record:

```text
tokenizer name
license
revision hash
vocab size
special tokens
```

### 5.3 Packing guidance

Use fixed-length packed sequences.

Initial context lengths:

```text
R125 smoke:  1024 or 2048
R125 main:   2048
R350 main:   2048 → 4096
R1B pilot:   4096
```

Do not use variable sequence lengths in the first implementation. Fixed sequence length simplifies routing metrics, throughput comparison, and checkpoint reproducibility.

### 5.4 Validation split

Use stable held-out validation shards.

Recommended:

```text
validation tokens: 20M–100M for R125
validation tokens: 100M+ for R350/R1B
```

Validation data must not be reshuffled between runs. All route ablations should use the same validation set.

---

## 6. Synthetic routing data

Synthetic routing data is not meant to make the model good at a benchmark. It exists to give the router early stable signals.

### 6.1 Task families

Codex should implement lightweight generators for:

1. Copy / reverse / transform sequences
2. Multi-step arithmetic strings
3. Simple symbolic rewriting
4. Parentheses / stack-like patterns
5. Repeated transformation requiring recurrence
6. Easy/medium/hard variants for difficulty-conditioned route length

### 6.2 Labels to generate

For each synthetic sample, generate optional route metadata:

```text
pseudo_route_type: advance | skip | recur | mixed | early_exit | late_exit
pseudo_route_length
expected_recurrence_count
expected_skip_count
difficulty_bin
```

These labels are used only for router imitation and analysis. They should not be required by the base LM loss.

---

## 7. Model families and configs

### 7.1 Baseline models

Codex must implement vanilla decoder-only baselines first.

#### Baseline 125M

```yaml
model_name: baseline_125m
architecture: decoder_only_llama_like
layers: 12
d_model: 768
n_heads: 12
ffn_type: swiglu
norm: rmsnorm
token_position: rope
context_length: 2048
vocab_size: 32000
```

#### Baseline 350M

```yaml
model_name: baseline_350m
layers: 24
d_model: 960
n_heads: 16
ffn_type: swiglu
norm: rmsnorm
token_position: rope
context_length: 2048_or_4096
vocab_size: 32000
```

#### Baseline 1B

```yaml
model_name: baseline_1b
layers: 32
d_model: 1536
n_heads: 24
ffn_type: swiglu
norm: rmsnorm
token_position: rope
context_length: 4096
vocab_size: 32000
```

Actual parameter counts should be confirmed by the code and recorded in `model_stats.json` for every config.

---

## 8. BRIAN route-core architecture

### 8.1 R125 route-core

```yaml
model_name: brian_r125
base: baseline_125m
pre_blocks: 2
route_pool_blocks: 8
post_blocks: 2
route_actions:
  internal_blocks: 8
  output_blocks: 1
block_position_dim: 64
max_route_steps: 4
top_k: 1
later_top_k: 2
global_kv: false
parallel_passing: false
```

### 8.2 R350 route-core

```yaml
model_name: brian_r350
base: baseline_350m
pre_blocks: 4
route_pool_blocks: 16
post_blocks: 4
route_actions:
  internal_blocks: 16
  output_blocks: 1
block_position_dim: 128
max_route_steps: 6_to_12
top_k: 1_to_2
global_kv: phase_2_only
parallel_passing: false
```

### 8.3 R1B route-core

```yaml
model_name: brian_r1b
base: baseline_1b
pre_blocks: 4_to_6
route_pool_blocks: 20_to_24
post_blocks: 4_to_6
block_position_dim: 128_or_256
max_route_steps: 8_to_16
top_k: 2
global_kv: true_after_route_core
parallel_passing: experimental_only
```

---

## 9. Routing states and required metrics

Each route step maintains:

```text
H_r: content hidden state
P_r: block-position state
route_logits_r
route_probs_r
selected_action_r
exit_flag_r
```

Codex must expose these in training logs:

```text
route_entropy
block_load_entropy
top1_block_histogram
topk_block_histogram
average_route_steps
exit_step_distribution
p_output_mean
skip_ratio
recur_ratio
advance_ratio
location_distance_mean
position_norm_mean
cost_loss
balance_loss
location_loss
route_imitation_accuracy
```

Critical diagnostic:

```text
corr(baseline_sample_loss, route_steps)
```

This checks whether difficult samples actually receive more internal computation.

---

## 10. Training stages

### Stage 0 — Vanilla baseline

Goal:

```text
Train or load a normal fixed-depth Transformer baseline.
```

Required outputs:

```text
baseline checkpoint
validation loss
sample-level validation CE for difficulty bins
model_stats.json
training throughput report
```

Exit criteria:

```text
loss curve stable
checkpoint resume works
validation deterministic across repeated eval
```

---

### Stage 1 — Fixed route wrapper

Goal:

```text
Wrap middle blocks as a route pool but force original sequential path.
```

Path:

```text
B1 → B2 → ... → Bm → OUT
```

Router can be trained to imitate this path but must not control forward yet.

Exit criteria:

```text
fixed-route loss within 1%–3% of baseline
route imitation accuracy > 98%
no hidden shape mismatch
position state stays finite and normalized
```

---

### Stage 2 — Router imitation: sequential + skip/recur

Goal:

```text
Teach router stable pseudo-routes before allowing free routing.
```

Pseudo actions:

```text
advance: B_i → B_{i+1}
skip:    B_i → B_{i+2}
recur:   B_i → B_i
exit:    B_i → OUT
```

Suggested pseudo-policy:

```text
easy samples:    skip-heavy, early OUT
medium samples:  mostly sequential
hard samples:    recurrent-heavy, late OUT
```

Difficulty signal:

```text
baseline_sample_ce → difficulty_bin → pseudo_route_length
```

Exit criteria:

```text
route imitation accuracy > 90% for mixed pseudo-policy
LM loss does not diverge
block usage non-degenerate
exit action supervised but not yet hard-enabled
```

---

### Stage 3 — Scheduled free routing

Goal:

```text
Gradually let router control the actual forward path.
```

Schedule example:

```text
90% pseudo / 10% router
70% pseudo / 30% router
50% pseudo / 50% router
20% pseudo / 80% router
0% pseudo / 100% router
```

Route imitation loss should decay:

```text
lambda_route: 1.0 → 0.5 → 0.2 → 0.05
```

Exit criteria:

```text
free routing validation loss does not collapse
route entropy remains above minimum threshold
average route steps controlled by cost loss
router does not always choose the same block
```

---

### Stage 4 — Output block as hard terminal action

Goal:

```text
Allow OUT action to terminate the latent route loop.
```

Rule:

```text
if top1_action == OUT:
    exit latent loop
else:
    continue internal routing
```

Do not exit merely because OUT appears in top-k. OUT must be top-1.

Exit criteria:

```text
model neither exits immediately nor never exits
average route steps follows cost loss changes
harder samples tend to exit later than easier samples
```

---

### Stage 5 — Global KV pool

Only start after Stage 4 succeeds.

Goal:

```text
Add canonical global compressed memory alongside local block memory.
```

First implementation:

```text
local KV: normal per-block local computation
global KV: canonical code pool
cache policy: sink + sliding window
compression: local → global write adapter
read: global → local read adapter
```

Do not implement heavy-hitter, three-tier memory, or branch memory in this stage.

Exit criteria:

```text
global read gate becomes non-zero
global attention mass measurable
global KV does not hurt standard LM loss significantly
memory-constrained long-context eval improves over local-only window
```

---

### Stage 6 — Parallel passing

Do not implement until Stages 0–5 are stable.

Goal:

```text
Change top-k weighted mixture into independent latent branches.
```

Required branch state:

```text
H_branch
P_branch
branch_score
base_memory_ref
branch_delta_memory
```

Required pruning:

```text
keep top-B branches
apply branch cost
prune weak branches
```

Exit criteria:

```text
branch count bounded
memory does not scale uncontrollably
reasoning benchmarks improve enough to justify compute
```

---

## 11. Loss terms

Total loss:

```text
L = L_LM
  + λ_route   * L_route_imitation
  + λ_balance * L_balance
  + λ_cost    * L_cost
  + λ_loc     * L_location
```

### 11.1 LM loss

Use normal next-token cross-entropy.

For early causality safety, first training mode may compute CE on final positions or fully causal per-position once router is verified.

### 11.2 Route imitation loss

Use CE or KL from pseudo-policy to router output.

### 11.3 Balance loss

Apply only to internal route blocks.

Do not include OUT in block balance loss.

### 11.4 Cost loss

Penalize expected internal computation.

```text
cost ≈ expected active internal block evaluations
```

This prevents endless internal routing.

### 11.5 Location loss

Encourage selected blocks to be geometrically consistent with the current block-position state.

This should be stronger early and weaker later.

### 11.6 Output-weighted CE caution

Do not directly use:

```text
loss = p_out * CE
```

unless `p_out` is stop-gradient. Otherwise, router can reduce loss by avoiding output on hard examples.

Preferred for later multi-output experiments:

```text
L_out = -log Σ_o p(o) exp(-CE_o)
```

For first implementation, keep CE separate and use cost/route losses to train output timing.

---

## 12. Global KV details for later implementation

### 12.1 Memory structure

```text
M_global = M_sink ∪ M_window
```

- `M_sink`: never evicted, small fixed set of slots
- `M_window`: recent compressed codes, sliding window

### 12.2 Adapter design

Start simple:

```text
per-block base adapter
optional per-head low-rank delta
```

Do not start with full per-block × per-head dense matrices if memory or kernel overhead becomes painful.

### 12.3 Evaluation for global KV

Global KV must be judged by:

```text
KV bytes/token
long-context accuracy
global attention mass
global read gate values
sink usage
local-only vs local+global under same memory budget
```

Not only by standard validation loss.

---

## 13. Evaluation plan

### 13.1 Core LM eval

```text
validation loss
perplexity
throughput tokens/sec
active block evals/token
```

### 13.2 Routing eval

Generate a routing report every checkpoint interval:

```text
route entropy
block load histogram
path examples
exit step histogram
position trajectory plots
cost-quality curve
```

### 13.3 Reasoning eval

Use small, fast evaluations initially:

```text
synthetic arithmetic
symbolic rewrite
copy/reverse/transform
short multi-hop QA
GSM8K subset after SFT or instruction adaptation, not as first pretraining metric
```

### 13.4 Long-context eval

Use only after global KV is implemented:

```text
needle retrieval
synthetic multi-hop tracing
RULER subset
LongBench subset
```

---

## 14. Ablation matrix

### 14.1 Route core

```text
A0 baseline fixed Transformer
A1 fixed route wrapper
A2 recurrent-only
A3 skip-only
A4 router without block-position
A5 router with block-position
A6 router with OUT disabled
A7 router with OUT enabled
A8 top-1 routing
A9 top-2 routing
```

### 14.2 Block-position

```text
B0 no block-position
B1 random position init
B2 circular/open-arc position init
B3 position only to router
B4 position to router + block
B5 no location bias
B6 no location loss
```

### 14.3 Global KV

```text
C0 local only
C1 local + global uncompressed
C2 local + global compressed
C3 global no sink
C4 global with sink
C5 window size sweep
C6 per-block adapter
C7 shared per-head low-rank delta
C8 per-block + per-head low-rank delta
```

### 14.4 Parallel passing

```text
D0 top-k weighted sum
D1 top-k independent branches
D2 beam size 2
D3 beam size 4
D4 branch cost off/on
D5 shared base memory + branch delta memory
```

---

## 15. Run naming convention

Use deterministic run names:

```text
<date>_<model>_<stage>_<data>_<context>_<seed>
```

Example:

```text
20260611_brian_r125_stage3_r125main2b_ctx2048_seed1
```

Every run directory should include:

```text
config_resolved.yaml
model_stats.json
data_manifest_ref.json
train_log.jsonl
eval_log.jsonl
routing_report.json
checkpoint_latest/
checkpoint_best/
```

---

## 16. Stage gates

Do not advance stages unless the gate is satisfied.

| Stage | Gate |
|---|---|
| Stage 0 → 1 | baseline trains, resumes, evaluates deterministically |
| Stage 1 → 2 | fixed-route wrapper within 1%–3% baseline loss |
| Stage 2 → 3 | router imitation stable, no collapse |
| Stage 3 → 4 | free routing does not destroy LM loss |
| Stage 4 → 5 | OUT action produces controllable exit distribution |
| Stage 5 → 6 | global KV shows measurable non-zero usage and at least one memory-budget benefit |
| Stage 6 → scale-up | parallel branch benefit justifies compute overhead |

---

## 17. Minimum acceptance criteria for first milestone

The first milestone is complete when Codex can run:

```text
1. prepare 100M-token smoke dataset
2. train baseline_125m for a short smoke run
3. train brian_r125 fixed-route wrapper
4. train router imitation on pseudo sequential path
5. train scheduled free routing for a short run
6. produce routing report and validation report
```

Minimum success:

```text
training does not crash
checkpoint resume works
fixed-route loss close to baseline
router logs are recorded
route entropy and block histogram are visible
OUT action can be supervised, even if hard exit is disabled
```

---

## 18. Recommended first three commands for the repo

Codex should aim to make these possible:

```bash
python scripts/prepare_data.py --config configs/data/r125_smoke.yaml
python scripts/train.py --config configs/train/stage0_baseline.yaml
python scripts/train.py --config configs/train/stage1_fixed_route.yaml
```

Then:

```bash
python scripts/train.py --config configs/train/stage2_router_imitation.yaml
python scripts/train.py --config configs/train/stage3_pseudo_skip_recur.yaml
python scripts/train.py --config configs/train/stage3_scheduled_free_routing.yaml
python scripts/eval.py --config configs/eval/routing_eval.yaml --run <stage3_scheduled_free_routing_run>
```

---

## 19. Resource planning by phase

### 19.1 R125 route-core phase

```text
Hardware: 1× H100 80GB
Data: 100M smoke → 2B main → 5B optional
Expected use: many short runs, many ablations
Budget class: low thousands USD for useful internal conclusion
```

### 19.2 R350 phase

```text
Hardware: 4×–8× H100 recommended
Data: 10B–30B tokens
Expected use: fewer but longer runs
Budget class: low-to-mid five figures USD depending ablation count
```

### 19.3 R1B phase

```text
Hardware: 8× H100 minimum for serious run; 16×–32× preferred
Data: 50B–100B tokens
Expected use: only after R125/R350 results justify scaling
Budget class: mid five figures or higher for serious multi-run validation
```

---

## 20. References for planning

These are source anchors used for resource and dataset planning. Codex does not need to scrape them during implementation, but the project documentation should preserve the links.

- NVIDIA H100 official specs: https://www.nvidia.com/en-us/data-center/h100/
- RunPod H100 pricing page: https://www.runpod.io/pricing
- Lambda cloud pricing page: https://lambda.ai/pricing
- FineWeb dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- TinyStories paper: https://arxiv.org/abs/2305.07759
- RULER repository: https://github.com/NVIDIA/RULER
- LongBench paper: https://arxiv.org/abs/2308.14508
- GSM8K repository: https://github.com/openai/grade-school-math

---

## 21. Final instruction to Codex

Implement the system in this order:

```text
1. Data manifest + tokenizer + packed dataset
2. Vanilla baseline model and trainer
3. Fixed-route BRIAN wrapper
4. Block-position state and metrics
5. Router imitation
6. Scheduled free routing
7. Output action
8. Global KV
9. Parallel passing
```

Do not skip directly to the full architecture.

The project succeeds only if we can explain *why* a result happened from route metrics, not merely report a validation loss.
