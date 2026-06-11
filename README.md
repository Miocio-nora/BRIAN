# BRIAN-Sphere-LLM

**BRIAN-Sphere-LLM** is a long-range research and engineering project for replacing the fixed middle-depth computation path of a Transformer with a learnable latent routing graph.

BRIAN stands for:

> **Block-Routed Inference with Adaptive Navigation over a Latent Operator Sphere**

The short project name is **BRIAN-Sphere** or **BRIAN**. The planned Python package name is `brian_sphere_llm`.

## Current Status

This repository contains the project plan, Codex engineering guidance, and a runnable v0.1 PyTorch research scaffold.

Implemented v0.1 pieces:

- reproducible data manifest and fixed-length token packing;
- synthetic routing smoke data;
- LLaMA-like decoder-only baseline;
- BRIAN route-core wrapper with pre / route-pool / post blocks;
- block-position state, latent router, pseudo policies, and route metrics;
- Stage 0 baseline, Stage 1 fixed route, Stage 2 router imitation, and Stage 3 scheduled routing entrypoints;
- top-2 weighted route fusion for free/scheduled routing;
- hard `OUT` terminal behavior for Stage 4;
- minimal canonical Global KV path with sink + sliding window retention for Stage 5;
- experimental Stage 6 parallel passing with beam scoring, pruning, shared base Global KV, and per-branch delta memory;
- JSONL train/eval logs, model stats, checkpoint save/resume, and routing report generation;
- B200-compatible conda environment using PyTorch CUDA 12.8 wheels.

The immediate priority is **BRIAN-R125 route-core validation**:

1. Train a fixed decoder-only Transformer baseline.
2. Implement a fixed route wrapper around the middle blocks.
3. Train router imitation on sequential and skip/recurrent pseudo routes.
4. Enable scheduled free routing.
5. Validate the block-position state.
6. Validate `OUT` as a terminal routing action.

See [BRIAN-Sphere-LLM_PROJECT_PLAN.md](./BRIAN-Sphere-LLM_PROJECT_PLAN.md) for the full technical plan.
See [CODEX_GUIDANCE.md](./CODEX_GUIDANCE.md) for implementation guidance.

## Setup

Create the project environment:

```bash
conda env create -f environment.yml
conda activate brian-sphere
```

The environment pins PyTorch CUDA 12.8 wheels:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
```

This is intended for Blackwell/B200 hosts with a CUDA 12.8-capable driver.

## Quick Smoke Run

Prepare a tiny synthetic dataset:

```bash
python scripts/prepare_data.py --config configs/data/r125_tiny_debug.yaml
```

Run the local smoke stages:

```bash
python scripts/train.py --config configs/train/stage0_tiny_debug.yaml
python scripts/train.py --config configs/train/stage1_tiny_debug.yaml
python scripts/train.py --config configs/train/stage2_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_top2_tiny_debug.yaml
python scripts/train.py --config configs/train/stage4_tiny_debug.yaml
python scripts/train.py --config configs/train/stage5_tiny_debug.yaml
python scripts/train.py --config configs/train/stage6_tiny_debug.yaml
```

Generate a routing report:

```bash
python scripts/eval.py --config configs/eval/routing_eval.yaml --run <run_dir>
```

Routing reports include `route_entropy`, `block_load_entropy`, `route_path_diversity`, block histograms, exit distributions, active block evals, and position/global/parallel diagnostics when available.

Generate a stage gate report across multiple runs:

```bash
python scripts/make_stage_gate_report.py \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> <stage5_run> <stage6_run>
```

Include Stage 4 cost-control evidence in the stage gate:

```bash
python scripts/eval.py \
  --config configs/eval/stage_gate_eval.yaml \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> \
  --cost-control-report <cost_control_report.json>
```

Generate the required difficulty-step diagnostic for a routed run:

```bash
python scripts/eval.py \
  --config configs/eval/difficulty_step_eval.yaml \
  --baseline-run <stage0_baseline_run> \
  --routed-run <routed_run>
```

This writes `difficulty_step_report.json` and per-sample JSONL rows into the routed run directory. The key metric is `difficulty_step_correlation = corr(baseline_cross_entropy, route_steps)`.

Run the lightweight synthetic reasoning eval:

```bash
python scripts/eval.py \
  --config configs/eval/reasoning_eval.yaml \
  --run <run_dir> \
  --sample-count 24
```

This writes a reasoning report with exact-match accuracy, teacher-forced target token accuracy, per-task/per-difficulty summaries, and routed compute diagnostics.

Run the lightweight long-context / Global KV eval:

```bash
python scripts/eval.py \
  --config configs/eval/long_context_eval.yaml \
  --run <stage5_global_kv_run> \
  --sample-count 12
```

This writes a needle-retrieval / two-hop tracing report with exact-match accuracy, teacher-forced target token accuracy, truncation rate, estimated fp16 KV/global-code memory budgets, and Global KV routing diagnostics.

Compare a local-KV baseline against one or more Global KV candidates:

```bash
python scripts/eval.py \
  --config configs/eval/long_context_eval.yaml \
  --run <stage4_local_kv_run> \
  --output reports/long_context_local.json

python scripts/eval.py \
  --config configs/eval/long_context_eval.yaml \
  --run <stage5_global_kv_run> \
  --output reports/long_context_global.json

python scripts/eval.py \
  --config configs/eval/long_context_compare.yaml \
  --baseline-report reports/long_context_local.json \
  --reports reports/long_context_global.json \
  --output reports/long_context_compare.json
```

The comparison report checks that Global KV is active, the estimated Global KV code budget is below the local raw-KV context budget, and exact-match / teacher-forced quality is not worse than the local-KV report beyond `quality_tolerance`. Pass it into the Stage 5 gate:

```bash
python scripts/eval.py \
  --config configs/eval/stage_gate_eval.yaml \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> <stage5_run> \
  --long-context-compare-report reports/long_context_compare.json
```

Generate a compute-adjusted comparison report:

```bash
python scripts/eval.py \
  --config configs/eval/compute_report.yaml \
  --baseline-run <stage0_baseline_run> \
  --runs <stage0_baseline_run> <routed_run_1> <routed_run_2>
```

This writes `reports/compute_report.json` with parameter ratios, active layer eval ratios, estimated FLOPs/token, estimated GPU-hours, validation loss deltas, and throughput ratios.

Run the Stage 3 block-position smoke ablations:

```bash
python scripts/train.py --config configs/train/stage3_no_position_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_router_only_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_circular_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_random_tiny_debug.yaml
```

The formal BRIAN-R125 position ablation manifest is `configs/experiments/route_core_position_ablations.yaml`.

Resolve the full Package A BRIAN-R125 route-core manifest:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/route_core_r125_package.yaml \
  --dry-run
```

This manifest covers A0-A7 from the project plan: fixed baseline, fixed route wrapper, sequential router imitation, skip/recur router imitation, scheduled free routing with block-position state, no-position ablation, no hard output-action ablation, and no-location-loss ablation.

Resolve an experiment manifest without training:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_position_ablations.yaml \
  --dry-run
```

Run a bounded tiny experiment package:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_position_ablations.yaml \
  --include-baseline \
  --limit 2
```

Generate a cost-control report after a Stage 4 sweep:

```bash
python scripts/eval.py \
  --config configs/eval/cost_control_report.yaml \
  --runs <cost0_run> <cost001_run> <cost01_run> <cost05_run>
```

The fast smoke manifest is `configs/experiments/tiny_cost_control.yaml`; the BRIAN-R125 sweep manifest is `configs/experiments/route_core_cost_control.yaml`.

Run the Stage 5 Global KV ablation packages:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_global_kv.yaml \
  --dry-run
```

The fast smoke manifest is `configs/experiments/tiny_global_kv.yaml`; the BRIAN-R125 sweep manifest is `configs/experiments/route_core_global_kv.yaml`. These cover the local-KV baseline, no-sink Global KV, default sink+window Global KV, and a small cache-window sweep.

Compare top-k weighted fusion against parallel passing:

```bash
python scripts/eval.py \
  --config configs/eval/parallel_compare.yaml \
  --baseline-run <stage5_topk_global_kv_run> \
  --runs <stage6_parallel_run> \
  --output reports/parallel_compare.json
```

The comparison report checks that parallel branches are active, branch score margins are logged, validation loss is not worse than the baseline beyond `max_validation_loss_delta`, and active compute / estimated FLOPs stay under the configured ratios. Pass it into the Stage 6 gate:

```bash
python scripts/eval.py \
  --config configs/eval/stage_gate_eval.yaml \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> <stage5_run> <stage6_run> \
  --parallel-compare-report reports/parallel_compare.json
```

Run the Stage 6 parallel-passing packages:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_parallel_passing.yaml \
  --dry-run
```

The fast smoke manifest is `configs/experiments/tiny_parallel_passing.yaml`; the BRIAN-R125 sweep manifest is `configs/experiments/route_core_parallel_passing.yaml`. These cover PP0 top-k weighted fusion and PP1 beam-2 independent branch passing.

Resolve the first BRIAN-R350 scaling package:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/route_core_r350_scaling.yaml \
  --dry-run
```

This manifest covers the Package B scaffold from the project plan: B0 fixed 350M baseline, B1 routed main, B2 no-position ablation, B3 no hard output-action ablation, and B4 current `mixed_skip_recur` difficulty-pattern pseudo-route curriculum. It uses `configs/data/r350_main_10b.yaml` and B200-compatible bf16 training configs.

Run tests:

```bash
pytest
```

## Core Idea

A standard decoder-only Transformer executes a fixed sequence of blocks:

```text
input -> B1 -> B2 -> ... -> BL -> output
```

BRIAN-Sphere-LLM replaces part of the middle stack with a routeable block pool:

```text
input -> pre-blocks -> router-controlled latent block path -> OUT -> post-blocks / LM head
```

Instead of forcing every input through the same middle-layer sequence, the model can learn computation paths such as:

```text
B3 -> B5 -> B5 -> B2 -> B7 -> OUT
B2 -> B3 -> OUT
B4 -> B4 -> B6 -> B8 -> OUT
```

The first research goal is not to beat large public models. The first goal is to prove that the route-core system is trainable, measurable, and controllable at small scale.

## Minimal Route-Core System

The routed model maintains two states:

```text
S_r = (H_r, P_r)
```

Where:

- `H_r` is the content hidden state.
- `P_r` is the block-position / operator-position state.

The router selects an action:

```text
a_r in {B1, B2, ..., Bm, OUT}
```

If the action is an internal block, the model applies that block and updates position:

```text
H_{r+1} = B_{a_r}(H_r, P_r)
P_{r+1} = E_{a_r}
```

If the action is `OUT`, the model exits the latent routing loop and produces logits through the output/post-block path.

## First Target Architecture: BRIAN-R125

The recommended first serious model is a LLaMA-like decoder-only Transformer at roughly 110M-150M parameters.

| Item | Suggested value |
| --- | ---: |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| FFN | SwiGLU / gated MLP |
| Norm | RMSNorm |
| Token position | RoPE |
| Vocabulary | 32k tokenizer |
| Initial context | 2k, optionally 4k |
| Pre blocks | 2 |
| Route pool | 8 middle blocks |
| Post blocks | 2 |
| Route actions | 8 internal blocks + `OUT` |
| Max latent route steps | 4-8 |
| Initial routing | top-1 |
| Later routing | top-2 weighted fusion |
| Global KV | off for first route-core stage |
| Parallel passing | off |

Planned block split:

```text
pre:   B1, B2
pool:  B3, B4, B5, B6, B7, B8, B9, B10
post:  B11, B12
```

## Training Roadmap

The project deliberately avoids training the full system from scratch in one step. The planned curriculum is:

```text
fixed path -> pseudo routing -> scheduled free routing -> output action -> global KV -> parallel passing
```

Main stages:

| Stage | Goal |
| --- | --- |
| 0 | Train a fixed Transformer baseline. |
| 1 | Convert middle blocks into a route pool while forcing the original path. |
| 2 | Train pseudo skip/recurrent routing with imitation loss. |
| 3 | Gradually allow router-controlled forward paths. |
| 4 | Enable `OUT` as a hard terminal action. |
| 5 | Add optional canonical global KV memory after route-core stability. |
| 6 | Add optional parallel latent passing only after earlier stages succeed. Experimental beam-2 model integration is available behind Stage 6 configs. |

## Required Diagnostics

Routing behavior is a first-class research output. Every routed model should report:

- validation loss and perplexity;
- route entropy;
- block load entropy;
- average route steps;
- exit step distribution;
- route path diversity;
- recurrent and skip ratios;
- location distance;
- active block evaluations per token;
- difficulty-step correlation;
- `OUT` probability by difficulty.

The most important route-core diagnostic is:

```text
corr(baseline_cross_entropy, route_steps)
```

A positive correlation suggests that harder examples receive more internal computation.

## Go / No-Go Criteria

Proceed from BRIAN-R125 route-core experiments to BRIAN-R350 only if:

- the fixed route wrapper stays within 1-3% validation loss of the fixed baseline;
- router imitation accuracy exceeds 95%;
- scheduled free routing does not collapse validation loss;
- average route steps can be controlled by cost loss;
- block load does not collapse to one internal block;
- the block-position ablation shows a measurable difference;
- `OUT` is neither always early nor never used.

If these fail, the plan is to fix route training, position design, and curriculum before scaling.

## Planned Repository Structure

The implementation layout is:

```text
BRIAN-Sphere-LLM/
  README.md
  BRIAN-Sphere-LLM_PROJECT_PLAN.md
  CODEX_GUIDANCE.md
  environment.yml
  pyproject.toml
  configs/
  src/brian_sphere_llm/
  scripts/
  tests/
  data/
  runs/
  experiments/
  reports/
```

## What Is Intentionally Deferred

The early project should not start with:

- 7B+ models;
- full parallel passing;
- complex multi-tier KV memory;
- per-head full-matrix global adapters;
- RL-style router training;
- large-scale SFT/RLHF;
- benchmark chasing before route stability is proven.

Early success is defined by route stability, position usefulness, compute controllability, and preservation of language-modeling quality.

## Project Definition

**BRIAN-Sphere-LLM is a latent routing Transformer framework that learns to navigate a block/operator space with explicit computation-position state, terminal output actions, and optional shared canonical memory, replacing fixed middle-layer depth with adaptive internal computation paths.**
