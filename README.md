# BRIAN-Sphere-LLM

**BRIAN-Sphere-LLM** is a long-range research and engineering project for replacing the fixed middle-depth computation path of a Transformer with a learnable latent routing graph.

BRIAN stands for:

> **Block-Routed Inference with Adaptive Navigation over a Latent Operator Sphere**

The short project name is **BRIAN-Sphere** or **BRIAN**. The planned Python package name is `brian_sphere_llm`.

## Current Status

This repository contains the project plan, Codex engineering guidance, and a runnable v0.1 PyTorch research scaffold.

Implemented v0.1 pieces:

- reproducible data manifest and fixed-length token packing;
- offline tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, and metadata) for smoke data;
- synthetic routing smoke data with manifest-retained pseudo-route metadata covering copy, reverse, transform, arithmetic, rewrite, parentheses, and repeated transforms;
- LLaMA-like decoder-only baseline;
- BRIAN route-core wrapper with pre / route-pool / post blocks;
- block-position state, latent router, pseudo policies, and route metrics;
- Stage 0 baseline, Stage 1 fixed route, Stage 2 router imitation, and Stage 3 scheduled routing entrypoints;
- top-2 weighted route fusion for free/scheduled routing;
- hard `OUT` terminal behavior for Stage 4;
- minimal canonical Global KV path with sink + sliding window retention for Stage 5;
- experimental Stage 6 parallel passing with beam scoring, pruning, shared base Global KV, and per-branch delta memory;
- JSONL train/eval logs with throughput, latency, CUDA memory diagnostics, model stats, data manifest references, checkpoint save/resume, and routing report generation;
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

Large 1B pilot and main-validation configs use `batch_size: 1`, `gradient_accumulation_steps: 4`, BF16, `activation_checkpointing: true`, and linear warmup with cosine LR decay so B200/H100 runs keep microbatch memory small while preserving a larger effective batch. Under DDP, gradient accumulation uses `no_sync()` for non-final microbatches, and routed 1B configs set `ddp_find_unused_parameters: true` for dynamic routing paths.

## Data Recipe Ladder

The planned fixed-length data recipes are declared under `configs/data/`:

| Recipe | Target tokens | Sequence length | Purpose |
|---|---:|---:|---|
| `r125_smoke` | 100M | 2048 | first baseline and fixed-route checks |
| `r125_main_2b` | 2B | 2048 | first serious R125 route-core validation |
| `r125_main_5b` | 5B | 2048 | stronger R125 run if 2B looks promising |
| `r350_main_10b` | 10B | 4096 | first R350 scaling trend check |
| `r350_main_30b` | 30B | 4096 | stronger R350 run |
| `r1b_pilot_10b` | 10B | 4096 | 1B architecture pilot |
| `r1b_main_50b` | 50B | 4096 | serious 1B validation after R350 evidence |

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

For multi-GPU jobs, launch the same training entrypoint with `torchrun`; the trainer reads `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`, uses a distributed train sampler, wraps the model with DDP, applies `no_sync()` during accumulated non-final microbatches, and writes checkpoints/reports only from rank 0:

```bash
torchrun --nproc_per_node=<gpu_count> scripts/train.py --config configs/train/stage0_r1b_baseline.yaml
```

Generate a routing report:

```bash
python scripts/eval.py --config configs/eval/routing_eval.yaml --run <run_dir>
```

Training writes `routing_report.json` at checkpoint saves by default (`write_routing_report_on_checkpoint: true`). Routing reports include `route_entropy`, `block_load_entropy`, `route_path_diversity`, path examples, block histograms, exit distributions, active block evals, skip/recur/advance ratios, position/location trajectories, train and inference timing, cost-quality curve points, and position/global/parallel diagnostics when available. The report includes `checks` and `overall_status`; Stage 1-6 gates require a passing routed report.

Generate a standard LM metrics report:

```bash
python scripts/eval.py \
  --config configs/eval/lm_eval.yaml \
  --run <run_dir> \
  --reports <reasoning_report.json> <long_context_report.json>
```

This writes `reports/lm_eval_report.json` with validation loss, perplexity, requested throughput/routing metrics, optional downstream task accuracy, and an aggregate benchmark score from supplied downstream reports.

When `resume: true` loads `checkpoint_latest`, training appends `resume_events.jsonl`. Stage 0 gates require `checkpoint_latest` and `checkpoint_best` artifacts plus a valid resume event showing the latest checkpoint path, positive resumed step, larger target step, and loaded optimizer state.

Generate fixed-route stability evidence for a Stage 1 run:

```bash
python scripts/eval.py \
  --config configs/eval/fixed_route_stability.yaml \
  --run <stage1_fixed_route_run>
```

This writes `fixed_route_stability_report.json` with fixed-path shape, finite-logit/loss, route-target, and position-norm checks. Stage 1 gates use it as explicit no-instability evidence.

Generate a stage gate report across multiple runs:

```bash
python scripts/make_stage_gate_report.py \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> <stage5_run> <stage6_run>
```

Every stage gate also verifies that each run carries `config_resolved.yaml`, `train_log.jsonl`, `model_stats.json` with a positive integer parameter count, a valid `data_manifest_ref.json` with tokenized-data paths, sequence length, train/validation token counts, manifest hash, expected and realized source mixture evidence, and a passing `lm_eval_report.json` with validation loss, perplexity, and throughput. Routed stages also require `checkpoint_best/state.pt` and active block evals/token in the validation report. Stage 3 requires a positive difficulty-step correlation.

Include Stage 4 cost-control and OUT-by-difficulty evidence in the stage gate:

```bash
python scripts/eval.py \
  --config configs/eval/stage_gate_eval.yaml \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> \
  --cost-control-report <cost_control_report.json> \
  --out-by-difficulty-report <out_by_difficulty_report.json>
```

Generate the required difficulty-step diagnostic for a routed run:

```bash
python scripts/eval.py \
  --config configs/eval/difficulty_step_eval.yaml \
  --baseline-run <stage0_baseline_run> \
  --routed-run <routed_run>
```

This writes `difficulty_step_report.json` and per-sample JSONL rows into the routed run directory. The key metric is `difficulty_step_correlation = corr(baseline_cross_entropy, route_steps)`.

Check repeated eval determinism for a checkpoint:

```bash
python scripts/eval.py \
  --config configs/eval/eval_determinism.yaml \
  --run <run_dir>
```

This writes `eval_determinism_report.json` by running the same checkpoint twice with the same seed and comparing numeric validation/routing metrics. Stage 0 gates use this as explicit determinism evidence.

Generate baseline validation CE difficulty bins:

```bash
python scripts/eval.py \
  --config configs/eval/baseline_difficulty.yaml \
  --run <stage0_baseline_run>
```

This writes `baseline_difficulty_report.json` plus sample JSONL rows with baseline cross-entropy and easy/medium/hard bins. Stage 0 gates use this as explicit sample-level CE evidence.

Generate Stage 2 pseudo-route curriculum evidence from those difficulty bins:

```bash
python scripts/eval.py \
  --config configs/eval/pseudo_route_curriculum.yaml \
  --run <stage2_router_imitation_run> \
  --baseline-report <stage0_baseline_run>/baseline_difficulty_report.json
```

This writes `pseudo_route_curriculum_report.json` showing that easy samples receive skip/early-exit targets, hard samples receive recurrent targets, and all samples include supervised `OUT`. Stage 2 gates use it as explicit curriculum evidence.

Generate Stage 3 scheduled-routing evidence:

```bash
python scripts/eval.py \
  --config configs/eval/scheduled_routing.yaml \
  --run <stage3_scheduled_free_routing_run>
```

This writes `scheduled_routing_report.json` showing that router probability increases, route imitation weight decays, the schedule reaches fully router-controlled routing, and logged train/eval schedule values match the config. Stage 3 gates use it as explicit scheduled-free-routing evidence.

Run the lightweight synthetic reasoning eval:

```bash
python scripts/eval.py \
  --config configs/eval/reasoning_eval.yaml \
  --run <run_dir> \
  --sample-count 24
```

This writes a reasoning report with exact-match accuracy, teacher-forced target token accuracy,
visible-CoT token estimates, per-task/per-difficulty summaries, and routed compute diagnostics.

Summarize whether the OUT action reduces routed compute on easy samples:

```bash
python scripts/eval.py \
  --config configs/eval/out_by_difficulty.yaml \
  --reasoning-report <reasoning_report.json>
```

This reads the reasoning report's sample JSONL and writes an OUT-by-difficulty report with easy/medium/hard route-step, active-compute, and output-probability summaries. Stage 4 gates use this as explicit evidence that hard samples do not use less routed compute than easy samples.

Run the lightweight long-context / Global KV eval:

```bash
python scripts/eval.py \
  --config configs/eval/long_context_eval.yaml \
  --run <stage5_global_kv_run> \
  --sample-count 12
```

This writes a Package C long-context report covering needle retrieval, synthetic multi-hop tracing,
RULER-style retrieval, LongBench-style QA, long arithmetic traces, and program traces, with exact-match
accuracy, teacher-forced target token accuracy, truncation rate, estimated fp16 KV/global-code memory
budgets, and Global KV routing diagnostics including sink/window attention mass.
The report also includes a coverage summary for expected task families and difficulties.

Summarize the Stage 5 Global KV sink + sliding-window retention evidence:

```bash
python scripts/eval.py \
  --config configs/eval/global_kv_retention.yaml \
  --run <stage5_global_kv_run>
```

This writes `global_kv_retention_report.json` in the run directory by default. The report checks that Global KV is enabled, sink/window slots are configured, sink/window attention mass is measured, global read/cache metrics are non-zero, and cache slots stay within the configured retention capacity.

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
  --global-kv-retention-report <stage5_global_kv_run>/global_kv_retention_report.json \
  --long-context-compare-report reports/long_context_compare.json
```

Generate a compute-adjusted comparison report:

```bash
python scripts/eval.py \
  --config configs/eval/compute_report.yaml \
  --baseline-run <stage0_baseline_run> \
  --runs <stage0_baseline_run> <routed_run_1> <routed_run_2>
```

This writes `reports/compute_report.json` with parameter ratios, active layer eval ratios, estimated FLOPs/token, estimated GPU-hours, validation loss deltas, throughput ratios, latency/token, train step time, inference timing, and CUDA memory snapshots when available.

Compare inference timing with and without hard `OUT` exits:

```bash
python scripts/eval.py \
  --config configs/eval/hard_exit_compare.yaml \
  --baseline-run <stage4_no_hard_exit_run> \
  --run <stage4_hard_exit_run>
```

This writes `reports/hard_exit_compare.json` with hard-exit configuration checks, inference time and latency ratios, route-step ratios, and validation-loss deltas.

Run the Stage 3 block-position smoke ablations:

```bash
python scripts/train.py --config configs/train/stage3_no_position_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_router_only_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_circular_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_random_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_no_location_bias_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_no_location_loss_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_direct_add_tiny_debug.yaml
python scripts/train.py --config configs/train/stage3_position_separate_state_tiny_debug.yaml
```

The formal BRIAN-R125 position ablation manifest is `configs/experiments/route_core_position_ablations.yaml`.
It covers P0-P9: no-position, random/open-arc/circular geometry, router-only position,
router+block position, no location bias, no location loss, direct position-hidden addition,
and separate adapter position state.

Generate a position-ablation evidence report:

```bash
python scripts/eval.py \
  --config configs/eval/position_ablation.yaml \
  --run <main_position_run> \
  --runs <no_position_run> <router_only_position_run> <circular_position_run>
```

This writes `reports/position_ablation_report.json` with validation-loss and routing/position metric deltas. Pass it to the Go/No-Go report with `--position-ablation-report`.

Resolve the full Package A BRIAN-R125 route-core manifest:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/route_core_r125_package.yaml \
  --dry-run
```

This manifest covers A0-A9 from the project plan: fixed baseline, fixed route wrapper,
sequential router imitation, skip/recur router imitation, scheduled free routing with
block-position state, no-position ablation, no hard output-action ablation,
no-location-loss ablation, top-1 routing, and top-2 weighted routing.

Audit a formal experiment package manifest before training:

```bash
python scripts/eval.py \
  --config configs/eval/experiment_coverage.yaml \
  --experiment-manifest configs/experiments/route_core_r125_package.yaml
```

This writes `reports/experiment_coverage_report.json`, checking that the manifest covers the required Project Plan package entries, that train configs resolve, and that stages/modes/model flags match the intended ablation roles.
The same report supports the R350 scaling, cost-control, Global KV, and parallel-passing manifests with `profile: auto`.

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

Completed experiment packages write `experiment_results.json`, `compute_report.json`, and `experiment_package_report.json` under the output directory. The package report checks that every selected manifest entry produced a run, routing report, compute row, and baseline comparison evidence for non-baseline entries.

Generate a cost-control report after a Stage 4 sweep:

```bash
python scripts/eval.py \
  --config configs/eval/cost_control_report.yaml \
  --runs <cost0_run> <cost001_run> <cost01_run> <cost05_run>
```

The fast smoke manifest is `configs/experiments/tiny_cost_control.yaml`; the BRIAN-R125 sweep manifest is `configs/experiments/route_core_cost_control.yaml`.
Both manifests are covered by `configs/eval/experiment_coverage.yaml`, which verifies C0-C3 cost-loss weights before training.

Run the Stage 5 Global KV ablation packages:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_global_kv.yaml \
  --dry-run
```

Summarize a completed Stage 5 Global KV ablation package:

```bash
python scripts/eval.py \
  --config configs/eval/global_kv_ablation.yaml \
  --experiment-manifest configs/experiments/tiny_global_kv.yaml \
  --runs \
    <local_kv_run> \
    <global_uncompressed_run> \
    <global_compressed_run> \
    <global_no_sink_run> \
    <global_with_sink_run> \
    <global_window_small_run> \
    <global_window_large_run> \
    <global_per_block_adapter_run> \
    <global_head_delta_adapter_run> \
  --reports \
    <long_context_local.json> \
    <long_context_uncompressed.json> \
    <long_context_compressed.json> \
    <long_context_no_sink.json> \
    <long_context_with_sink.json> \
    <long_context_window_small.json> \
    <long_context_window_large.json> \
    <long_context_per_block_adapter.json> \
    <long_context_head_delta_adapter.json>
```

This writes `reports/global_kv_ablation_report.json` with C0-C7 coverage checks, sink/no-sink retention deltas,
window-size sweep rows, per-block and per-head low-rank adapter comparisons, and optional long-context memory/quality metrics.
If the long-context reports do not cover every run, the report stays at `warn` instead of `pass`.

The fast smoke manifest is `configs/experiments/tiny_global_kv.yaml`;
the BRIAN-R125 sweep manifest is `configs/experiments/route_core_global_kv.yaml`.
These cover the local-KV baseline, uncompressed and compressed Global KV, no-sink Global KV,
default sink+window Global KV, a small cache-window sweep, per-block Global KV adapters,
and per-head low-rank adapter deltas.

Compare top-k weighted fusion against parallel passing:

```bash
python scripts/eval.py \
  --config configs/eval/parallel_passing.yaml \
  --run <stage6_parallel_run>

python scripts/eval.py \
  --config configs/eval/parallel_compare.yaml \
  --baseline-run <stage5_topk_global_kv_run> \
  --runs <stage6_parallel_run> \
  --output reports/parallel_compare.json
```

The safety report checks beam size, branch cost, branch count pruning, score-margin diagnostics, and per-branch delta-cache slots against the configured Global KV window. The comparison report checks that parallel branches are active, branch score margins are logged, validation loss is not worse than the baseline beyond `max_validation_loss_delta`, and active compute / estimated FLOPs stay under the configured ratios. Pass both into the Stage 6 gate:

```bash
python scripts/eval.py \
  --config configs/eval/stage_gate_eval.yaml \
  --runs <stage0_run> <stage1_run> <stage2_run> <stage3_run> <stage4_run> <stage5_run> <stage6_run> \
  --parallel-passing-report <stage6_parallel_run>/parallel_passing_report.json \
  --parallel-compare-report reports/parallel_compare.json
```

Generate a conservative Go/No-Go decision report:

```bash
python scripts/eval.py \
  --config configs/eval/go_no_go_report.yaml \
  --stage-gate-report reports/stage_gate_report.json \
  --compute-report reports/compute_report.json \
  --out-by-difficulty-report reports/out_by_difficulty_report.json \
  --global-kv-ablation-report reports/global_kv_ablation_report.json \
  --long-context-compare-report reports/long_context_compare.json \
  --parallel-compare-report reports/parallel_compare.json
```

This maps the project plan's R125/R350, R350/1B, and R1B success Go/No-Go criteria to existing evidence. Missing evidence stays explicit as `missing`, failed criteria produce a `stop` recommendation, and supplied parallel-compare evidence is retained as optional evidence without changing the decision.

To check the R1B success criteria directly, override the phase:

```bash
python scripts/eval.py \
  --config configs/eval/go_no_go_report.yaml \
  --phase r1b_success \
  --stage-gate-report reports/stage_gate_report.json \
  --compute-report reports/compute_report.json \
  --global-kv-ablation-report reports/global_kv_ablation_report.json \
  --long-context-compare-report reports/long_context_compare.json
```

The `r1b_success` phase requires non-collapsed routing, compute-adjusted evaluation evidence, controlled Global KV memory, acceptable inference latency, and at least one stable core advantage: compute-adjusted validation loss, reasoning accuracy, long-context memory efficiency, or lower visible-CoT token use at similar reasoning accuracy.

Audit the project plan's Risk Register against collected evidence:

```bash
python scripts/eval.py \
  --config configs/eval/risk_audit.yaml \
  --stage-gate-report reports/stage_gate_report.json \
  --routing-report <run_dir>/routing_report.json \
  --position-ablation-report reports/position_ablation_report.json \
  --global-kv-retention-report <stage5_global_kv_run>/global_kv_retention_report.json \
  --global-kv-ablation-report reports/global_kv_ablation_report.json \
  --long-context-compare-report reports/long_context_compare.json \
  --parallel-passing-report <stage6_parallel_run>/parallel_passing_report.json \
  --parallel-compare-report reports/parallel_compare.json
```

This writes `reports/risk_audit_report.json`, mapping Section 20 risk symptoms to explicit `pass`, `warn`, or `fail` evidence. Missing reports remain `warn`; triggered symptoms retain the plan mitigation text next to the failing risk.

Run the Stage 6 parallel-passing packages:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/tiny_parallel_passing.yaml \
  --dry-run
```

The fast smoke manifest is `configs/experiments/tiny_parallel_passing.yaml`;
the BRIAN-R125 sweep manifest is `configs/experiments/route_core_parallel_passing.yaml`.
These cover PP0 top-k weighted fusion, PP1 beam-2 independent branch passing, PP2 beam-4 capacity,
PP3/PP4 branch-cost off/on ablations, PP5/PP6 top-1-vs-top-k OUT terminal rules,
and PP7/D5 shared base Global KV plus branch delta memory checks.

Resolve the first BRIAN-R350 scaling package:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/route_core_r350_scaling.yaml \
  --dry-run
```

This manifest covers the Package B scaffold from the project plan: B0 fixed 350M baseline, B1 routed main, B2 no-position ablation, B3 no hard output-action ablation, and B4 current `mixed_skip_recur` difficulty-pattern pseudo-route curriculum. It uses `configs/data/r350_main_10b.yaml` and B200-compatible bf16 training configs.

Resolve the gated BRIAN-R1B pilot package after R350 evidence justifies scale-up:

```bash
python scripts/eval.py \
  --config configs/eval/r1b_pilot_coverage.yaml \
  --output reports/route_core_r1b_pilot_coverage.json
```

This manifest defines D0 fixed 1B baseline and D1 routed 1B Global KV pilot on `configs/data/r1b_pilot.yaml`. It uses BF16 with `activation_checkpointing: true`, gradient accumulation, DDP no-sync accumulation, and warmup/cosine LR decay for B200/H100 memory control. D1 enables `ddp_find_unused_parameters` for dynamic routing and keeps `parallel_passing: false`; parallel remains an experimental follow-up rather than part of the R1B pilot default.

Resolve the gated scale follow-up packages after the relevant go/no-go evidence passes:

```bash
python scripts/eval.py --config configs/eval/r125_5b_followup_coverage.yaml
python scripts/eval.py --config configs/eval/r350_30b_followup_coverage.yaml
python scripts/eval.py --config configs/eval/r1b_main_validation_coverage.yaml
```

Dry-run their train manifests before submitting expensive jobs:

```bash
python scripts/run_experiment.py --config configs/experiments/route_core_r125_5b_followup.yaml --dry-run
python scripts/run_experiment.py --config configs/experiments/route_core_r350_30b_followup.yaml --dry-run
python scripts/run_experiment.py --config configs/experiments/route_core_r1b_main_validation.yaml --dry-run
```

These packages bind the planned `r125_main_5b`, `r350_main_30b`, and `r1b_main_50b` recipes to explicit baseline/routed configs. The 1B main-validation configs keep BF16 plus `activation_checkpointing: true`, gradient accumulation, DDP no-sync accumulation, and warmup/cosine LR decay for B200 memory control; the routed 1B config enables `ddp_find_unused_parameters` and leaves `parallel_passing: false`.

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
- route path examples;
- route path diversity;
- recurrent and skip ratios;
- location distance;
- position/location trajectories;
- active block evaluations per token;
- cost-quality curve;
- latency/token, train step time, inference timing, and CUDA memory snapshots;
- difficulty-step correlation;
- `OUT` probability by difficulty.
Global KV reports additionally track global read gate, local/global read ratios, global cache slots, sink/window attention mass, and cache window/capacity utilization.

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
