# Global KV Rerun Summary for External Discussion

Date: 2026-06-22

This note summarizes the current BRIAN route-core experiments for discussion
with GPT Pro. It focuses on the R125 Package A route-core line, the corrected
5B token-balanced reruns, and the gap between low training/validation loss and
poor public/reasoning benchmarks.

## One-Sentence Status

The route-core framework itself is not the immediate bottleneck, but both
hidden-state Global KV and attention-level Global KV show strong evidence of
late shortcut/overfit behavior: early checkpoints score well on synthetic
reasoning, then benchmark quality collapses while PPL keeps improving or while
cache usage becomes dominant.

## Evaluation Suite

- Synthetic reasoning s600: `configs/eval/reasoning_eval_s600.yaml`
  - 600 synthetic samples.
  - Reports exact-match accuracy and teacher-forced token accuracy.
- Public multiple-choice guardrail: `configs/eval/public_benchmark_s600.yaml`
  - 200 validation samples each from PIQA, HellaSwag, and ARC-Easy.
  - 600 total samples.
  - Length-normalized answer-choice likelihood.
- Training/eval logs:
  - PPL from `eval_log.jsonl` where available.
  - Route metrics: path count/diversity, block-load entropy, route entropy.
  - Global memory metrics: hidden Global KV read gate / global-local read ratio;
    attention Global KV last-token attention mass.

## Data Correction

The original `r125_main_5b` data was effectively FineWeb-Edu dominated:
FineWeb-Edu was about 96.7% of realized tokens despite the intended 70/10/10/5/5
mixture. This made the old 5B Global KV results hard to interpret.

The corrected dataset is:

```text
configs/data/r125_main_5b_balanced.yaml
```

It uses token-balanced mixture scheduling. Realized token shares:

| Source | Token share |
|---|---:|
| FineWeb-Edu | 0.6999 |
| TinyStories | 0.1000 |
| synthetic routing | 0.1001 |
| symbolic math QA | 0.0500 |
| structured code | 0.0500 |

## Main Benchmark Table

Empty val-loss/PPL cells mean that the corresponding historical run did not
save an LM-eval report in the same format.

| Run | Val loss | PPL | Reason exact | Teacher acc | Public avg | PIQA | Hella | ARC-E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B baseline | 3.186 | 24.194 | 0.545 | 0.856 | 0.383 | 0.525 | 0.260 | 0.365 |
| 5B baseline |  |  | 0.648 | 0.901 | 0.410 | 0.590 | 0.280 | 0.360 |
| 2B fixed route | 3.095 | 22.097 | 0.525 | 0.866 | 0.383 | 0.535 | 0.265 | 0.350 |
| 2B A0 | 3.028 | 20.665 | 0.122 | 0.758 | 0.385 | 0.545 | 0.280 | 0.330 |
| 2B selected | 3.029 | 20.678 | 0.357 | 0.801 | 0.353 | 0.480 | 0.280 | 0.300 |
| 2B noise | 3.028 | 20.659 | 0.162 | 0.740 | 0.385 | 0.565 | 0.260 | 0.330 |
| 2B A0+selected+noise |  |  | 0.023 | 0.584 | 0.387 | 0.575 | 0.270 | 0.315 |
| 2B Ain+anchor | 3.016 | 20.401 | 0.218 | 0.746 | 0.380 | 0.510 | 0.285 | 0.345 |
| 2B Ain+selected | 3.016 | 20.415 | 0.010 | 0.557 | 0.388 | 0.570 | 0.270 | 0.325 |
| 2B Ain+anchor+selected+noise |  |  | 0.353 | 0.817 | 0.398 | 0.600 | 0.235 | 0.360 |
| 5B A0+selected |  |  | 0.160 | 0.775 | 0.393 | 0.520 | 0.305 | 0.355 |
| 5B A0 random/no-location |  |  | 0.185 | 0.789 | 0.405 | 0.575 | 0.250 | 0.390 |
| 5B A0 covfloor stronglen |  |  | 0.425 | 0.842 | 0.412 | 0.555 | 0.305 | 0.375 |
| 5B A0 covfloor lenloose |  |  | 0.040 | 0.635 | 0.368 | 0.545 | 0.255 | 0.305 |
| 5B Ain+anchor+selected+noise |  |  | 0.047 | 0.637 | 0.398 | 0.545 | 0.290 | 0.360 |
| old hidden Global KV |  |  | 0.002 | 0.386 | 0.317 | 0.450 | 0.225 | 0.275 |
| old hidden Global KV slow-noise |  |  | 0.000 | 0.117 | 0.327 | 0.430 | 0.295 | 0.255 |
| old attention Global KV |  |  | 0.003 | 0.287 | 0.330 | 0.485 | 0.230 | 0.275 |
| old attention Global KV slow-noise |  |  | 0.000 | 0.312 | 0.317 | 0.520 | 0.200 | 0.230 |
| old hidden Global KV no-position |  |  | 0.007 | 0.561 | 0.368 | 0.515 | 0.270 | 0.320 |

## Balanced Global KV Reruns

These two reruns use the corrected token-balanced 5B dataset, slow router noise,
self-recur cap, selected-balance/coverage constraints, checkpoint retention
every 5000 steps, and benchmark probes every 15000 steps.

Configs:

```text
configs/train/corrected_global_kv_r125_5b_balanced_slow_noise.yaml
configs/train/corrected_attention_global_kv_r125_5b_balanced_slow_noise.yaml
```

| Run | Step | PPL | Reason exact | Teacher acc | Public avg | PIQA | Hella | ARC-E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced hidden Global KV | 15000 | 11.332 | 0.735 | 0.918 | 0.372 | 0.540 | 0.270 | 0.305 |
| balanced hidden Global KV | 30000 | 9.372 | 0.622 | 0.914 | 0.362 | 0.500 | 0.265 | 0.320 |
| balanced hidden Global KV | 45000 | 6.523 | 0.000 | 0.275 | 0.358 | 0.500 | 0.250 | 0.325 |
| balanced hidden Global KV | 60000 | 4.613 | 0.000 | 0.151 | 0.342 | 0.515 | 0.245 | 0.265 |
| balanced hidden Global KV | 75000 | 44.271 | 0.008 | 0.409 | 0.322 | 0.455 | 0.230 | 0.280 |
| balanced hidden Global KV | 76294 | 34.390 | 0.008 | 0.434 | 0.327 | 0.440 | 0.230 | 0.310 |
| balanced attention Global KV | 15000 | 11.249 | 0.715 | 0.918 | 0.372 | 0.535 | 0.265 | 0.315 |
| balanced attention Global KV | 30000 | 9.030 | 0.553 | 0.885 | 0.365 | 0.530 | 0.270 | 0.295 |
| balanced attention Global KV | 45000 | 7.303 | 0.347 | 0.874 | 0.373 | 0.535 | 0.250 | 0.335 |
| balanced attention Global KV | 60000 | 4.450 | 0.023 | 0.376 | 0.340 | 0.515 | 0.200 | 0.305 |
| balanced attention Global KV | 75000 | 3.496 | 0.003 | 0.335 | 0.323 | 0.470 | 0.210 | 0.290 |
| balanced attention Global KV | 76294 | 3.236 | 0.005 | 0.240 | 0.350 | 0.495 | 0.240 | 0.315 |

## Routing and Cache Diagnostics

The collapse is not well explained by a simple global routing collapse at the
moment reasoning first fails. Hidden Global KV at 45k has higher train path
count than at 15k/30k, but reasoning exact is already zero. This points more
toward global-memory shortcut behavior and benchmark/task-specific routing
degeneration than toward a single-path router collapse.

| Run | Step | PPL | Reason exact | Train path count | Train block ent | Bench block ent | Global read/gate | Global mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hidden Global KV | 15000 | 11.332 | 0.735 | 6.125 | 0.853 | 0.664 | 0.300 | 0.429 |
| hidden Global KV | 30000 | 9.372 | 0.622 | 7.000 | 0.905 | 0.663 | 0.452 | 0.826 |
| hidden Global KV | 45000 | 6.523 | 0.000 | 18.875 | 0.792 | 0.364 | 0.825 | 4.701 |
| hidden Global KV | 60000 | 4.613 | 0.000 | 11.000 | 0.680 | 0.466 | 0.810 | 4.255 |
| hidden Global KV | 75000 | 44.271 | 0.008 | 1.000 | 0.306 | 0.306 | 0.770 | 3.346 |
| hidden Global KV | 76294 | 34.390 | 0.008 | 1.000 | 0.306 | 0.306 | 0.762 | 3.206 |
| attention Global KV | 15000 | 11.249 | 0.715 | 10.250 | 0.909 | 0.787 | 0.180 | 0.180 |
| attention Global KV | 30000 | 9.030 | 0.553 | 6.125 | 0.907 | 0.808 | 0.460 | 0.460 |
| attention Global KV | 45000 | 7.303 | 0.347 | 8.500 | 0.913 | 0.841 | 0.577 | 0.577 |
| attention Global KV | 60000 | 4.450 | 0.023 | 5.875 | 0.791 | 0.752 | 0.482 | 0.482 |
| attention Global KV | 75000 | 3.496 | 0.003 | 5.125 | 0.806 | 0.771 | 0.499 | 0.499 |
| attention Global KV | 76294 | 3.236 | 0.005 | 5.000 | 0.790 | 0.754 | 0.478 | 0.478 |

Additional route-path visualization checks for hidden Global KV:

| Step | Unique paths | Top path share | Node entropy norm | Dead nodes |
|---:|---:|---:|---:|---|
| 15000 | 336715 | 0.009 | 0.934 | none |
| 30000 | 505296 | 0.025 | 0.948 | none |
| 45000 | 659364 | 0.017 | 0.965 | none |

So the 45k hidden Global KV failure is not a classic all-paths-collapse event.
At 75k/76294, hidden Global KV does show a later routing collapse and PPL
instability, but that happens after reasoning has already failed.

## Current Interpretation

1. Loss/PPL alone is actively misleading for this architecture. Some candidates
   with modestly worse loss are better on benchmarks; Global KV can reach good
   PPL while benchmark behavior collapses.

2. Balanced data fixed the earlier dataset-mixture bug and made early Global KV
   checkpoints look strong. The 15k checkpoints for both hidden and attention
   Global KV beat the 5B baseline on synthetic reasoning exact/teacher accuracy.

3. Hidden Global KV appears especially prone to a memory shortcut. By 45k,
   hidden Global KV has high global-read gate, high global-to-local read ratio,
   zero reasoning exact, and sharply lower benchmark block entropy.

4. Attention Global KV is less immediately destructive than hidden Global KV.
   It retains nonzero reasoning through 45k and keeps lower/more bounded global
   attention mass, but it still collapses by 60k.

5. Public MCQ accuracy is a weak guardrail at this model/data scale. It moves
   slowly and remains low, while synthetic reasoning exposes much sharper phase
   changes. The public suite is still useful, but it is not sufficient alone.

6. No-router-position is not currently a priority. The earlier no-position
   evidence was not clearly positive, so the balanced no-position branch was
   not continued.

## Questions for GPT Pro

1. Is the Global KV failure best interpreted as overfitting, objective shortcut,
   exposure-bias/generation collapse, or a memory-interface instability?

2. How should Global KV be regularized without killing its potential?
   Candidate knobs:
   - cap or schedule global-read gate;
   - add global-read dropout;
   - add local-read floor;
   - add entropy regularization over global-vs-local reads;
   - delay Global KV activation until route-core behavior stabilizes;
   - freeze or slow the global cache adapters early.

3. For attention-level Global KV, should cross-block KV be:
   - read-only during early training;
   - detached from gradients for some schedule;
   - limited by per-layer or per-head budget;
   - separated into sink/window pools with explicit loss terms?

4. What is the right early-stopping or model-selection criterion?
   The best checkpoint by reasoning is around 15k, not the final checkpoint.

5. Is synthetic reasoning s600 over-weighted as a decision signal, or is it the
   right diagnostic for route-core stability because public MCQ is too noisy at
   this scale?

## Proposed Next Experiments

1. Select checkpoints by benchmark, not by final PPL.
   - Treat 15k Global KV checkpoints as meaningful positive evidence.
   - Do not use final 76294 checkpoints as default.

2. Add Global KV regularization.
   - Hidden Global KV: cap/schedule `global_read_gate_mean`, add local-read
     floor, and consider read dropout.
   - Attention Global KV: cap attention mass to global tokens or schedule it
     upward slowly.

3. Run short ablations only.
   - Stop at 15k/30k/45k until the collapse mechanism is controlled.
   - Avoid full 76k runs unless intermediate benchmarks remain healthy.

4. Keep benchmark probes during training.
   - The benchmark hook caught a real phase transition that final-only eval
     would have obscured.

5. Do not spend more compute on balanced no-position until a stronger reason
   appears.
