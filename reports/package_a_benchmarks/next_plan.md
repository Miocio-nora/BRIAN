# Next Plan After Package A Benchmarks

Date: 2026-06-15

This plan freezes the next evaluation and training steps without launching any
new training jobs.

## Current Interpretation

The fixed-route BRIAN run is a framework baseline, not the main target. It has
already shown that the route-core wrapper itself is not the primary bottleneck,
so it should remain a reference point but should not consume routine training
compute.

The current free-routing evidence is mixed:

- validation loss favors the Ain-anchor family, but this is not sufficient;
- synthetic reasoning favors selected-balance as a single stabilizer;
- the full `Ain + anchor + selected + noise` combination is strongest on the
  current s600 benchmark suite despite worse validation loss;
- `A0 + selected + noise` fails synthetic reasoning s600 and should not be the
  default just because loss/public MCQ look acceptable;
- noise improves path diversity without obvious loss/public-MCQ damage;
- fixed-route remains much stronger than free routing on synthetic reasoning.

This supports the user's hypothesis: the routed framework has more freedom and
is likely harder to fit at small scale, so current 2B-token evidence should be
treated as diagnostic rather than final.

## Regular Benchmark Suite

Every candidate checkpoint should be evaluated with the same fixed suite:

1. LM/routing eval from training logs:
   - validation loss;
   - perplexity;
   - throughput/latency;
   - active block evals per token;
   - route path diversity/count;
   - block entropy;
   - route entropy;
   - OUT probability and route steps.

2. Synthetic reasoning, 600 samples:
   - config: `configs/eval/reasoning_eval_s600.yaml`;
   - tasks: copy, reverse, arithmetic, rewrite;
   - use both exact-match and teacher-forced token accuracy.

3. Public multiple choice, 600 samples:
   - config: `configs/eval/public_benchmark_s600.yaml`;
   - PIQA: 200 validation samples;
   - HellaSwag: 200 validation samples;
   - ARC-Easy: 200 validation samples;
   - length-normalized answer-choice log-likelihood.

4. Routing behavior probes:
   - `configs/eval/difficulty_step_latest.yaml`;
   - `configs/eval/out_by_difficulty.yaml`.

The fixed-route run should stay in summary tables as a reference but does not
need to be rerun unless the benchmark implementation changes.

`post_train_benchmarks.enabled: true` is now wired into `scripts/train.py` for
the prepared combo and S-class configs, so the two s600 benchmark reports run
automatically after `checkpoint_latest` is written.

## Prepared 2B Combo Validation

Completed.

| ID | Config | Purpose |
|---|---|---|
| C0 | `configs/train/corrected_package_a_r125_2b_a0_selected_noise.yaml` | Rejected as default: reasoning s600 collapses despite acceptable public MCQ. |
| C1 | `configs/train/corrected_package_a_r125_2b_ain_anchor_selective_noise.yaml` | High-priority scale candidate: best teacher-forced reasoning, best public average, high route diversity. |

2B decision:

- do not carry C0 as the main A0 default;
- carry C1 to the next scale as an explicitly benchmark-driven candidate;
- keep C0 config available only as a stress test for whether scale repairs the
  A0 selected/noise interaction.

## Prepared Larger-Scale Validation

Run only with same-token baseline included.

| ID | Config | Purpose |
|---|---|---|
| Sbase | `configs/train/corrected_package_a_r125_5b_baseline.yaml` | Token-budget-matched plain Transformer baseline for the 5B scale scene. |
| S0 | `configs/train/corrected_package_a_r125_5b_a0_selected_noise.yaml` | 5B-token stress test of the A0 selected/noise interaction, not current default. |
| S1 | `configs/train/corrected_package_a_r125_5b_ain_anchor_selective_noise.yaml` | 5B-token validation of the strongest 2B benchmark candidate: independent IN with anchor + selected-balance + noise. |

The prepared 5B configs use `r125_main_5b`, `batch_size: 32`, and
`max_steps: 76294`, which corresponds to roughly 5B tokens at sequence length
2048. Sbase is required: the S-class comparison is not interpretable without a
same-token plain baseline. These configs should be reviewed against available
compute before launch.

## What Not To Do Next

- Do not spend more training on fixed-route unless the framework changes.
- Do not run 16-block or larger route pools without selected-balance-style
  constraints.
- Do not decide by validation loss alone.
- Do not treat the expanded public MCQ suite as the sole decision metric; it is
  a guardrail, not the whole target.
