# Package A Benchmark Summary

Date: 2026-06-15

Scope: corrected R125 2B Package A follow-up runs on `r125_main_2b`.
This benchmark pass exists because validation loss/perplexity alone are not
enough to judge route-core behavior.

Important implementation note: the first benchmark attempt exposed an eval bug
in `src/brian_sphere_llm/eval/difficulty_report.py`: the shared routed eval
helper did not pass `routing.constraints` or `routing_options` into the model.
That meant reasoning/difficulty evals did not exactly match training-time eval
routing. The bug was fixed before the current numbers were generated.

## Regular Suite

All candidate checkpoints should use `checkpoint_latest` and the same fixed
suite:

- training eval telemetry: validation loss, perplexity, latency, active block
  evals, route diversity/count, block entropy, route entropy, OUT probability,
  and route steps;
- synthetic reasoning s600: `configs/eval/reasoning_eval_s600.yaml`;
- public multiple-choice s600: `configs/eval/public_benchmark_s600.yaml`;
- optional routing probes: `configs/eval/difficulty_step_latest.yaml` and
  `configs/eval/out_by_difficulty.yaml`.

The training entrypoint now supports `post_train_benchmarks.enabled: true`, so
future configured runs automatically execute the two s600 benchmark reports
after training finishes and `checkpoint_latest` is written.

## S600 Results

| Label | Val loss | PPL | Path div | Path count | Block ent | Reason exact | Teacher acc | Public avg | PIQA | HellaSwag | ARC-Easy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 3.1861 | 24.194 | n/a | n/a | n/a | 0.545 | 0.856 | 0.383 | 0.525 | 0.260 | 0.365 |
| `fixed_route` | 3.0954 | 22.097 | 0.031 | 1.00 | 1.000 | 0.525 | 0.866 | 0.383 | 0.535 | 0.265 | 0.350 |
| `A0` | 3.0284 | 20.665 | 0.191 | 6.12 | 0.886 | 0.122 | 0.758 | 0.385 | 0.545 | 0.280 | 0.330 |
| `noise` | 3.0282 | 20.659 | 0.320 | 10.25 | 0.884 | 0.162 | 0.740 | 0.385 | 0.565 | 0.260 | 0.330 |
| `selected` | 3.0290 | 20.678 | 0.352 | 11.25 | 0.921 | 0.357 | 0.801 | 0.353 | 0.480 | 0.280 | 0.300 |
| `A0_selected_noise` | 3.0322 | 20.743 | 0.230 | 7.38 | 0.888 | 0.023 | 0.584 | 0.387 | 0.575 | 0.270 | 0.315 |
| `ain_anchor` | 3.0156 | 20.401 | 0.125 | 4.00 | 0.870 | 0.218 | 0.746 | 0.380 | 0.510 | 0.285 | 0.345 |
| `ain_selected` | 3.0163 | 20.415 | 0.160 | 5.12 | 0.968 | 0.010 | 0.557 | 0.388 | 0.570 | 0.270 | 0.325 |
| `ain_anchor_selected_noise` | 3.0688 | 21.517 | 0.395 | 12.62 | 0.913 | 0.353 | 0.817 | 0.398 | 0.600 | 0.235 | 0.360 |

Public multiple-choice uses 200 validation samples each from PIQA, HellaSwag,
and ARC-Easy. Synthetic reasoning uses 600 generated samples.

## Judgment

The earlier loss-only read was wrong. `A0_selected_noise` has only a small
validation-loss regression, but it collapses badly on synthetic reasoning
(`0.023` exact, `0.584` teacher accuracy). The A0 version of selected + noise
should not be treated as an additive improvement.

`ain_anchor_selected_noise` is the most interesting new result. It has the
worst validation loss in this set, but it is best on teacher-forced reasoning
accuracy, best on public average, best on PIQA, tied/nearly tied with
`selected` on reasoning exact match, and has the strongest path diversity.
This does not prove the architecture is better, but it does prove that loss/PPL
alone would reject a candidate that benchmark evidence says is high-priority.

`selected` remains a strong synthetic-reasoning stabilizer but is weak on the
public MCQ guardrail. `noise` remains useful as a diversity/regularization
signal, but on A0 it does not combine cleanly with selected at the current
weights.

The independent-IN conclusion changes from "mostly unresolved and likely
unsafe" to "unsafe alone, promising only when anchor + selected balance + noise
are combined." The combination should be tested at the next scale before making
a final architectural decision.

## Current Decision

Do not decide Package A by validation loss alone.

Carry forward as active candidates:

- `selected_balance`;
- train-time router logit `noise`;
- top-2 weighted routing;
- `Ain + anchor + selected_balance + noise` as a high-priority scale candidate.

Treat cautiously:

- `A0 + selected_balance + noise`, because this exact 2B combo failed synthetic
  reasoning despite acceptable public MCQ.

Do not carry forward:

- top1-only routing;
- unregularized multi-block expansion.
