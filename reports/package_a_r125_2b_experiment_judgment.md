# Package A R125 2B Experiment Judgment

Date: 2026-06-13

Scope: completed `r125_main_2b` Package A runs A0-A8. This judgment uses the
experimental design and observed results, not merely the presence of artifacts.

## Decision

Do not scale this result directly to R350, Global KV, or parallel passing yet.

The core local route mechanism is implemented, trainable, measurable, and able
to trade quality for lower active compute. The full planned R125 route-core
claim is not proven yet, mainly because OUT/hard-exit and cost-controlled
difficulty-conditioned compute are not established by the current Package A.

Recommended next step:

1. Treat A6, the scheduled router without output action, as the current best
   routed candidate.
2. Treat hard OUT as not ready for scale-up: A8 fixed the A7 location-loss
   confound but still trails A6 by 0.0235 validation loss.
3. If scaling evidence is needed, scale the A6-style local route-core path
   first; keep OUT/hard-exit as a targeted follow-up rather than the main path.
4. Do not spend on R350 until the selected R125 path is confirmed at 5B and the
   remaining cost/difficulty evidence is addressed.

## Main Results

| ID | Experimental question | Validation loss | Active compute vs A0 | Judgment |
|---|---|---:|---:|---|
| A0 | fixed Transformer baseline | 3.1861 | 1.000 | reference |
| A1 | fixed route wrapper | 3.0954 | 1.000 | wrapper is not harmful |
| A2 | sequential router imitation | 3.0957 | 1.000 | simple router imitation works |
| A3 | skip/recur router imitation | 3.3930 | 0.555 | pseudo policy is learnable but quality-damaging |
| A4 | scheduled free router with block position | 3.2631 | 0.642 | route-core works, but with quality cost |
| A5 | no block position | 3.2828 | 0.628 | position helps modestly |
| A6 | no output action | 3.2513 | 0.642 | best current routed candidate |
| A7 | no location loss, hard exit enabled | 3.3055 | 0.630 | OUT/hard-exit not proven |
| A8 | output action + location loss | 3.2748 | 0.630 | OUT improved vs A7, still worse than A6 |

## What The Experiment Shows

Fixed routing is safe. A1 improves over A0 at the same active compute, so the
route-wrapper implementation itself is not the limiting factor.

Router imitation is trainable. A2 matches A1, and A3 reaches perfect reported
route-imitation accuracy. The A3 loss degradation is therefore more likely a
route-policy/objective problem than a basic training or implementation failure.

Scheduled free routing is real but not yet superior. A4 passes the scheduled
routing report, keeps non-degenerate routing metrics, and uses about 64 percent
of baseline active layer compute. Its loss is 0.077 above A0, so the current
evidence is compute-saving behavior with a quality penalty, not a quality win.

Block-position state is useful but not decisive. A5 is 0.0197 worse than A4 and
the position ablation report passes. This supports keeping block-position state
in the route-core path, but the effect is modest.

OUT/hard-exit is not established. A7 removes location loss and is 0.0542 worse
than A6. A8 restores location loss and improves over A7, but remains 0.0235
worse than A6. The A6-to-A8 hard-exit comparison has valid roles and slightly
better latency/active compute for A8, but fails the validation-loss check.

## What Is Not Proven

The current Package A does not prove cost-control behavior. There is no valid
same-stage Stage 4 output-action cost sweep with multiple cost weights.

The current Package A does not prove difficulty-conditioned compute. The strict
stage gate reports missing difficulty-step and out-by-difficulty evidence.

The current Package A does not prove that the OUT action improves the useful
route-core model. The best routed run is A6, which disables output action.

The current Package A does not justify adding Global KV or parallel passing yet.
Those are downstream features and would confound the unresolved local route-core
questions.

## Generated Evidence

Generated local reports:

- `experiments/generated/route_core_r125_2b_package/compute_report.json`
- `experiments/generated/route_core_r125_2b_package/experiment_package_report.json`
- `experiments/generated/route_core_r125_2b_package/position_ablation_report.json`
- `experiments/generated/route_core_r125_2b_package/hard_exit_compare.json`
- `experiments/generated/route_core_r125_2b_package/stage_gate_report.json`
- `experiments/generated/route_core_r125_2b_package/go_no_go_r125_to_r350.json`
- `experiments/generated/route_core_r125_2b_decision_followup/hard_exit_compare.json`
- `experiments/generated/route_core_r125_2b_decision_followup/compute_report.json`

The formal go/no-go report returns `stop` for R125 to R350. This is the correct
strict-plan outcome, but the practical interpretation is narrower: the local
route-core mechanism is promising, while OUT/cost/difficulty evidence is still
missing.

## A8 Follow-Up

A8 was run to isolate the missing Stage 4 comparison:

| Comparison | A6 no OUT | A8 OUT + location loss |
|---|---:|---:|
| Validation loss | 3.2513 | 3.2748 |
| Validation delta vs A6 | 0.0000 | +0.0235 |
| Perplexity | 25.82 | 26.44 |
| Active layer compute vs A6 | 1.000 | 0.982 |
| Inference latency vs A6 | 1.000 | 0.994 |
| Location distance mean | 0.0072 | 0.0120 |

The A6-to-A8 hard-exit comparison status is `warn`: A8 passes the role,
hard-exit, timing, and route-step checks, but fails `validation_loss_not_worse`.
A8 also fixes the A7 location-loss confound: A7's location distance was 0.3395,
while A8's is 0.0120. The practical conclusion is that location loss is needed,
but hard OUT is still not the best R125 route-core path.

Recompute the comparison:

```bash
PYTHONPATH=src python scripts/eval.py --config configs/eval/hard_exit_compare.yaml \
  --baseline-run runs/package_a_r125_2b_A6_no_output_action \
  --runs runs/package_a_r125_2b_A8_output_action_location_loss \
  --output experiments/generated/route_core_r125_2b_decision_followup/hard_exit_compare.json
```

Coverage config:

```text
configs/eval/r125_2b_decision_followup_coverage.yaml
```
