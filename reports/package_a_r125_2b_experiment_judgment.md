# Package A R125 2B Experiment Judgment

Date: 2026-06-13

Scope: completed `r125_main_2b` Package A runs A0-A7. This judgment uses the
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
2. Run a minimal R125 follow-up instead of a full larger package:
   - A0 baseline vs A6 on `r125_main_5b`, if the goal is scaling evidence for
     the useful local route-core path.
   - One missing Stage 4 output-action run with location loss enabled, if the
     goal is specifically to prove OUT/hard-exit.
3. Do not spend on R350 until the selected R125 path is confirmed at 5B or the
   OUT/hard-exit missing cell is resolved.

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

OUT/hard-exit is not established. The hard-exit comparison A6 to A7 passes the
role and timing checks, but A7 is 0.0542 worse in validation loss. Also, A7
removes location loss, so this experiment does not isolate OUT/hard-exit cleanly.

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

The formal go/no-go report returns `stop` for R125 to R350. This is the correct
strict-plan outcome, but the practical interpretation is narrower: the local
route-core mechanism is promising, while OUT/cost/difficulty evidence is still
missing.
