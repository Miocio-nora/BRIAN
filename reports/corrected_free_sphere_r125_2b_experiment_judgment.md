# Corrected Free-Sphere R125 2B Experiment Judgment

Date: 2026-06-13

Scope: corrected R125 2B free-sphere route-core experiments after rejecting the
original hand-written mixed skip/recur target path objective.

## Decision

The corrected route-core direction is now the main R125 evidence path. The
original Package A skip/recur target-path results remain useful as engineering
checks, but they should not drive scale-up decisions.

Current best completed run: `free_sphere_r125_2b_A0_coverage_no23`.

This run gives the best language-model quality so far, but still has route-path
concentration. The current follow-up package is therefore testing whether the
A0 behavior can be made less concentrated without losing its LM advantage.

## Completed 2x2 Results

| ID | Setup | Validation loss | Perplexity | Path diversity | Block entropy norm | Judgment |
|---|---|---:|---:|---:|---:|---|
| A0 | coverage warmup, no selected-balance / transition-diversity | 3.0284 | 20.6650 | 0.1914 | 0.8859 | best LM quality; keep as anchor |
| A1 | coverage warmup + selected-balance + transition-diversity | 3.0459 | 21.0294 | 0.8516 | 0.9373 | diversity improves, LM worsens slightly |
| B0 | pure free, no selected-balance / transition-diversity | 3.1557 | 23.4687 | 0.2227 | 0.3481 | weak quality and poor coverage |
| B1 | pure free + selected-balance + transition-diversity | 3.1143 | 22.5168 | 0.3594 | 0.6886 | better than B0, still behind coverage warmup |

## Interpretation

Coverage warmup matters. A0 and A1 both beat the pure-free variants, so the
router still benefits from a stable early routing curriculum.

The 2/3 losses are not universally beneficial. Strong selected-balance and
transition-diversity greatly improve diversity and block balance in A1, but the
LM loss is worse than A0. They are useful pressures, but the current weights are
too expensive for the best quality run.

Pure free routing is not ready as the default R125 path. B0 collapses too much,
and B1 only partially repairs it.

Top-2 weighted fusion currently matters. The top1-only follow-up completed with
validation loss 3.2347, block-load entropy 0.0, path diversity 0.03125, and a
single repeated-block-7 route family. Top1-only is therefore not a viable
current route-core direction.

## Active Follow-Up Package

These runs are defined by
`configs/experiments/route_core_r125_2b_corrected_package_a_followup.yaml`.

| ID | Setup | Status | Current/Final observation |
|---|---|---|---|
| A0opt | A0 + weak selected-balance | running | in-progress; latest eval step 15000, validation loss 3.2723 |
| Apos | A0 without block-position input/loss | running | in-progress; latest eval step 17500, validation loss 3.2516 |
| Aout | A0 without hard OUT termination | running | in-progress; latest eval step 17500, validation loss 3.2093 |
| Atop1 | A0 with top1-only routing | complete | final validation loss 3.2347; route collapse to repeated block 7 |

The in-progress values above are snapshots only. Do not compare them as final
outcomes until A0opt, Apos, and Aout reach `step=30518`.

## Practical Next Step

Finish the three running follow-ups, then update this report with final losses,
path-diversity metrics, block histograms, and route-path visualizations. If
A0opt reduces concentration without hurting LM loss too much, it becomes the
next R125 5B candidate. If Aout keeps improving, hard OUT should remain under
question rather than being treated as settled.

## Evidence

Train configs:

- `configs/train/free_sphere_r125_2b_a0_coverage_no23.yaml`
- `configs/train/free_sphere_r125_2b_a1_coverage_23loss.yaml`
- `configs/train/free_sphere_r125_2b_b0_pure_no23.yaml`
- `configs/train/free_sphere_r125_2b_b1_pure_23loss.yaml`
- `configs/train/corrected_package_a_r125_2b_a0opt_weak_selected_balance.yaml`
- `configs/train/corrected_package_a_r125_2b_apos_no_position.yaml`
- `configs/train/corrected_package_a_r125_2b_aout_no_hard_exit.yaml`
- `configs/train/corrected_package_a_r125_2b_atop1_top1_only.yaml`

Experiment manifests:

- `configs/experiments/route_core_r125_2b_free_sphere_2x2.yaml`
- `configs/experiments/route_core_r125_2b_corrected_package_a_followup.yaml`

Run directories:

- `runs/free_sphere_r125_2b_A0_coverage_no23`
- `runs/free_sphere_r125_2b_A1_coverage_23loss`
- `runs/free_sphere_r125_2b_B0_pure_no23`
- `runs/free_sphere_r125_2b_B1_pure_23loss`
- `runs/corrected_package_a_r125_2b_A0opt_weak_selected_balance`
- `runs/corrected_package_a_r125_2b_Apos_no_position`
- `runs/corrected_package_a_r125_2b_Aout_no_hard_exit`
- `runs/corrected_package_a_r125_2b_Atop1_top1_only`
