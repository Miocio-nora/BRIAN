# Corrected Free-Sphere R125 2B Experiment Judgment

Date: 2026-06-14

Scope: corrected R125 2B free-sphere route-core experiments after rejecting the
original hand-written mixed skip/recur target path objective.

## Decision

The corrected route-core direction is now the main R125 evidence path. The
original Package A skip/recur target-path results remain useful as engineering
checks, but they should not drive scale-up decisions.

Current best validation-loss run: `corrected_package_a_r125_2b_AinAnchorA_input_anchor`.

Current best synthetic-reasoning single stabilizer: weak `selected_balance` on
the A0 coverage-warmup route. Current best combined benchmark candidate:
`Ain + input_anchor + selected_balance + router_logit_noise`. It has worse
validation loss, but stronger s600 benchmark evidence, so it must not be
rejected by PPL alone.

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

## Completed Follow-Up Package

These runs are defined by
`configs/experiments/route_core_r125_2b_corrected_package_a_followup.yaml`.

| ID | Setup | Final observation |
|---|---|---|
| A0opt | A0 + weak selected-balance | validation loss 3.0290; improves synthetic reasoning strongly over A0; public MCQ does not confirm a global win |
| Apos | A0 without block-position input/loss | validation loss 3.0636; worse than A0, so keep block-position |
| Aout | A0 without hard OUT termination | validation loss 3.0552; not selected as default |
| Atop1 | A0 with top1-only routing | validation loss 3.2347; route collapse to one repeated path |
| Noise | A0 + train-time router logit noise | validation loss 3.0282; improves A0 diversity and reasoning exact match without hurting loss |
| AinAnchorA | independent IN + input anchor | validation loss 3.0156; strong LM result, but benchmark does not prove it is safely positive |
| AinSelective | independent IN + weak selected-balance | validation loss 3.0163; strong LM result, but synthetic reasoning fails in this benchmark scene |
| A0SelectedNoise | A0 + weak selected-balance + router-logit noise | validation loss 3.0322; public MCQ is acceptable, but reasoning s600 collapses, so this exact combo is not a default |
| AinAnchorSelectedNoise | independent IN + input anchor + weak selected-balance + router-logit noise | validation loss 3.0688; best teacher-forced reasoning, best public average, high route diversity; high-priority scale candidate |

## Practical Next Step

Stop Package A follow-up for this benchmark scene. Move to the next benchmark
scene with `selected_balance`, router-logit `noise`, and top-2 weighted routing
kept as active ingredients. Treat independent IN alone as unresolved, but carry
the full `Ain + anchor + selected + noise` combination as a serious scale
candidate.

The fine-grained branch was implemented as a separate follow-up:
`configs/train/finegrained_r125_2b_pool16_ain_coverage_no23.yaml`. It uses 16
route-pool nodes, smaller route-block FFNs, 32 max route steps, and independent
IN. The initial batch-32 run OOMed; the batch-16/accum-2 retry showed obvious
path collapse and was stopped. Larger route pools should be retried only with
selective-balance-style constraints.

The benchmark pass is tracked in `reports/package_a_benchmarks/summary.md`.
It uses `reasoning_eval`, `out_by_difficulty`, `lm_eval`, and
`difficulty_step_eval` on `checkpoint_latest`.

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
- `configs/train/corrected_package_a_r125_2b_ain_independent_in.yaml`
- `configs/train/corrected_package_a_r125_2b_ain_independent_in_4gpu.yaml`
- `configs/train/corrected_package_a_r125_2b_ain_anchorA_input_anchor.yaml`
- `configs/train/corrected_package_a_r125_2b_ain_selective_balance.yaml`
- `configs/train/corrected_package_a_r125_2b_anoise_logit_noise.yaml`
- `configs/train/corrected_package_a_r125_2b_a0_selected_noise.yaml`
- `configs/train/corrected_package_a_r125_2b_ain_anchor_selective_noise.yaml`
- `configs/train/finegrained_r125_2b_pool16_ain_coverage_no23.yaml`

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
- `runs/corrected_package_a_r125_2b_Anoise_logit_noise`
- `runs/corrected_package_a_r125_2b_AinAnchorA_input_anchor`
- `runs/corrected_package_a_r125_2b_Ain_selective_balance`
- `runs/corrected_package_a_r125_2b_A0_selected_noise`
- `runs/corrected_package_a_r125_2b_Ain_anchor_selective_noise`
