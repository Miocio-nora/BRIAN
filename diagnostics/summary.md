# Global KV Checkpoint Diagnostics

Scope: existing checkpoints only; no retraining. Future-work training recipes are intentionally excluded.

## Clean validation set

- Active `data/tokenized/r125_main_5b_balanced/val.bin` is now exact-token clean against train.
- Legacy validation is preserved as `val_legacy.bin` / `val_legacy.idx`.
- Detailed filtering stats are in `data/tokenized/r125_main_5b_balanced/val_clean_stats.json`.

## Answers

1. Causal leakage: YES by suffix invariance. 7/7 checkpoints changed prefix logits when only suffix changed; max prefix-logit diff ranges from 1.53 to 10.44.
2. Full forward vs incremental: NOT CONSISTENT. 42/42 tested positions exceeded 1e-3 max-logit diff; max diff ranges from 1.06 to 23.51.
3. Global KV role: Global is not merely auxiliary. Global-off usually worsens clean-val PPL and reasoning, especially hidden 45k and attention 30k/45k/60k. This means the trained model has become dependent on the Global path, but because P0.1/P0.2 fail, that dependency is contaminated by a non-causal full-sequence/cache interface.
4. Priority: fix causal/cache lifecycle first. Do not interpret the late Global collapse as ordinary overfitting or solve it primarily with regularization until suffix invariance and full-vs-incremental pass.

## Key Evidence

- Hidden 45k: default clean-val PPL is 12.43, but global-off PPL is 766,083.55; batch memory swap reaches 3,897,147.76. This shows hard dependence on specific Global content.
- Attention 30k: default PPL is 19.57, global-off PPL is 48.87, and small reasoning exact drops from 0.625 to 0.0.
- Attention 45k: default PPL is 14.83, global-off PPL is 7,893.70; sink-only is much better than window-only but still worse than default.
- Attention 60k: default PPL is 7.56 while cached s600 reasoning exact is only 0.0233; global mass is 0.4035. This is the clearest loss/benchmark separation.
- Slot shuffle is mostly neutral, but batch memory swap is destructive at later checkpoints. The model depends on the memory content, not just the number of slots.

## Caveats

- P0/P1 intervention reasoning uses only 8 generated samples for speed; use the cached s600/public columns in `checkpoint_selection_matrix.csv` for stronger benchmark judgment.
- `public_avg` in intervention CSVs is intentionally blank because no new public benchmark sweep was run.

## Best cached checkpoints by s600 reasoning

| rank | family | checkpoint | s600 exact | s600 teacher | public s200 | clean val ppl | global mass |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | hidden | checkpoint_step_00015000 | 0.7350 | 0.9178 | 0.3717 | 27.1074 | 0.0000 |
| 2 | attention | checkpoint_step_00015000 | 0.7150 | 0.9185 | 0.3717 | 26.9640 | 0.1321 |
| 3 | hidden | checkpoint_step_00030000 | 0.6217 | 0.9140 | 0.3617 | 19.9785 | 0.0000 |
| 4 | attention | checkpoint_step_00030000 | 0.5533 | 0.8855 | 0.3650 | 19.5748 | 0.3165 |
| 5 | attention | checkpoint_step_00045000 | 0.3467 | 0.8738 | 0.3733 | 14.8279 | 0.3970 |
| 6 | attention | checkpoint_step_00060000 | 0.0233 | 0.3759 | 0.3400 | 7.5556 | 0.4035 |
| 7 | hidden | checkpoint_step_00045000 | 0.0000 | 0.2750 | 0.3583 | 12.4297 | 0.0000 |

## Generated files

- `p0_suffix_invariance.csv`
- `p0_full_vs_incremental.csv`
- `p0_global_sweep.csv`
- `p0_memory_intervention.csv`
- `p1_norm_audit.csv`
- `p1_route_intervention.csv`
- `checkpoint_selection_matrix.csv`
