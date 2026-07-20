# Q10 Strict Per-Head d64 DP-C2048 5B Result

**Date:** 2026-07-20

**Branch:** `rc-kv-per-head-cache`

**Status:** training and checkpoint evaluation complete

## 1. Question

Q10 tests cache capacity inside the strict per-head architecture. Q9 maps each
64-dimensional attention-head K and V into separate 32-dimensional cache
codes. Q10 changes both codes to 64 dimensions, removing forced dimensional
compression while preserving head isolation.

This does not test head isolation itself; Q9 already combines isolation with a
2x per-head compression. Q10 asks whether that remaining compression explains
part of the reasoning deficit that survives in Q9.

## 2. Controlled Contract

Q10 inherits the complete Q9 5B contract:

- balanced 5B-token training corpus and legacy validation split;
- CPBC-DP, depth-prefix visibility, C2048, and U1;
- eight free blocks and at most 16 route steps;
- local batch 16 on two GPUs, global batch 32, and BF16;
- identical optimizer, learning rate, seed, routing controls, and 76,294 steps;
- model-only checkpoints at 15k, 30k, 45k, 60k, 75k, and final;
- capability benchmarks outside the active DDP process.

Only `bdre_key_dim` and `bdre_value_dim` change from 32 to 64.

| Layout | Cache scalars/token | Relative cache | Parameters |
| --- | ---: | ---: | ---: |
| Strict per-head d32 | 768 | 1x | 140,308,105 |
| Strict per-head d64 | 1,536 | 2x | 141,094,537 |

Q10 adds 786,432 parameters, or 0.56% relative to Q9. The parameter difference
is small but explicit; this is a capacity ablation, not a parameter-matched
comparison.

## 3. B200 Acceptance

A 20-step DDP2 smoke used the production local batch 16 and sequence length
2048. It processed 1,310,720 tokens and completed evaluation/checkpoint I/O.

| Measurement | Q10 d64 smoke | Q9 d32 reference |
| --- | ---: | ---: |
| Median global throughput | 134,322 token/s | 170,711 token/s |
| Peak allocated/rank | 107,542 MiB | 74,278 MiB |
| Explicit BDRE cache/rank | 13,056 MiB | 6,528 MiB |
| DDP used parameters | 135/135 | 135/135 |
| Missing-gradient parameters | 0 | 0 |

Loss stayed finite for all 20 updates. There was no OOM, NCCL failure, missing
gradient, or numerical failure. Q10 is approximately 21% slower and uses about
45% more peak allocated memory than the completed Q9 run, but retains roughly
74 GiB of allocation headroom on a 183,359 MiB B200.

The smoke projected 10.5-11.5 hours end to end. The formal run completed in
about 10 hours 8 minutes, inside that range.

The 1.6 GiB temporary smoke checkpoint is not a research asset and is removed
after extracting these acceptance metrics.

## 4. Configuration

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d64.yaml
configs/train/smoke_q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2.yaml
configs/train/q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2_legacyval.yaml
scripts/run_q10_per_head_d64_dp_c2048_5b.sh
scripts/run_q10_per_head_d64_5b_benchmark_matrix.sh
```

Formal launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q10_per_head_d64_dp_c2048_5b.sh
```

The six retained boundaries use the historical S600 contracts. Public V2 is
run at both 75k for an exact optimizer-step comparison and final for model
selection. GSM8K/MATH-500 is not repeated because the completed three-way
matrix established that suite as an accuracy-floor guardrail at this scale.

## 5. Training Completion

Q10 started at `2026-07-20 06:03 JST` in tmux session
`brian_q10_per_head_d64_dp_c2048_5b_g01` on physical GPU 0-1.

W&B run: `mio_nora/brian-sphere-llm/n4y1oj27`

The six-point matrix, both Public V2 boundaries, and paired-test conclusions
are backfilled under `benchmark_backfill/*` in the original run summary.

All 76,294 optimizer steps and 5,000,003,584 tokens completed. The first and
last training records span 10 hours 6 minutes 49 seconds; final validation
finished about one minute later.

| Measurement | Result |
| --- | ---: |
| End-to-end runtime | about 10 h 8 min |
| Median logged global throughput | 142,699 token/s |
| Peak allocation per rank | 107,542 MiB |
| Final validation loss | 1.3695 |
| Final validation PPL | 3.9335 |
| Final validation block entropy | 0.9902 |
| Final validation path diversity | 1.0000 |
| Missing DDP gradient parameters | 0 |

There was no OOM, non-finite loss, NCCL failure, missing gradient, or classic
route collapse. Six model-only checkpoints remain at 15k, 30k, 45k, 60k,
75k, and final.

## 6. Checkpoint Matrix

All capability values below use the same Reason S600 and Public S600 examples
as the earlier baseline, shared-cache, and per-head d32 runs.

| Step | PPL | Reason exact | Teacher acc | Public S600 | Arithmetic | Copy | Reverse | Rewrite | Reason block H |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 5.0562 | 56.33% | 89.67% | 37.67% | 25.33% | 84.00% | 42.00% | 74.00% | 0.8013 |
| 30,000 | 4.4705 | 75.83% | 93.06% | 37.17% | 36.67% | 97.33% | 85.33% | 84.00% | 0.8325 |
| 45,000 | 4.1585 | 82.17% | 94.53% | **39.83%** | 47.33% | 98.00% | 95.33% | 88.00% | 0.8876 |
| 60,000 | 4.0070 | 77.67% | 93.75% | 39.50% | 44.67% | 98.00% | 90.67% | 77.33% | **0.8968** |
| 75,000 | 3.9558 | 82.50% | 94.35% | 39.33% | 43.33% | 100.00% | **100.00%** | 86.67% | 0.8192 |
| 76,294 | **3.9335** | **84.17%** | **94.91%** | 39.33% | 46.00% | **100.00%** | 98.00% | **92.67%** | 0.8786 |

PPL decreases monotonically while exact capability does not. Reason falls by
4.50 points from 45k to 60k even though PPL improves, then reaches its maximum
at final. Public S600 peaks at 45k instead. Final is the best Q10 checkpoint
for Reason, teacher accuracy, and language modeling; 75k remains the clean
same-step architecture comparison.

## 7. Matched Architecture Comparison

Best S600 values are maxima over the six retained boundaries. Public V2 uses
step 75,000 for every model.

| Model | Params | Final PPL | Best Reason | Best teacher | Best Public S600 | Public V2 75k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain Transformer | 137.84M | 3.9491 | **87.67%** | **96.68%** | 39.00% | 35.40% |
| Shared-cache d32 | 140.31M | 3.9535 | 82.17% | 94.84% | 40.17% | 34.60% |
| Strict per-head d32 | 140.31M | **3.9022** | 79.50% | 94.64% | **40.67%** | **35.52%** |
| Strict per-head d64 | 141.09M | 3.9335 | 84.17% | 94.91% | 39.83% | **35.52%** |

Increasing the per-head cache from d32 to d64 recovers 4.67 Reason points at
the best boundary, but worsens final PPL by 0.0313 and does not improve the
matched full public score. At 75k both per-head models answer exactly 7,408 of
20,854 Public V2 examples correctly. Their predictions differ substantially,
but the discordant counts are exactly 1,485 versus 1,485 (`p=1.0`).

The 75k Reason family table localizes the synthetic gain:

| Family | Baseline | Shared d32 | Per-head d32 | Per-head d64 |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic | **69.33%** | 48.67% | 40.00% | 43.33% |
| Copy | **100.00%** | **100.00%** | 97.33% | **100.00%** |
| Reverse | 99.33% | 84.00% | 93.33% | **100.00%** |
| Rewrite | 82.00% | **96.00%** | 87.33% | 86.67% |

d64 mainly restores copy/reverse behavior and modestly improves arithmetic
over d32. It does not solve the large arithmetic deficit against the plain
Transformer.

## 8. Public V2

The matched 75k aggregate groups are:

| Group | Samples | Baseline | Shared d32 | Per-head d32 | Per-head d64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy core | 12,450 | 32.21% | 31.97% | **32.37%** | 31.85% |
| Expanded reasoning | 7,430 | 42.48% | 40.66% | 42.56% | **43.42%** |
| MMLU math/logic | 974 | 22.18% | 22.07% | 22.18% | **22.28%** |
| **All examples** | **20,854** | 35.40% | 34.60% | **35.52%** | **35.52%** |

Paired exact McNemar results for d64 at 75k are:

| Comparison | Accuracy delta | d64 only | Other only | Exact p |
| --- | ---: | ---: | ---: | ---: |
| d64 minus baseline | +0.12 pp | 1,501 | 1,475 | 0.64676 |
| d64 minus shared d32 | +0.92 pp | 1,769 | 1,577 | 0.00096 |
| d64 minus per-head d32 | 0.00 pp | 1,485 | 1,485 | 1.00000 |

Thus d64 significantly retains the per-head recovery over shared cache, but it
is statistically tied with both the baseline and d32. Its group profile shifts
about 0.9 points toward expanded reasoning while losing about 0.5 points on
legacy core relative to d32; this is redistribution, not a broad gain.

Final reaches 7,473/20,854 (`35.835%`, macro `32.273%`), 65 more correct than
75k. The paired final-versus-75k comparison is not resolved (`p=0.178`). Final
also remains tied with the baseline (`+0.44 pp`, `p=0.106`) and per-head d32
(`+0.31 pp`, `p=0.249`). BoolQ accounts for most of the final aggregate gain,
so it should not be described as uniform task improvement.

## 9. Routing And Geometry

Validation block entropy stays between 0.9750 and 0.9942 across retained
boundaries. At final, all eight free blocks appear in the top-1 histogram; the
least-used free block has 14 selections in the logged batch and the most-used
has 29. Final path diversity is 1.0. This rules out classic dead-block or
single-path collapse.

Reason routing is more concentrated than validation routing: normalized block
entropy is 0.8192 at 75k and 0.8786 at final. This is comparable to d32 final
at 0.8866, and capability rises while the metric moves in both directions.
There is no evidence here for a simple monotonic diversity-accuracy relation.

Position geometry does contract. The validation minimum internal angle falls
from 41.65 degrees at 15k to 14.61 degrees at final, while key/value fusion
entropies remain stable near 1.62/1.83. Geometry concentration is therefore a
real remaining issue, but it has not produced functional block death in this
run.

## 10. Systems Cost And Decision

| System metric | Baseline | Shared d32 | Per-head d32 | Per-head d64 |
| --- | ---: | ---: | ---: | ---: |
| Median training throughput | 618,738 | 219,409 | 170,711 | 142,699 token/s |
| End-to-end runtime | 2 h 25 min | 6 h 48 min | 8 h 32 min | about 10 h 8 min |
| Peak allocation/rank | 32.9 GiB | 48.3 GiB | 72.5 GiB | 105.0 GiB |

d64 is 16.4% slower than d32 by median throughput, takes about 18.5% longer,
and uses 44.8% more peak allocation. It remains about 4.34x slower than the
plain baseline.

The capacity ablation is informative but not a default promotion. It shows
that d32 compression contributes to the synthetic Reason deficit, especially
copy/reverse stability. It does not improve matched Public V2, does not improve
PPL over d32, and carries a large memory/runtime cost. Keep d32 as the practical
strict per-head setting and retain d64 as the high-capacity research variant.
Use Q10 final for its strongest Reason/LM state and 75k for matched
architecture comparisons.
