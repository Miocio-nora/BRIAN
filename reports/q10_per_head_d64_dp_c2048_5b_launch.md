# Q10 Strict Per-Head d64 DP-C2048 5B Launch

**Date:** 2026-07-20

**Branch:** `rc-kv-per-head-cache`

**Status:** production-shape smoke accepted; formal launch pending

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

At the measured steady rate, pure 5B training is approximately 10.3 hours.
Checkpointing, validation, startup compilation, and synchronization put the
expected end-to-end time near 10.5-11.5 hours.

The 1.6 GiB temporary smoke checkpoint is not a research asset and is removed
after extracting these acceptance metrics.

## 4. Configuration

```text
configs/model/brian_r125_bdre_cpbc_dp_c2048_per_head_triton_recompute_d64.yaml
configs/train/smoke_q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2.yaml
configs/train/q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2_legacyval.yaml
scripts/run_q10_per_head_d64_dp_c2048_5b.sh
```

Formal launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q10_per_head_d64_dp_c2048_5b.sh
```

The 75k checkpoint remains the primary capability-selection boundary. The
legacy S600 reports and Capability V2 full public suite should be run after DDP
exits. Full GSM8K/MATH-500 is optional for Q10 because the completed three-way
matrix established that it is at the accuracy floor for this scale.
