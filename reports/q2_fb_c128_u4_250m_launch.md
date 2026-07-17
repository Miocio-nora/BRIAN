# Q2 FB C128-U4 250M Launch

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 0-1 only, DDP2

## 1. Question

The Q1 C128-U1 run selected Full-Bank visibility over Depth-Prefix visibility.
The C512-U1 follow-up was faster but lost most sequence reasoning capability.
This experiment tests whether allowing continuous gradients across four C128
chunks improves or destabilizes FB while retaining the accepted 128-token
completion granularity.

| Arm | Visibility | Chunk | U | Completion stride | Gradient horizon | Token budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q1 reference | full bank | 128 | 1 | 128 | 128 | 250,019,840 |
| Q2 candidate | full bank | 128 | 4 | 128 | 512 | 250,019,840 |

Unlike C512-U1, this is a controlled gradient-horizon ablation. Forward cache
values, completion boundaries, route policy, and model parameters are the same
before optimizer updates. U4 intentionally changes gradients by retaining the
autograd graph across four consecutive chunks.

## 2. Controlled Contract

The candidate directly reuses the Q1-FB Triton model config. It keeps data,
token order, seed, initialization, optimizer, constant `3e-4` learning rate,
routing curriculum, slow-noise schedule, loss weights, global batch 32, local
batch 16, legacy validation, checkpoint cadence, and both S600 suites.

The only controlled train-config change is:

```yaml
stateful_tbptt:
  chunk_size: 128
  detach_interval_chunks: 4
```

The run uses 3,815 optimizer steps and evaluates at steps 1,272, 2,544, and
3,815. Model-only retained checkpoints preserve all three evaluation points.

## 3. Decision Rule

U4 must be judged against U1 using PPL, reasoning exact, teacher accuracy,
public average, per-family and hard-difficulty reasoning, route/cache health,
throughput, and memory. PPL alone is not an acceptance criterion.

The result also diagnoses the C512 failure. If C128-U4 remains healthy, the
C512 regression points to coarse completion granularity. If C128-U4 regresses
similarly, the longer continuous gradient horizon is also implicated.

## 4. DDP2 Smoke

A 20-step smoke completed on physical GPU 0-1 with finite loss and the intended
stateful-TBPTT contract.

| Metric | FB C128-U1 | FB C128-U4 smoke |
| --- | ---: | ---: |
| Chunks / backward groups | 16 / 16 | 16 / 4 |
| Gradient horizon | 128 tokens | 512 tokens |
| Last-10 median throughput | 40,505 token/s | 37,271 token/s |
| Last-10 U4 range | - | 34,933-37,597 token/s |
| Peak allocated per rank | 8,465 MiB | 17,531 MiB |
| Final normalized block entropy | healthy | 0.9895 |
| Final path diversity | 1.0 | 1.0 |

U4 is approximately 8% slower and uses about 2.1x the allocated memory. This
is expected: U4 retains four chunk graphs so gradients can cross completion
boundaries. It is a credit-assignment experiment, not a speed optimization.
At the smoke median, pure 250M-token training is approximately 1 h 52 min;
checkpoint benchmarks and I/O increase wall time.

## 5. Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q2_fb_c128_u4_250m.sh
```

The formal run is resumable from `checkpoint_latest` and writes all benchmark
artifacts below its run directory.
