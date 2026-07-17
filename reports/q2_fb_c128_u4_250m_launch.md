# Q2 FB C128-U4 250M Result

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

## 5. Formal Result

The formal run completed all 3,815 steps and all six scheduled benchmark jobs.
The end-to-end wall time, including checkpoint evaluation and benchmark I/O,
was approximately 2 h 05 min.

### 5.1 Checkpoint trajectory

| Step | Arm | Legacy PPL | Reason exact | Teacher acc. | Public avg. |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,272 | C128-U1 | 14.494 | 2.50% | 57.21% | 33.50% |
| 1,272 | C128-U4 | 15.135 | 1.50% | 53.19% | 31.83% |
| 2,544 | C128-U1 | 9.791 | 15.17% | 72.85% | 33.33% |
| 2,544 | C128-U4 | 9.643 | 11.50% | 74.40% | 32.50% |
| 3,815 | C128-U1 | 8.128 | 27.67% | 77.78% | 34.67% |
| 3,815 | C128-U4 | **7.994** | **37.67%** | **83.06%** | 33.33% |

U4 starts behind U1, crosses it late, and finishes 10.00 percentage points
higher on reasoning exact. Its final PPL improvement is only 1.65%, so PPL does
not explain or reliably select the reasoning result.

### 5.2 Final reasoning breakdown

| Slice | C128-U1 | C128-U4 | U4 - U1 |
| --- | ---: | ---: | ---: |
| Easy | 53.50% | 52.00% | -1.50 pp |
| Medium | 20.00% | 38.50% | +18.50 pp |
| Hard | 9.50% | 22.50% | +13.00 pp |
| Arithmetic | 2.00% | 7.33% | +5.33 pp |
| Copy | 59.33% | 52.67% | -6.67 pp |
| Reverse | 36.67% | 45.33% | +8.67 pp |
| Rewrite | 12.67% | 45.33% | +32.66 pp |

On the identical 600 samples, 112 are solved by both arms, 54 only by U1, and
114 only by U4. The net gain is therefore not caused by a handful of examples.
It is concentrated in medium/hard sequence transformations, especially
rewrite, while simple copy regresses.

### 5.3 Public benchmark and routing

Public accuracy does not show a general-capability gain: U4 finishes at 33.33%
versus 34.67% for U1. The paired outcomes are 163 both-correct, 45 U1-only, and
37 U4-only. This eight-example difference is weak evidence and should be read
as effectively inconclusive at this sample size. ARC-Easy changes from 29.0%
to 30.5%, HellaSwag from 28.0% to 23.5%, and PIQA from 47.0% to 46.0%.

The reasoning gain is not explained by extra inference depth or route collapse:

| Reasoning routing metric | C128-U1 | C128-U4 |
| --- | ---: | ---: |
| Average route steps | 13.618 | 13.617 |
| Route entropy | 1.927 | 1.913 |
| Normalized block-load entropy | 0.697 | 0.685 |
| Recur ratio | 0.464 | 0.357 |
| Skip ratio | 0.452 | 0.586 |

There is no evidence of a U4-specific route collapse, although its benchmark
block-load entropy is slightly lower rather than better. U4 changes the learned
routing mix toward more skip and less recurrence, but uses essentially the same
total route depth. The reasoning gain therefore is not a diversity gain.

### 5.4 Cost

| Formal training metric | C128-U1 | C128-U4 | Change |
| --- | ---: | ---: | ---: |
| Median throughput after warmup | 40,505 token/s | 39,555 token/s | -2.35% |
| Peak allocated per rank | 8,465 MiB | 17,531 MiB | +107.1% |
| Last-100 mean LM loss | 3.6646 | 3.6444 | -0.0202 |

U4 has little throughput cost because it reduces the number of backward groups
from 16 to 4, but retaining four chunk graphs approximately doubles activation
memory.

## 6. C512 Diagnosis

C128-U4 and C512-U1 both expose a nominal 512-token gradient horizon, but their
final results differ sharply:

| Arm | Completion stride | Gradient horizon | PPL | Reason exact | Teacher acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| C128-U1 | 128 | 128 | 8.128 | 27.67% | 77.78% |
| C128-U4 | 128 | 512 | 7.994 | **37.67%** | **83.06%** |
| C512-U1 | 512 | 512 | 8.130 | 9.50% | 69.90% |

This rules out the continuous 512-token gradient horizon as a sufficient
explanation for the C512 regression. The coarse 512-token completion/prefill
approximation remains the primary suspect. It also demonstrates that
near-identical validation PPL can hide a 28-point reasoning difference between
CPBC policies.

## 7. Decision

C128-U4 is a positive gradient-horizon result for the current FB mechanism,
with a substantial sequence-reasoning gain and no routing instability. It is
not yet evidence of a broad public-benchmark improvement, and the task-level
trade-off includes a copy regression. Since this is one controlled seed, U4
should remain a candidate rather than an unconditional default.

The next informative ablation is C128-U2. It tests whether a 256-token gradient
horizon captures the U4 gain with lower activation memory. A repeat seed is
only necessary before making a publication-level causal claim.

## 8. Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q2_fb_c128_u4_250m.sh
```

The formal run is resumable from `checkpoint_latest` and writes all benchmark
artifacts below its run directory.
