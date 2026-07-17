# Q4 FB C128-U2 250M Result

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 0-1 only, DDP2

## 1. Question

The matched C128-U4 run improved final S600 reasoning exact from 27.67% to
37.67%, but approximately doubled activation memory relative to C128-U1. This
experiment tests whether the intermediate U2 gradient horizon retains that
reasoning gain at lower memory cost.

| Arm | Visibility | Chunk | U | Gradient horizon | Token budget |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 reference | full bank | 128 | 1 | 128 | 250,019,840 |
| Q4 candidate | full bank | 128 | 2 | 256 | 250,019,840 |
| Q2 reference | full bank | 128 | 4 | 512 | 250,019,840 |

U controls only the number of consecutive chunks whose recurrent/cache state
remains connected in autograd. C128-U2 keeps two 128-token chunk graphs
connected, runs eight backward groups over a 2,048-token sequence, and detaches
state every 256 tokens.

## 2. Controlled Contract

Q4 directly extends the Q1 C128-U1 train configuration. The only functional
override is:

```yaml
stateful_tbptt:
  detach_interval_chunks: 2
```

Data, token order, seed, initialization, model, FB completion semantics,
optimizer, constant `3e-4` learning rate, routing curriculum, slow-noise
schedule, loss weights, local/global batch, legacy validation, checkpoint
cadence, and both S600 suites remain matched.

The run uses 3,815 optimizer steps and evaluates at steps 1,272, 2,544, and
3,815. Retained model checkpoints preserve all three points.

## 3. Decision Rule

The primary comparison is the three-arm U1/U2/U4 trajectory. U2 is useful if it
improves medium/hard reasoning and teacher accuracy over U1 without the full U4
memory cost. Public S600, task-family trade-offs, route/cache health, PPL,
throughput, and memory remain required secondary checks. PPL alone is not an
acceptance criterion.

## 4. DDP2 Smoke

A 20-step smoke completed on physical GPU 0-1 with finite loss and the intended
stateful-TBPTT contract.

| Metric | C128-U1 | C128-U2 | C128-U4 |
| --- | ---: | ---: | ---: |
| Chunks / backward groups | 16 / 16 | 16 / 8 | 16 / 4 |
| Gradient horizon | 128 tokens | 256 tokens | 512 tokens |
| Last-10 median throughput | 39,992 | 39,361 | 37,271 token/s |
| Peak allocated per rank | 8,462 | 11,217 | 17,531 MiB |

U2 is 1.6% slower than U1 in the matched smoke and adds 2,755 MiB of peak
allocated memory. It retains 36% of the U1-to-U4 memory increase while providing
half of U4's gradient horizon. Final normalized block entropy is 0.9903 and path
diversity is 1.0; no early routing instability is visible.

## 5. Formal Result

The formal run completed all 3,815 optimizer steps and all six scheduled
benchmark jobs in approximately 2 h 07 min.

### 5.1 Checkpoint trajectory

| Step | Legacy PPL | Reason exact | Teacher acc. | Public avg. |
| ---: | ---: | ---: | ---: | ---: |
| 1,272 | 14.428 | 2.50% | 53.86% | 31.17% |
| 2,544 | 9.825 | 11.67% | 66.45% | 33.00% |
| 3,815 | 8.085 | 16.00% | 74.41% | 33.50% |

U2 improves continuously with training, but its final reasoning result remains
below both matched references:

| Final metric | C128-U1 | C128-U2 | C128-U4 |
| --- | ---: | ---: | ---: |
| Legacy PPL | 8.128 | 8.085 | **7.994** |
| Reason exact | 27.67% | 16.00% | **37.67%** |
| Teacher accuracy | 77.78% | 74.41% | **83.06%** |
| Public S600 | **34.67%** | 33.50% | 33.33% |

The 0.5% PPL improvement over U1 accompanies an 11.67-point reasoning
regression. Validation PPL therefore does not select the gradient horizon.

### 5.2 Reasoning breakdown

| Slice | C128-U1 | C128-U2 | C128-U4 |
| --- | ---: | ---: | ---: |
| Easy | 53.50% | 20.50% | 52.00% |
| Medium | 20.00% | 22.00% | 38.50% |
| Hard | 9.50% | 5.50% | 22.50% |
| Arithmetic | 2.00% | 2.00% | 7.33% |
| Copy | 59.33% | 38.67% | 52.67% |
| Reverse | 36.67% | 2.00% | 45.33% |
| Rewrite | 12.67% | 21.33% | 45.33% |

The U2 regression is concentrated in reverse and copy rather than distributed
uniformly across tasks. On the identical 600 samples, U1 and U2 solve 48
together, 118 are U1-only, and 48 are U2-only. Against U4, 33 are U2-only and
163 are U4-only. This is a broad paired-sample difference, not a few-example
artifact.

### 5.3 Public benchmark and routing

The public result is effectively inconclusive. U2 finishes at 33.50% versus
34.67% for U1 and 33.33% for U4. Against U1, the paired counts are 46 U1-only
and 39 U2-only; against U4 they are 40 U2-only and 39 U4-only.

| Final reasoning routing metric | C128-U1 | C128-U2 | C128-U4 |
| --- | ---: | ---: | ---: |
| Average route steps | 13.618 | 13.567 | 13.617 |
| Route entropy | 1.927 | 1.832 | 1.913 |
| Normalized block-load entropy | 0.697 | 0.652 | 0.685 |
| Recur ratio | 0.464 | 0.520 | 0.357 |
| Skip ratio | 0.452 | 0.425 | 0.586 |

U2 does not exhibit classic single-path collapse: route depth remains matched,
training path diversity is 1.0, and all blocks remain active. It does learn a
more recurrent and somewhat more concentrated benchmark routing policy. That
correlates with, but does not by itself establish the cause of, the sequence
task regression.

### 5.4 Cost

| Formal training metric | C128-U1 | C128-U2 | C128-U4 |
| --- | ---: | ---: | ---: |
| Median throughput after warmup | 40,505 | 38,748 | 39,555 token/s |
| Peak allocated per rank | 8,465 | 11,217 | 17,531 MiB |
| Last-100 mean LM loss | 3.6646 | 3.6597 | 3.6444 |

U2 adds 32.5% peak allocated memory over U1. Its throughput was measured while
part of the run overlapped with the U8 job, so it is not an isolated speed
ranking.

## 6. Decision

C128-U2 is rejected as the current quality/default candidate. It neither
recovers the U4 reasoning gain nor improves public accuracy, despite nearly
identical PPL. The observed U1/U2/U4 curve is non-monotonic at this seed, so
gradient horizon should be treated as an optimization-policy choice rather
than a smooth capacity knob.

This single-seed result is sufficient for selecting the next engineering
candidate, but not for claiming that U2 is generally inferior. The concurrently
running U8 arm is required before deciding whether U4 is a local optimum or
whether longer horizons resume the positive trend.

## 7. Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q4_fb_c128_u2_250m.sh
```

The run is resumable from `checkpoint_latest`; all scheduled benchmark outputs
are written below its run directory.
