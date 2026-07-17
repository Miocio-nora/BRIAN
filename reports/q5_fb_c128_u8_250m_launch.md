# Q5 FB C128-U8 250M Result

**Date:** 2026-07-18

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 2-3 only, DDP2

## 1. Question

The matched C128-U4 run improved final S600 reasoning exact from 27.67% to
37.67%, while Q4 concurrently tests U2. Q5 tests the next logarithmic gradient
horizon point before committing resources to full-sequence U16.

| Arm | Chunk | U | Backward groups | Gradient horizon | Token budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 reference | 128 | 1 | 16 | 128 | 250,019,840 |
| Q2 reference | 128 | 4 | 4 | 512 | 250,019,840 |
| Q5 candidate | 128 | 8 | 2 | 1,024 | 250,019,840 |

U8 retains eight consecutive C128 chunk graphs, detaches at the half-sequence
boundary, and executes two backward groups over each 2,048-token sequence.

## 2. Controlled Contract

Q5 directly extends the Q1 C128-U1 configuration. The only functional override
is:

```yaml
stateful_tbptt:
  detach_interval_chunks: 8
```

Forward values at fixed weights, FB completion boundaries, model, data, token
order, seed, routing, BDRE depth score, optimizer, learning rate, losses,
global batch, legacy validation, checkpoint cadence, and both S600 suites remain
matched.

## 3. Decision Rule

U8 is accepted over U4 only if medium/hard reasoning or teacher accuracy
continues to improve without material copy/public regression or route/cache
instability. PPL alone is not an acceptance criterion.

If U8 improves, the prepared U16 endpoint becomes the next gradient-horizon
test. If U8 is flat, U4 remains preferable on memory. If U8 regresses, the
U2/U4 region contains the likely optimum and U16 is held.

Concurrent U2 and U8 jobs share host resources, so live throughput is not an
accepted isolated speed comparison. Quality and per-rank memory are the primary
outputs.

## 4. DDP2 Smoke

A 20-step DDP2 smoke completed on physical GPU 2-3 while Q4-U2 was training on
GPU 0-1. All losses were finite and the runtime contract matched U8:

| Metric | Result |
| --- | ---: |
| Chunks / backward groups | 16 / 2 |
| Detach interval / gradient horizon | 8 / 1,024 tokens |
| Last-10 median throughput | 37,351 token/s |
| Last-10 range | 34,753-37,566 token/s |
| Peak allocated per rank | 29,505 MiB |
| Final normalized block entropy | 0.9901 |
| Final path diversity | 1.0 |

U8 fits comfortably on a B200 at local batch 16 and uses approximately 58.6%
of the U16 smoke's allocated memory. Its throughput is not an isolated speed
result because U2 was concurrently using the same host.

## 5. Formal Result

The formal run completed all 3,815 optimizer steps and all six scheduled
benchmark jobs in approximately 2 h 05 min.

### 5.1 Checkpoint trajectory

| Step | Legacy PPL | Reason exact | Teacher acc. | Public avg. |
| ---: | ---: | ---: | ---: | ---: |
| 1,272 | 15.083 | 1.17% | 54.13% | 30.00% |
| 2,544 | 9.616 | 20.17% | 72.91% | 32.33% |
| 3,815 | 8.042 | 31.83% | 80.19% | 32.00% |

U8 leads every matched arm on reasoning at step 2,544, but U4 improves more
strongly over the final third of training and finishes ahead.

### 5.2 Final gradient-horizon matrix

| U | Horizon | PPL | Reason exact | Teacher acc. | Public S600 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 8.128 | 27.67% | 77.78% | **34.67%** |
| 2 | 256 | 8.085 | 16.00% | 74.41% | 33.50% |
| 4 | 512 | **7.994** | **37.67%** | **83.06%** | 33.33% |
| 8 | 1,024 | 8.042 | 31.83% | 80.19% | 32.00% |

The quality curve is non-monotonic. U8 remains better than U1, but extending
the gradient horizon beyond U4 does not improve the final aggregate result.

### 5.3 U4 versus U8

| Reasoning slice | U4 | U8 | U8 - U4 |
| --- | ---: | ---: | ---: |
| Easy | 52.00% | 47.50% | -4.50 pp |
| Medium | 38.50% | 22.50% | -16.00 pp |
| Hard | 22.50% | **25.50%** | +3.00 pp |
| Arithmetic | 7.33% | 4.00% | -3.33 pp |
| Copy | 52.67% | 48.00% | -4.67 pp |
| Reverse | 45.33% | 38.67% | -6.66 pp |
| Rewrite | 45.33% | 36.67% | -8.66 pp |

On the identical 600 reasoning samples, 108 are solved by both arms, 118 only
by U4, and 83 only by U8. An exact paired binomial test on the 201 discordant
samples gives `p=0.0163`. This supports selecting U4 on this suite, while the
single seed and synthetic task mix still limit broader causal claims.

The public benchmark is not meaningfully different: U4 scores 33.33% and U8
32.00%, with 46 U4-only versus 38 U8-only outcomes (`p=0.445`). U8 therefore
provides no public-suite evidence that compensates for its reasoning deficit.

### 5.4 Routing and cost

| Metric | U4 | U8 |
| --- | ---: | ---: |
| Average reasoning route steps | 13.617 | 13.270 |
| Reasoning route entropy | 1.913 | 1.820 |
| Normalized reasoning block entropy | 0.685 | 0.665 |
| Reasoning recur ratio | 0.357 | 0.452 |
| Reasoning skip ratio | 0.586 | 0.408 |
| Median train throughput | 39,555 | 38,868 token/s |
| Peak allocated per rank | 17,531 | 29,505 MiB |

U8 has no classic route collapse: final training path diversity is 1.0 and
block use remains broad. It nevertheless learns a shorter, more recurrent, and
slightly more concentrated benchmark policy than U4. This is a correlated
behavioral difference, not proof that routing changes cause the quality gap.

U8 is 1.7% slower and uses 68.3% more allocated memory than U4. Since it also
finishes behind on the primary capability suite, it is Pareto-dominated for the
next full-budget run.

## 6. Decision

Select **C128-U4** for the first formal 5B FB run. Hold U16: U8 already shows
that extending beyond U4 is not monotonically beneficial, so a full-sequence
endpoint is no longer the highest-value use of two GPUs.

The 5B launch must use a new U4 configuration extending
`cpbc_r125_5b_fb_u1_c128_triton_fused_reader_incremental_ddp2_legacyval.yaml`
and overriding only `detach_interval_chunks: 4`. The older
`cpbc_r125_5b_fb_u4_c128_ddp2_legacyval.yaml` inherits the pre-Triton backend
and must not be used for this run.

At the measured U4 throughput, 5B pure training is approximately 35.1 hours on
two B200s; checkpoint validation and six benchmark stages should put the
end-to-end estimate near 36-38 hours.

## 7. Launch

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_q5_fb_c128_u8_250m.sh
```

The run is resumable from `checkpoint_latest`; all scheduled benchmark outputs
are written below its run directory.
