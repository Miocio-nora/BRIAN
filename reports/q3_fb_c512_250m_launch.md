# Q3 FB C512-U1 250M Launch and Result

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 0-1 only, DDP2

## 1. Question

The completed Q1 pilot selected CPBC-FB over CPBC-DP at C128. This pilot asks
whether a coarser C512 completion boundary retains that quality signal while
reducing chunk and backward fragmentation.

| Arm | Visibility | Chunk | U | Completion boundaries | Gradient horizon | Token budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q1 reference | full bank | 128 | 1 | 16 | 128 | 250,019,840 |
| Q3 candidate | full bank | 512 | 1 | 4 | 512 | 250,019,840 |

This is an operational C128-versus-C512 comparison, not a fully isolated
chunk-granularity ablation: U1 makes the gradient horizon grow with the chunk.
The later comparison against FB-C128-U4 will match the 512-token gradient
horizon and isolate completion granularity more cleanly.

## 2. Controlled Contract

The candidate keeps the Q1-FB data, token order, seed, optimizer, constant
`3e-4` learning rate, routing curriculum, slow-noise schedule, loss weights,
model dimensions, global batch 32, local batch 16, Triton reader, legacy
validation, and benchmark definitions. It runs 3,815 optimizer steps with
65,536 tokens per step.

The only model execution change is `execution.chunk_size: 512`; the matching
stateful-TBPTT chunk is also 512. Under seed 1, C128 and C512 each contain
164,884,105 state elements and have the same complete initialization hash:

```text
bf32b57988cca5c2e114a60cb5bf7588055d3b32372494e414a7b9d6bd784beb
```

Legacy validation, strict incremental reasoning S600, public S600, retained
checkpoints, route visualization, and BDRE cache visualization run at steps
1,272, 2,544, and 3,815.

## 3. DDP2 Smoke

A 20-step smoke completed on physical GPU 0-1 in the `brian-sphere`
environment.

| Metric | FB C128-U1 Q1 | FB C512-U1 smoke |
| --- | ---: | ---: |
| Last-10 median throughput | 40,505 token/s | 122,554 token/s |
| Last-10 range | approximately 36-42k | 122,236-123,154 token/s |
| Peak allocated per rank | 8,465 MiB | 17,633 MiB |
| Chunks / backward groups | 16 / 16 | 4 / 4 |
| Gradient horizon | 128 tokens | 512 tokens |

The C512 candidate is approximately 3.0x faster than the completed Q1-FB run
while using about 2.1x its allocated memory. All 20 losses were finite; final
normalized block entropy was 0.9887 and path diversity was 1.0. Pure 250M-token
training is approximately 34 minutes at the smoke rate. Checkpoint evaluation,
compilation, checkpoint I/O, and W&B add wall time.

## 4. Decision Rule

The candidate must not be selected by PPL alone. The three-checkpoint matrix
must compare validation/PPL, reasoning exact, teacher accuracy, public average
and task breakdown, per-family reasoning accuracy, hard-difficulty accuracy,
route entropy and coverage, route length, compiler K/V entropy, writer and
last-step mass, throughput, and memory.

C512 is accepted as the operational FB default only if its speedup does not
come with a consistent capability regression. The later C128-U4 result remains
necessary before attributing any difference specifically to completion
granularity rather than the longer gradient horizon.

## 5. Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q3_fb_c512_250m.sh
```

The formal run is resumable from `checkpoint_latest` and writes benchmark
artifacts under its run directory.

## 6. Completed Result

The run completed all 3,815 steps and all three checkpoint benchmark stages on
2026-07-17. C512 delivered the expected systems gain but failed the capability
acceptance criterion.

### 6.1 Checkpoint Matrix

| Chunk | Step | PPL | Reason exact | Teacher acc | Public avg | PIQA | HellaSwag | ARC-Easy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1,272 | 14.4939 | 0.0250 | 0.5721 | 0.3350 | 0.445 | 0.260 | 0.300 |
| 512 | 1,272 | **14.1717** | 0.0167 | 0.5214 | 0.2983 | 0.440 | 0.205 | 0.250 |
| 128 | 2,544 | 9.7914 | **0.1517** | **0.7285** | 0.3333 | 0.445 | 0.245 | 0.310 |
| 512 | 2,544 | **9.5799** | 0.0583 | 0.6423 | **0.3483** | 0.460 | 0.270 | 0.315 |
| 128 | 3,815 | **8.1283** | **0.2767** | **0.7778** | **0.3467** | 0.470 | 0.280 | 0.290 |
| 512 | 3,815 | 8.1301 | 0.0950 | 0.6990 | 0.3367 | 0.460 | 0.255 | 0.295 |

Final PPL is effectively identical, but C512 loses 18.17 percentage points of
reasoning exact and 7.88 points of teacher accuracy. On paired samples, C128
alone solves 133 reasoning examples while C512 alone solves 24; the exact
McNemar p-value is approximately `1.7e-19`. Public S600 is statistically
unresolved (`p=0.576`) and does not offset the reasoning regression.

### 6.2 Capability Decomposition

| Task | C128 exact | C512 exact | Difference |
| --- | ---: | ---: | ---: |
| Arithmetic | 0.0200 | **0.0400** | +0.0200 |
| Copy | **0.5933** | 0.0867 | -0.5067 |
| Reverse | **0.3667** | 0.1267 | -0.2400 |
| Rewrite | 0.1267 | 0.1267 | 0.0000 |

The regression is concentrated in sequence retention and ordering, the two
families where FB-C128 had its strongest signal. Hard-difficulty exact also
falls from 0.095 to 0.015. This is not explained by aggregate route collapse:
C512 finishes with normalized block entropy 0.9472, path diversity 0.9922, and
14.875 average route steps. Compiler K/V entropy and last-step mass also remain
close to the healthy C128 ranges.

### 6.3 Efficiency

| Metric | FB C128-U1 | FB C512-U1 | Change |
| --- | ---: | ---: | ---: |
| Median training throughput | 40,505 token/s | 124,382 token/s | 3.07x |
| Median optimizer-step time | 1.618 s | 0.527 s | 0.33x |
| Peak allocated per rank | 8,465 MiB | 17,633 MiB | 2.08x |

## 7. Decision

FB-C512-U1 is rejected as the operational default. It is a successful systems
shape but an unacceptable quality trade at this token budget: language-model
loss hides a large sequence-capability regression.

Because C512-U1 changes both completion granularity and the U1 gradient horizon,
this run does not identify which change is causal. The next controlled step is
FB-C128-U2/U4. In particular, C128-U4 matches the 512-token gradient horizon
while retaining 128-token completion boundaries. If C128-U4 remains healthy,
the C512 failure can be attributed primarily to coarse bank completion rather
than the longer gradient horizon. C256 should be considered only after that
diagnosis.
