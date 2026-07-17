# Q6 FB C128-U16 250M Preparation

**Date:** 2026-07-18

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 2-3 only, DDP2

## 1. Question

The matched C128-U4 run improved final S600 reasoning exact from 27.67% to
37.67%. Q4 tests the intermediate U2 point and Q5 tests U8. Q6 prepares the
opposite endpoint: whether removing all within-sequence TBPTT detach boundaries
continues the gain or destabilizes optimization.

| Arm | Chunk | U | Backward groups | Gradient horizon | Token budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 reference | 128 | 1 | 16 | 128 | 250,019,840 |
| Q2 reference | 128 | 4 | 4 | 512 | 250,019,840 |
| Q6 endpoint | 128 | 16 | 1 | 2,048 | 250,019,840 |

For a 2,048-token training sequence, U16 retains all 16 C128 chunk graphs and
performs one backward group. It is full-sequence BPTT for the recurrent/cache
state, while the optimizer still takes one step per global batch as in every
matched arm.

## 2. Controlled Contract

Q6 directly extends the Q1 C128-U1 configuration. The only functional override
is:

```yaml
stateful_tbptt:
  detach_interval_chunks: 16
```

Forward values at fixed weights, FB completion boundaries, model, data, token
order, seed, routing, optimizer, learning rate, losses, global batch, legacy
validation, checkpoint cadence, and both S600 suites remain matched.

## 3. Decision Rule

U16 is an endpoint diagnostic, not automatically the preferred production
shape. It must improve medium/hard reasoning or teacher accuracy over U4 without
material copy/public regression or route/cache instability. PPL alone is not
an acceptance criterion.

U16 should be launched only if U8 improves over U4 and leaves the upper endpoint
unresolved. If U8 is flat or regresses, U16 remains held while the U2/U4 region
is selected. Any future concurrent run must treat live throughput as
host-contended rather than an isolated speed comparison.

## 4. DDP2 Smoke

A 20-step DDP2 smoke completed on physical GPU 2-3 while Q4-U2 was training on
GPU 0-1. All losses were finite and the runtime metrics confirmed the intended
full-sequence graph:

| Metric | Result |
| --- | ---: |
| Chunks / backward groups | 16 / 1 |
| Detach interval / gradient horizon | 16 / 2,048 tokens |
| Last-10 median throughput | 38,545 token/s |
| Last-10 range | 34,581-38,939 token/s |
| Peak allocated per rank | 50,323 MiB |
| Final normalized block entropy | 0.9900 |
| Final path diversity | 1.0 |

U16 fits comfortably on a B200 at local batch 16. Its allocated activation
footprint is approximately 5.9x U1 and 2.9x U4. The measured throughput is
competitive, but is not an isolated speed result because U2 was concurrently
using the same host.

## 5. Status

The U16 smoke and formal configuration are retained as an endpoint asset, but
the formal 250M run is intentionally held. U8 was selected as the next
higher-information gradient-horizon experiment.

Prepared command, not currently launched:

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_q6_fb_c128_u16_250m.sh
```

The run is resumable from `checkpoint_latest`; all scheduled benchmark outputs
are written below its run directory.
