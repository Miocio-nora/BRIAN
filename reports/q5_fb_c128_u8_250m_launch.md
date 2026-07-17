# Q5 FB C128-U8 250M Launch

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

## 5. Launch

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_q5_fb_c128_u8_250m.sh
```

The run is resumable from `checkpoint_latest`; all scheduled benchmark outputs
are written below its run directory.
