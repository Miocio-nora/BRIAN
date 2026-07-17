# Q4 FB C128-U2 250M Launch

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

## 5. Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q4_fb_c128_u2_250m.sh
```

The run is resumable from `checkpoint_latest`; all scheduled benchmark outputs
are written below its run directory.
