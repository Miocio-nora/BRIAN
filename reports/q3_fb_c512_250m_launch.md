# Q3 FB C512-U1 250M Launch

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
