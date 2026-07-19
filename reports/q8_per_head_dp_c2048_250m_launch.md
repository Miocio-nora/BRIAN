# Q8 Strict Per-Head DP-C2048 250M Pilot

**Date:** 2026-07-19

**Branch:** `rc-kv-per-head-cache`

## 1. Question

This pilot asks whether strict per-head RC-KV remains viable on the much faster
CPBC-DP C2048 execution path. It follows the formal FB C128-U4 result, which
matched DP quality closely enough that its substantially lower throughput was
not justified.

## 2. Controlled Architecture

```text
cache layout:      strict per-head
key cache dim:     32 per head
value cache dim:   32 per head
visibility:        CPBC-DP
chunk size:        2048
detach interval:   U1
reader:            Triton fused
prefix compiler:   recompute
route pool:        8 blocks
maximum route:     16 steps
```

Per-head d32 contains 140,308,105 parameters, exactly matching shared-d32.
This is therefore the primary architecture comparison; d16 is retained only as
a lower-memory efficiency option and is not used in Q8.

## 3. Training Contract

```text
budget:             250,019,840 tokens
updates:            3,815
hardware:           GPU 0-1, two B200 GPUs
local batch:         16 sequences per rank
global batch:        32 sequences
sequence length:     2048
data:                balanced R125 training corpus
validation:          legacy validation split
checkpoint steps:    1,272, 2,544, and 3,815
```

Reasoning S600 and public S600 are deliberately disabled inside the active DDP
job. The retained checkpoints will be evaluated afterward by independent
single-GPU processes. This preserves the benchmark contract while preventing a
rank from waiting in an NCCL barrier during a long rank-0-only benchmark.

## 4. Acceptance

Before launch, Q8 must pass:

- configuration validation and the parameter-count audit;
- strict per-head Triton forward/backward tests;
- a 20-step DDP2 production-shape smoke at local batch 16 and sequence 2048;
- finite loss, synchronized gradients, and stable CUDA memory;
- a throughput estimate based on compiled smoke steps.

The run will not be interpreted from PPL alone. Checkpoint selection will use
legacy validation, reasoning exact, teacher accuracy, public S600, routing
entropy, and training efficiency together.

## 5. Smoke Acceptance

The production-shape smoke completed 20 updates on GPU 0-1 with local batch 16
per rank and sequence length 2048.

| Measurement | Result |
| --- | ---: |
| Steady median global throughput | 147,259 token/s |
| Steady range after step 5 | 111,519-149,579 token/s |
| Last-step throughput | 146,461 token/s |
| Peak allocated memory per rank | 74,279 MiB |
| Final loss | 44.5686 |
| Missing gradient parameters | 0 |

The first update took 12.0 seconds because it included Triton/Inductor
compilation. Compiled updates take approximately 0.44 seconds. Validation and
routing metrics remained finite, manual stateful DDP synchronization completed
on both ranks, and normalized block entropy was 0.9962.

At the measured median, 250,019,840 pure training tokens require approximately
28.3 minutes. Three validation/checkpoint boundaries should put end-to-end
runtime near 35-45 minutes. The smoke therefore accepts local batch 16 without
gradient accumulation; reducing batch size is unnecessary.

## 6. Formal Launch

The formal DDP2 run started on GPU 0-1 at `2026-07-19 16:47 JST` in tmux
session `brian_q8_per_head_dp_c2048_g01`:

```bash
bash scripts/run_q8_per_head_dp_c2048_250m.sh
```

W&B run: `mio_nora/brian-sphere-llm/3morcq00`

Startup acceptance reached step 79 with a 149,887 token/s median after step 5,
74,279 MiB peak allocation per rank, finite loss, normalized block entropy
0.9934, full monitored path diversity, and zero locally missing gradient
parameters. No OOM, NCCL error, or data/config mismatch was observed.
