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

## 7. Completed 250M Result

Q8 completed all 3,815 updates at `2026-07-19 17:15 JST`. End-to-end runtime
was 27 minutes 37 seconds, including compilation, three legacy validations,
checkpoint I/O, visualizations, and W&B synchronization.

| Step | Validation loss | PPL |
| ---: | ---: | ---: |
| 1,272 | 2.7252 | 15.259 |
| 2,544 | 2.3166 | 10.142 |
| 3,815 | **2.1151** | **8.290** |

Final out-of-process capability results:

| Metric | Result |
| --- | ---: |
| Reasoning S600 exact | 21.17% |
| Teacher-forced token accuracy | 74.45% |
| Public S600 average | 32.83% |
| PIQA | 44.0% |
| HellaSwag | 28.0% |
| ARC-Easy | 26.5% |

Final validation routing remained broad with 0.9536 normalized block entropy
and 0.9922 path diversity. On reasoning S600, however, normalized block-load
entropy fell to 0.5947 while mean route length was 13.97. This is a real
distribution-specific concentration signal to track, but not a single-block or
single-path collapse.

At 250M, the result is weaker than the historical FB C128-U4 point (PPL 7.994,
37.67% reasoning exact, 83.06% teacher accuracy, and 33.33% public average).
That comparison is not architecture-identifying: it changes cache layout,
visibility, completion stride, and gradient horizon together. Historical C512
also showed poor early reasoning despite matched PPL, while the completed
shared DP-C2048 5B run recovered strong reasoning. Q8 is therefore accepted as
an execution and learning-signal pilot, not evidence against strict per-head
cache.

## 8. 5B Follow-Up

The parameter-matched 5B follow-up uses the same model and execution path:

```text
configs/train/q9_cpbc_r125_5b_dp_u1_c2048_per_head_d32_ddp2_legacyval.yaml
```

It retains the matched shared DP-C2048 global batch, data, optimizer, routing
schedule, validation split, and 5B token budget. In-process capability suites
remain disabled. Model-only checkpoints are retained at 15k, 30k, 45k, 60k,
75k, and final; `checkpoint_latest` continues to preserve resumable optimizer
state every 5k. At Q8 steady throughput, pure 5B training is approximately 8.3
hours and end-to-end runtime is expected near 9 hours.

## 9. 5B Launch Status

Q9 started on GPU 0-1 at `2026-07-19 17:23 JST` in tmux session
`brian_q9_per_head_dp_c2048_5b_g01`:

```bash
bash scripts/run_q9_per_head_dp_c2048_5b.sh
```

W&B run: `mio_nora/brian-sphere-llm/glshdyh4`

Startup acceptance passed step 170 with approximately 164k token/s sustained
throughput, 0.399 seconds per optimizer step, and 74,279 MiB peak allocation per
rank. Loss remained finite, all 135 expected parameters participated in DDP,
and no OOM, NCCL, or data/config mismatch was observed. At this measured rate,
pure training is approximately 8.5 hours; legacy validation, checkpoint I/O,
and synchronization put the expected end-to-end runtime near 9 hours.

Capability benchmarks remain intentionally outside the active DDP process.
After training exits, reasoning S600 and public S600 will be evaluated against
the retained 15k/30k/45k/60k/75k/final model-only checkpoints so benchmark
latency cannot leave the second rank blocked in a collective operation.

Q9 subsequently completed the full 5B budget and all six out-of-process
benchmark boundaries. Its final PPL is 3.9022; checkpoint 75k reaches 79.50%
reasoning exact and 40.67% public S600. The complete matched comparison,
systems cost, and revised role of the 250M pilot are in
[q9_per_head_dp_c2048_5b_report.md](./q9_per_head_dp_c2048_5b_report.md).
