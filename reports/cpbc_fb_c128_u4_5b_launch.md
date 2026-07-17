# CPBC-FB C128-U4 Formal 5B Launch

**Date:** 2026-07-18

**Branch:** `rc-kv-triton-reader`

**Hardware:** physical GPU 0-1, two NVIDIA B200 GPUs

## 1. Selection

The controlled 250M-token gradient-horizon ablation selected U4 for the first
formal CPBC-FB run. At matched C128 full-bank semantics, U4 finished with
37.67% reasoning exact versus 27.67%, 16.00%, and 31.83% for U1, U2, and U8.
Public S600 was effectively flat across the arms, so the choice is based on the
combined capability matrix rather than PPL alone.

DDP2 is retained instead of DDP4. Historical matched-batch scaling provides
only about 1.52x throughput from twice as many GPUs, while DDP2 is roughly 32%
more efficient per allocated GPU. The bottleneck remains rank-local recurrent
reader computation rather than memory capacity.

## 2. Formal Contract

```text
config:
  configs/train/cpbc_r125_5b_fb_u4_c128_triton_fused_reader_incremental_ddp2_legacyval.yaml
visibility:       CPBC-FB
chunk size:       128 tokens
detach interval:  4 chunks
gradient horizon: 512 tokens
backend:          Triton fused reader with incremental prefix compilation
world size:       2
local batch:      16 sequences per rank
global batch:     32 sequences
context length:   2048
training budget:  76,294 updates, approximately 5B tokens
data:             balanced R125 5B train data
validation:       legacy validation split
resume:           enabled
```

The U4 file extends the accepted optimized U1 configuration and changes the
gradient detach interval from 1 to 4. It does not inherit the historical
pre-Triton U4 configuration.

## 3. Evaluation And Checkpoints

- Legacy validation and model checkpoint: every 5,000 updates.
- Model-only retained checkpoints: up to 20.
- Reasoning S600 and public S600: every 15,000 updates and at completion.
- W&B: online, project `brian-sphere-llm`.

Checkpoint selection will use validation PPL, reasoning exact, teacher
accuracy, public average, route/cache health, and block entropy together.
Final PPL is not the sole selection rule.

## 4. Expected Runtime

The 250M U4 run sustained approximately 39.6k global tokens/s after warmup.
At that rate, pure 5B training is approximately 35.1 hours. Validation,
checkpoint benchmarks, checkpoint I/O, and startup put the expected
end-to-end duration at approximately 36-38 hours.

## 5. Launch And Monitoring

The formal run started at `2026-07-18 02:27 JST`. Its W&B run is
`mio_nora/brian-sphere-llm/1l8fnxmu`.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_cpbc_fb_c128_u4_5b.sh
```

The detached tmux session is named `brian_fb_c128_u4_5b_g01`. Training metrics
are written to:

```text
runs/cpbc_r125_5b_fb_u4_c128_triton_fused_reader_incremental_ddp2_legacyval/train_log.jsonl
```

The run directory also contains the terminal-dashboard event stream,
checkpoints, evaluation logs, route visualizations, and the W&B run metadata.

Initial acceptance reached step 53 with finite loss, a 37.6k token/s median
over the latest 30 updates, 17.5 GiB peak CUDA allocation per rank, 0.994
normalized block entropy, and 16 distinct paths for the 16 monitored samples.
The observed loss moved from 113.6 on the compile-heavy first update to 41.6
at step 53. No startup routing collapse, DDP error, or W&B upload failure was
observed.
