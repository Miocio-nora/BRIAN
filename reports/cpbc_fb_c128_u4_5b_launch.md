# CPBC-FB C128-U4 Formal 5B Launch

**Date:** 2026-07-18 to 2026-07-19

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
DDP timeout:      1,800 seconds
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
- Incremental reasoning S600 and public S600: every 15,000 updates and at completion.
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

## 6. Step-15k Interruption And Recovery

Training reached step 15,000 with a complete resumable checkpoint before the
first capability suite. Validation loss was 1.63799 (PPL 5.14483), reasoning
exact was 67.67%, teacher-forced token accuracy was 89.85%, and public S600 was
37.17%. The public breakdown was PIQA 53.0%, HellaSwag 25.5%, and ARC-Easy
33.0%.

The process then exited because the inherited FB benchmark entry still used
the legacy serial reasoning evaluator. Reasoning completed successfully in
573.8 seconds, but rank 1 had already been waiting at the post-benchmark NCCL
barrier; validation, checkpoint I/O, and benchmark startup pushed the total
wait beyond NCCL's default 600-second collective timeout. This was an
orchestration timeout, not OOM, numerical divergence, routing collapse, or a
corrupt checkpoint.

Recovery made two operational changes without changing the model, optimizer,
data order, or training hyperparameters:

1. The optimized FB benchmark contract now uses
   `reasoning_eval_s600_incremental.yaml`, matching the accepted DP 5B entry.
2. The formal DDP process group has an explicit 1,800-second timeout so a
   legitimate rank-0-only capability suite cannot be mistaken for a deadlock.

The missing public result was recovered directly from
`checkpoint_step_00015000` in 196.1 seconds and appended to the checkpoint
benchmark log. `checkpoint_latest` and both rank-state files identify step
15,000; the checkpoint contains model, optimizer, RNG, sampler epoch, and
microbatch position state required to continue at step 15,001.

The run resumed at `2026-07-18 10:34 JST` under the same W&B ID. The resume
event confirms optimizer, RNG, rank-state, epoch, and microbatch-position
restoration. The training log advances directly from step 15,000 to 15,001;
initial post-resume acceptance reached step 15,013 at a 41.4k token/s median,
with finite loss, approximately 0.99 normalized block entropy, and 16/16
distinct monitored paths.

## 7. Step-75k Boundary And Driver Recovery

The resumed run completed update 75,000 of 76,294, or 98.3% of the nominal 5B
budget. Validation, `checkpoint_latest`, and the retained model-only
`checkpoint_step_00075000` were written successfully. Only 1,294 updates
remained.

The process then failed in benchmark orchestration rather than model compute.
Rank 1 entered the post-benchmark NCCL barrier while rank 0 performed
validation, checkpoint I/O, and the rank-0-only capability suite. The wait
exceeded the configured 1,800-second process-group timeout. NCCL terminated
both ranks while CUDA teardown was active, leaving them blocked in the NVIDIA
UVM write lock and making `nvidia-smi` itself unresponsive until the node was
rebooted.

After reboot, reasoning S600 and public S600 were rerun directly from the
retained 75k checkpoint as independent single-GPU processes. No DDP process
group or training resume was involved. Both suites completed normally. Future
training jobs must not keep a rank in an active NCCL barrier while another rank
runs an unbounded external benchmark; checkpoint capability evaluation should
be an out-of-process post-checkpoint job.

## 8. Capability Matrix And Decision

| Step | Val loss | PPL | Reason exact | Teacher acc | Public avg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 1.6380 | 5.1448 | 0.6767 | 0.8985 | 0.3717 |
| 30,000 | 1.5120 | 4.5356 | 0.7783 | 0.9187 | 0.3617 |
| 45,000 | 1.4505 | 4.2652 | 0.7533 | 0.9292 | 0.3700 |
| 60,000 | 1.4062 | 4.0804 | 0.7900 | 0.9349 | 0.3883 |
| 75,000 | **1.3780** | **3.9671** | **0.8167** | **0.9400** | **0.3933** |

The 75k public result consists of 59.5% PIQA, 26.0% HellaSwag, and 32.5%
ARC-Easy. Routing remained active: validation normalized block entropy was
0.9890, path diversity was 0.9766, and mean route length was 13.125. The run
therefore did not end in numerical or aggregate routing collapse.

At the aligned 75k checkpoint:

| Model | Val loss | PPL | Reason exact | Teacher acc | Public avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched Transformer baseline | **1.3737** | **3.9499** | **0.8767** | **0.9668** | 0.3883 |
| CPBC-DP C2048-U1 | 1.3791 | 3.9713 | 0.8217 | 0.9484 | 0.3900 |
| CPBC-FB C128-U4 | 1.3780 | 3.9671 | 0.8167 | 0.9400 | **0.3933** |

FB is effectively tied with DP on validation and the 600-example public suite,
but is 0.50 percentage points lower on reasoning exact and 0.84 points lower
on teacher accuracy. Both routed variants remain materially behind the matched
baseline on reasoning. The small public differences are only a few examples
and are not evidence of an FB capability advantage.

Continuing the final 1.7% of updates is not justified. The 75k checkpoint is
sufficient for this comparison, and FB C128-U4 does not provide a quality gain
that pays for its much lower training throughput. Further cache-layout,
cache-dimension, and routing ablations should use the substantially faster
CPBC-DP C2048 path first. FB remains a semantic reference for experiments that
specifically require full-bank visibility, not the default development
platform.
