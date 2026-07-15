# BDRE Stateful TBPTT Implementation Report

**Status:** exact-forward implementation validated; rejected as a formal R125
training backend because token-serial throughput is not viable

**Date:** 2026-07-16

**Branch:** `bdre-stateful-tbptt`

## 1. Scope

This work adds a single-GPU training backend for exact token-serial BDRE
forward execution with stateful truncated backpropagation through time
(TBPTT). It is not chunked prefill and does not process multiple positions of
one sequence in parallel.

Parallel work comes from the batch dimension. At each token position, all
sequences in the local batch route together and active samples are grouped by
their selected free block. The configured R125 target is one GPU with
`batch_size: 32`.

## 2. Training Contract

For a sequence of length `T` and TBPTT chunk size `C`:

1. Process tokens serially from `0` through `C - 1` for every batch item.
2. Compute next-token losses, including the label that crosses the chunk
   boundary.
3. Backpropagate that chunk loss.
4. Preserve all pre/post and BDRE KV values, but detach their autograd history.
5. Continue with the next token chunk.
6. Execute `optimizer.step()` only after the complete sequence and configured
   microbatch accumulation have finished.

The persistent cache values and forward logits are unchanged by the chunk
boundary. The intentional approximation is gradient truncation: losses in a
later chunk cannot update K/V writer computations from an earlier chunk.

The LM loss uses summed token cross-entropy divided by the number of valid
next-token labels in the complete sequence. Summing chunk losses therefore
matches the original full-sequence mean loss. Existing routing and geometry
auxiliary losses are applied once, on the final sequence token, matching the
existing exact BDRE objective.

## 3. Additive Configuration

The existing exact model and train configs remain unchanged. The new entrypoints
are:

```text
configs/model/brian_r125_bdre_rckv_v1_tbptt.yaml
configs/train/bdre_rckv_r125_5b_tbptt_bs32_legacyval.yaml
configs/model/brian_tiny_bdre_rckv_tbptt.yaml
configs/train/stage5_bdre_tiny_tbptt_debug.yaml
```

The formal train settings are:

```yaml
batch_size: 32
gradient_accumulation_steps: 1
stateful_tbptt:
  enabled: true
  chunk_size: 8
```

`chunk_size: 8` is the initial calibration value, not a silently adaptive
default. Changing it changes the gradient horizon, so an OOM retry must restart
the optimizer step with an explicitly selected smaller value.

## 4. Exact Speed Optimizations

The following changes preserve selected actions, forward values, and cache
contents:

- pre/post attention state now stores contiguous K/V tensors rather than one
  Python tuple entry per token;
- BDRE block, optional depth, and writer histories now use a contiguous token
  dimension, eliminating repeated `torch.stack` reconstruction on every cache
  read;
- current-token writer prefixes are stacked once per route step rather than
  once per active block;
- `has_writer` is updated incrementally instead of reducing all previous writer
  masks at every route step;
- `grouped_host` dispatch copies selected top-1 actions to the host once per
  route step, then builds all active block groups there. This replaces up to
  eight separate Python `torch.any()` CUDA synchronization decisions.

The original `legacy_cuda_scan` dispatch remains available and is still the
default for old model configs. Only the new optimized model config enables
`grouped_host`.

## 5. Safety Boundaries

- Existing checkpoint parameters and state dictionaries are unchanged.
- Incremental cache objects are runtime-only and are not checkpoint payloads.
- Existing exact `forward()` and evaluation continue to use full token-serial
  semantics without gradient truncation.
- Stateful TBPTT currently rejects distributed execution. Dynamic routed DDP
  requires separate treatment of parameters used only in non-final `no_sync`
  chunks.
- The optimizer is never updated while a sequence still depends on KV values
  produced by the current parameter version.

## 6. Validation

| Check | Result |
| --- | --- |
| Exact full forward vs stream chunks | maximum logits difference `0.0` |
| Exact loss vs summed chunk losses | within `3e-6` |
| Cache values survive chunk detach | passed |
| Two independent chunk backwards without `retain_graph` | passed |
| `grouped_host` vs `legacy_cuda_scan` | passed at `atol=rtol=1e-6` |
| BDRE unit/integration suite | 20 passed |
| CUDA BF16 forward/backward | passed |
| Tiny three-step train/eval/checkpoint | passed on B200 GPU 4 |
| Tiny TBPTT peak allocated memory | stable at approximately 28 MiB |

An attempted R125 `BS=32`, `C=8`, 256-token calibration on GPU 4 was discarded
because an unrelated root-owned vLLM job loaded approximately 170 GiB on the
same GPU during the measurement. No timing or memory conclusion is taken from
that run.

An uncontended B200 calibration was completed on GPU 0 with the formal R125
model, BF16, `batch_size=32`, and `chunk_size=8`. The benchmark includes the
complete forward and backward for one random-token microbatch; optimizer update
overhead is excluded.

| Sequence | Tokens | Wall time | Throughput | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4,096 | 145.37 s | 28.18 tok/s | 3.28 GiB | 3.59 GiB |
| 256 | 8,192 | 320.93 s | 25.53 tok/s | 4.46 GiB | 4.86 GiB |

The raw reports are:

```text
reports/bdre_tbptt_bs32_chunk8_seq128_calibration.json
reports/bdre_tbptt_bs32_chunk8_seq256_calibration.json
```

Observed SM utilization was bursty and generally below 40%, while memory was
far from the 183 GiB B200 limit. The bottleneck is therefore token/route-step
control flow and many small kernels, compounded by attention over a growing
history, rather than memory capacity.

## 7. Throughput Decision

A two-point model `time(T) = aT + bT^2`, fitted only to the uncontended 128 and
256 token measurements, projects a 2048-token step at approximately 5,950
seconds (99 minutes), 11 tok/s, and about 21 GiB peak allocated memory. This is
an extrapolation, not a completed 2048-token measurement, and route behavior can
move the exact value. It is sufficient for the engineering decision: even the
more favorable measured 25.53 tok/s would require more than six years to consume
5B training tokens on one GPU.

The old full-autograd BDRE DDP4 attempt consumed approximately 97 GiB per rank
and produced no recorded optimizer step after about 54 minutes. Stateful TBPTT
therefore fixes the memory problem, but it does not fix the fundamental
throughput problem. Increasing batch size or changing the truncation boundary
cannot recover the several orders of magnitude needed for formal training.

Exact token-by-token execution remains the semantic oracle for correctness and
incremental inference. Formal BDRE training is blocked on a parallel prefill or
parallel training formulation with explicit approximation/equivalence tests.
The subsequent synchronous-prefix implementation and B200 calibration are
documented in `reports/bdre_synchronous_prefix_prefill_report.md`.

## 8. Commands

Tiny end-to-end validation:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. python scripts/train.py \
  --config configs/train/stage5_bdre_tiny_tbptt_debug.yaml
```

Short-sequence memory and throughput calibration:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/bdre_rckv_r125_5b_tbptt_bs32_legacyval.yaml \
  --chunk-size 8 \
  --output reports/bdre_tbptt_bs32_chunk8_calibration.json
```

The R125 stateful-TBPTT config is retained as a correctness and profiling
reference. Do not use it for a formal 5B run unless a later implementation
changes the token-serial throughput result above.
