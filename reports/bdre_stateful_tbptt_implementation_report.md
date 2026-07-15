# BDRE Stateful TBPTT Implementation Report

**Status:** implementation and tiny end-to-end validation complete; full R125
memory calibration pending an uncontended B200

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

## 7. Remaining Acceptance

Before starting the formal 5B run, use one uncontended B200 to execute one full
2048-token optimizer step and record:

```text
peak allocated and reserved CUDA memory
tokens per second
route-step host synchronization time
BDRE compile time fraction
pre/post attention time
final cache memory
```

Test `C=8` first. If it OOMs, restart the calibration with `C=4`; do not change
the chunk size inside a running optimizer step. The largest value with a useful
memory margin should be fixed in the formal config and report.

## 8. Commands

Tiny end-to-end validation:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. python scripts/train.py \
  --config configs/train/stage5_bdre_tiny_tbptt_debug.yaml
```

Single-step memory and throughput calibration:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/bdre_rckv_r125_5b_tbptt_bs32_legacyval.yaml \
  --chunk-size 8 \
  --output reports/bdre_tbptt_bs32_chunk8_calibration.json
```

Formal single-GPU training after memory calibration:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. python scripts/train.py \
  --config configs/train/bdre_rckv_r125_5b_tbptt_bs32_legacyval.yaml
```
