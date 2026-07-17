# RC-KV Inference Acceleration Report

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**Checkpoint:** Triton CPBC-DP/C2048/U1 5B, step 75,000

**Hardware:** one NVIDIA B200, FP32 evaluation

## 1. Scope

This change accelerates benchmark inference without changing model weights,
RC-KV memory semantics, route constraints, or the legacy benchmark datasets.
The original evaluators remain available as reference paths. Faster paths are
explicitly selected by evaluation configuration so strict checkpoint selection
and throughput-oriented capability screening cannot be confused.

The original reasoning generator recomputed the complete prefix for every new
token. Public multiple-choice scoring also submitted every choice as an
independent batch-of-one forward. These evaluator choices left the B200 mostly
idle even though the training path was already GPU efficient.

## 2. Implemented Paths

### 2.1 Batched incremental generation

Reasoning samples are grouped by exact `(prompt_length, answer_length)` and
generated in batches. RC-KV models prefill each prompt once through
`forward_stream_chunk`, retain the returned incremental state, and advance each
new token through `forward_incremental`. The model no longer reconstructs the
entire prefix at every decode position.

Grouping uses exact lengths and introduces no padding. Samples that would cross
the context boundary fall back to the sliding-window reference generator.

### 2.2 Exact-length teacher and choice batching

Teacher-forced reasoning sequences and public benchmark choices can be grouped
by exact full-sequence length. Each group is evaluated without padding, and
answer-token logits are selected with tensor masks. Full-vocabulary softmax is
computed only for scored answer positions.

### 2.3 Optional routing diagnostics

`forward_incremental` and the evaluation wrappers now accept
`summarize_routing`. Fast inner loops disable expensive BDRE diagnostic
reductions. Strict batch-of-one public evaluation deliberately keeps summaries
enabled because disabling them selects an incremental compiler path that is
slower for these short sequences.

## 3. Evaluation Profiles

| Profile | Generation | Teacher/public scoring | Intended use |
| --- | --- | --- | --- |
| Legacy reference | full-prefix recomputation | batch 1 | historical reproduction |
| Strict incremental | batched RC-KV incremental | reference teacher; legacy public | checkpoint selection with reference routing metrics |
| Fast capability | batched RC-KV incremental | exact-length batch 64 | rapid capability screening |

Configuration entrypoints:

```text
configs/eval/reasoning_eval_s600.yaml
configs/eval/reasoning_eval_s600_incremental.yaml
configs/eval/reasoning_eval_s600_fast.yaml
configs/eval/public_benchmark_s600.yaml
configs/eval/public_benchmark_s600_fast.yaml
```

Example fast evaluations:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/eval.py \
  --config configs/eval/reasoning_eval_s600_fast.yaml \
  --run <run_dir> --checkpoint checkpoint_step_00075000

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/public_benchmark.py \
  --config configs/eval/public_benchmark_s600_fast.yaml \
  --run <run_dir> --checkpoint checkpoint_step_00075000 \
  --output <output.json>
```

The formal Triton training configuration uses strict incremental reasoning and
legacy public scoring for checkpoint and post-training benchmarks.

## 4. B200 Results

### 4.1 Reasoning S600

| Mode | Batch | End-to-end time | Speedup | Core-result mismatch | Route-path note |
| --- | ---: | ---: | ---: | ---: | --- |
| Legacy reference | 1 | 670.51 s | 1.00x | reference | reference |
| Strict incremental | 16 | 402.41 s | 1.67x | 0/600 | selected routing metrics 0/600 |
| Fast capability | 16 | 242.52 s | 2.76x | 0/600 | numeric batching |
| Fast capability | 32 | 171.11 s | 3.92x | 0/600 | 63/600 route-step summaries differ |
| Fast capability | 64 | **150.10 s** | **4.47x** | **0/600** | 63/600 route-step summaries differ |

At batch 64, measured evaluator time is 139.03 s: 124.17 s generation and
14.86 s teacher scoring. End-to-end accuracy remains exactly the reference
value: reasoning exact `0.821667` and teacher accuracy `0.948431`. Generation
outputs match the reference for all 600 samples.

The fast teacher path is capability-equivalent in this test, but it is not a
bitwise routing audit. Batched GEMM changes floating-point reduction order near
router decision boundaries. Strict mode therefore retains batch-of-one teacher
for model-selection routing telemetry.

### 4.2 Public S600

| Mode | Batch | End-to-end time | Speedup | Prediction mismatch |
| --- | ---: | ---: | ---: | ---: |
| Legacy reference | 1 | 195.19 s | 1.00x | reference |
| Exact-length batch | 16 | 101.86 s | 1.92x | 0/600 |
| Exact-length batch | 32 | 82.15 s | 2.38x | 0/600 |
| Exact-length batch | 64 | **74.21 s** | **2.63x** | **0/600** |

At batch 64, model scoring consumes 60.17 s. Overall accuracy remains `0.390`
with PIQA `0.570`, HellaSwag `0.265`, and ARC-Easy `0.335`.

The 600 predictions are unchanged, but choice scores are not bitwise identical:
across 2,000 choices the mean absolute difference is approximately `5.17e-5`
and the maximum is `0.03711`. The fast public profile is therefore optional;
legacy public scoring remains the formal checkpoint-selection path.

### 4.3 Utilization

The batch-64 paths sustain approximately 89-90% GPU utilization and use roughly
48-50 GiB, compared with the sparse utilization of the original batch-of-one
evaluator. Exact-length bucket structure limits further batching: public S600
contains 149 distinct lengths, while reasoning generation contains 51 exact
shape groups. Raising the cap above 64 has little remaining scheduling benefit.

## 5. Rejected Precision Shortcuts

BF16 and TF32 evaluator variants were tested and removed. On reasoning S30,
BF16 was slower than FP32 after warmup and changed 2/30 core results. TF32 also
changed 2/30 core results and teacher accuracy without providing a useful speed
gain. FP32 remains the accepted inference precision.

## 6. Correctness and Regression Coverage

The accepted implementation checks:

- batched incremental generation groups only equal shapes and reuses state;
- generation outputs and capability metrics match the 600-sample reference;
- exact-length public batching matches reference scoring on a deterministic
  test model;
- strict public scoring requests the reference routing-summary path;
- CLI and YAML profiles expose the execution mode explicitly;
- synchronous-prefix incremental and stream equivalence tests continue to pass.

## 7. Remaining Bottleneck

At batch 64, generation is 89% of measured reasoning evaluation time. More
teacher batching cannot materially improve the result. The next meaningful
exact-preserving work is a dedicated one-token RC-KV decode path with persistent
GPU state and fewer Python/dispatch boundaries, followed by persistent evaluator
workers or CUDA graph capture where shapes are stable. These are separate kernel
engineering tasks and are not implied by the current fast profile.

## 8. Conclusion

The evaluator is no longer dominated by avoidable full-prefix recomputation or
batch-of-one scoring. Strict reasoning improves by 1.67x with reference metrics;
the explicit fast profiles improve reasoning by 4.47x and public scoring by
2.63x while preserving all reported capability predictions. Numerical routing
and score caveats are recorded rather than hidden, and legacy paths remain the
authority for strict checkpoint selection.
