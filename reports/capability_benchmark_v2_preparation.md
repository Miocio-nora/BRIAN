# Capability Benchmark V2 Preparation

**Date:** 2026-07-20

**Status:** implemented and validated; first three-model 5B matrix completed

## 1. Motivation

The legacy capability contract remains useful for checkpoint tracking, but its
public component contains only 200 PIQA, 200 HellaSwag, and 200 ARC-Easy
examples. A difference of 0.5 percentage points is only three examples, so the
current public result cannot resolve the small gap between shared-cache and
strict per-head RC-KV.

Capability V2 adds two independent layers:

1. a full-split multiple-choice suite that gives the current 140M-scale models
   measurable signal across commonsense, science, reading, mathematics, and
   formal logic;
2. strict generative GSM8K and MATH-500 evaluation for classical mathematical
   reasoning, retained even when current models are near the accuracy floor.

It does not replace or modify `reasoning_eval_s600.yaml` or
`public_benchmark_s600.yaml`. Historical tables continue to use those exact
contracts.

## 2. Expanded Public Suite

Configuration: `configs/eval/public_benchmark_full_v2.yaml`

| Group | Tasks | Labeled examples |
| --- | --- | ---: |
| Legacy core, full split | PIQA validation, HellaSwag validation, ARC-Easy validation | 12,450 |
| Expanded reasoning | ARC-Challenge test, OpenBookQA test, WinoGrande validation, BoolQ validation, CommonsenseQA validation | 7,430 |
| MMLU math/logic | elementary mathematics, high-school mathematics, college mathematics, abstract algebra, formal logic test splits | 974 |
| **Total** | 13 tasks | **20,854** |

The original task adapters and seeded S600 sampling behavior are unchanged.
New datasets are revision-pinned. Every report records the dataset fingerprint,
selected source indexes, selection SHA-256, actual count, and source split in a
single JSON report; no per-example manifest tree is created.

Each candidate answer is scored by continuation log likelihood with the same
length-normalized policy used by the legacy public evaluator. Reports include:

- micro accuracy and 95% Wilson interval;
- macro task accuracy;
- per-task count, accuracy, and confidence interval;
- legacy-core, expanded-reasoning, and MMLU math/logic group summaries;
- one JSONL sample audit file.

The V2 macro score and the old S600 average are different contracts and must not
be placed in the same table column.

## 3. Classic Generative Math Suite

Configuration: `configs/eval/classic_math_reasoning_full.yaml`

| Task | Source split | Examples | Maximum generated tokens | Primary metric |
| --- | --- | ---: | ---: | --- |
| GSM8K | `openai/gsm8k`, test | 1,319 | 128 | Math-Verify exact answer equivalence |
| MATH-500 | `HuggingFaceH4/MATH-500`, test | 500 | 256 | Math-Verify symbolic equivalence |

Both datasets use pinned Hugging Face revisions. Decoding is deterministic
greedy generation with zero-shot chain-of-thought prompting and EOS truncation.
MATH-500 is additionally summarized by subject and difficulty level. Parse rate
is reported separately from accuracy so malformed generation can be separated
from a parsed but incorrect answer.

The implemented prompt is a fixed BRIAN comparison protocol, not a claim of
drop-in comparability with a third-party leaderboard that may use different
few-shot examples, chat templates, stop strings, or generation budgets. The
report records `leaderboard_comparable: false` explicitly. Accuracy remains
strict: no partial-credit score is substituted when a model is near zero.

## 4. Commands

The B200 training environment now includes the CPU-only benchmark grader. For a
fresh editable installation, use:

```bash
python -m pip install -e '.[benchmarks]'
```

Run a bounded evaluator smoke first:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/eval.py \
  --config configs/eval/classic_math_reasoning_smoke.yaml \
  --run <run_dir> \
  --checkpoint checkpoint_step_00075000 \
  --output <run_dir>/classic_math_smoke_step75000.json
```

Run the two full suites:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/eval.py \
  --config configs/eval/public_benchmark_full_v2.yaml \
  --run <run_dir> \
  --checkpoint checkpoint_step_00075000 \
  --output <run_dir>/public_benchmark_full_v2_step75000.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python scripts/eval.py \
  --config configs/eval/classic_math_reasoning_full.yaml \
  --run <run_dir> \
  --checkpoint checkpoint_step_00075000 \
  --output <run_dir>/classic_math_step75000.json
```

## 5. First Evaluation Matrix

Do not evaluate every retained checkpoint initially. The first controlled pass
should use the capability-selected 75k checkpoints for:

1. the matched plain Transformer baseline;
2. shared-cache CPBC-DP C2048-U1;
3. strict per-head d32 CPBC-DP C2048-U1.

Run final checkpoints only after the 75k matrix completes. The existing result
already shows late task redistribution, so 75k and final must remain separate
rows rather than averaging them.

This matrix can resolve whether the current directional public gain extends
beyond three S600 examples and whether the arithmetic deficit persists on
external math tasks. It cannot by itself separate head isolation from cache
capacity; that remains the role of a controlled cache-layout/dimension
ablation.

## 6. Validation

- All 13 expanded public task loaders were exercised against their pinned or
  legacy-preserved datasets. The realized labeled count is 20,854.
- GSM8K and MATH-500 loaders resolve 1,319 and 500 test examples respectively;
  numeric and symbolic-equivalence grader probes passed.
- A fake-model end-to-end test covers report and sample artifact generation.
- The complete repository suite passes: `673 passed`.
- The environment remains `torch 2.11.0+cu128`; Math-Verify is CPU-only and
  does not alter the B200 CUDA package contract.

## 7. First Matrix Outcome

The step-75k baseline, shared-DP d32, and strict per-head d32 matrix completed
on 2026-07-20. Across all 20,854 public examples, strict per-head d32 scores
35.52% versus 35.40% for the baseline (`p=0.649`) and 34.60% for shared DP
(`p=0.00076` for per-head versus shared). Per-head therefore recovers the
shared-cache degradation but does not establish an overall win over the plain
Transformer.

All three models score approximately 1% on the strict GSM8K/MATH-500 suite, so
that suite is retained as a hard-tail guardrail rather than a primary selector
at this scale. The full table and paired analysis are in
[capability_benchmark_v2_5b_results.md](./capability_benchmark_v2_5b_results.md).
