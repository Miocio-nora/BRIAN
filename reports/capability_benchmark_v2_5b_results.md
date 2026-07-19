# Capability Benchmark V2: 5B Checkpoint Results

**Date:** 2026-07-20

**Checkpoint:** step 75,000 for all models

## 1. Evaluation Contract

This pass compares the three completed balanced-data 5B runs at the same
optimizer boundary:

1. the 137.84M-parameter plain Transformer baseline;
2. the 140.31M-parameter shared-cache CPBC-DP C2048-U1 model;
3. the parameter-matched 140.31M strict per-head d32 CPBC-DP C2048-U1 model.

Capability V2 is supplementary to the historical S600 contract. The public
suite scores all 20,854 labeled examples from 13 multiple-choice tasks. The
classic-math suite generates fixed budgets for all 1,319 GSM8K and 500
MATH-500 examples and grades them with Math-Verify 0.9.0. The math prompt is a
project comparison protocol and is explicitly not leaderboard-comparable.

## 2. Combined Result

The legacy columns preserve the original 75k reports. Public V2 micro accuracy
is weighted by all examples; macro accuracy gives each of the 13 tasks equal
weight.

| Model | Params | 75k PPL | Reason S600 | Teacher acc | Public S600 | Public V2 micro | Public V2 macro | GSM8K | MATH-500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain baseline | 137.84M | 3.9499 | **87.67%** | **96.68%** | 38.83% | 35.40% | 31.98% | **1.36%** | 1.00% |
| Shared-DP d32 | 140.31M | 3.9713 | 82.17% | 94.84% | 39.00% | 34.60% | 31.62% | 1.29% | 0.60% |
| Per-head DP d32 | 140.31M | **3.9104** | 79.50% | 93.50% | **40.67%** | **35.52%** | **32.21%** | 1.06% | **1.20%** |

The full public suite changes the interpretation of the 600-sample signal.
Per-head d32 led the baseline by 1.84 points on Public S600, but leads by only
0.12 points on the full suite. The S600 observation was real for those selected
examples, but it is not evidence of a broad public-benchmark win.

## 3. Full Public Suite

### 3.1 Group Results

| Group | Samples | Plain baseline | Shared-DP d32 | Per-head DP d32 |
| --- | ---: | ---: | ---: | ---: |
| Legacy core | 12,450 | 32.21% | 31.97% | **32.37%** |
| Expanded reasoning | 7,430 | 42.48% | 40.66% | **42.56%** |
| MMLU math/logic | 974 | **22.18%** | 22.07% | **22.18%** |
| **All examples** | **20,854** | 35.40% | 34.60% | **35.52%** |

Per-head d32 is nearly identical to the baseline in every aggregate group. The
shared-cache deficit is concentrated in expanded reasoning, especially BoolQ.
All three models remain close to chance across the selected MMLU math/logic
subjects.

### 3.2 Task Results

| Task | Samples | Plain baseline | Shared-DP d32 | Per-head DP d32 |
| --- | ---: | ---: | ---: | ---: |
| PIQA | 1,838 | **59.68%** | 59.47% | 59.41% |
| HellaSwag | 10,042 | 27.01% | 26.77% | **27.29%** |
| ARC-Easy | 570 | **35.26%** | 34.91% | 34.74% |
| ARC-Challenge | 1,172 | 22.01% | 21.76% | **23.29%** |
| OpenBookQA | 500 | 27.60% | 27.00% | **28.20%** |
| WinoGrande | 1,267 | **51.38%** | 51.14% | 51.30% |
| BoolQ | 3,270 | **55.57%** | 50.89% | 54.34% |
| CommonsenseQA | 1,221 | 23.91% | 26.13% | **26.29%** |
| MMLU elementary mathematics | 378 | **21.43%** | 21.16% | 21.16% |
| MMLU high-school mathematics | 270 | 21.11% | 21.11% | 21.11% |
| MMLU college mathematics | 100 | 21.00% | 21.00% | 21.00% |
| MMLU abstract algebra | 100 | 22.00% | 22.00% | 22.00% |
| MMLU formal logic | 126 | 27.78% | 27.78% | **28.57%** |

### 3.3 Paired Inference

McNemar tests use each model's correctness on the same examples. `A only` is
the number answered correctly only by model A; `B only` is the corresponding
count for model B.

| Comparison | Accuracy delta | A only | B only | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| Shared-DP minus baseline | -0.80 pp | 1,571 | 1,737 | 0.00411 |
| Per-head minus baseline | +0.12 pp | 1,524 | 1,498 | 0.64928 |
| Per-head minus shared-DP | +0.92 pp | 1,705 | 1,513 | 0.00076 |

Strict per-head d32 significantly recovers the full-suite degradation of the
shared cache, but it does not significantly outperform the plain baseline.
This is the central V2 result. Individual uncorrected task comparisons should
not be promoted after examining 13 tasks; they are useful for localization,
not independent architecture claims.

## 4. Classic Math

| Model | GSM8K | MATH-500 | Combined | Parse rate | Generation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain baseline | 18/1,319 (1.36%) | 5/500 (1.00%) | 23/1,819 (1.26%) | 99.89% | 481 s |
| Shared-DP d32 | 17/1,319 (1.29%) | 3/500 (0.60%) | 20/1,819 (1.10%) | 73.89% | 3,754 s |
| Per-head DP d32 | 14/1,319 (1.06%) | 6/500 (1.20%) | 20/1,819 (1.10%) | 87.47% | 3,874 s |

The 95% Wilson interval is 0.84-1.89% for the baseline and 0.71-1.69% for each
RC-KV model. Paired exact tests give `p=0.749` for either RC-KV model versus the
baseline and `p=1.0` for per-head versus shared cache. The suite is therefore a
floor/abnormality guardrail at this model scale, not a ranking benchmark.

The parse-rate difference is real behavior under the fixed prompt, but it does
not rescue the ranking value of 20-23 correct examples. Future routine math
evaluation should add easier arithmetic and word-problem tiers while retaining
GSM8K and MATH-500 only as the hard tail.

## 5. Systems Observation

The full public scorer took 37 seconds for the baseline and 193-195 seconds for
the two RC-KV models. Classic-math generation took 481 seconds for the baseline
and 3,754-3,874 seconds for RC-KV. The current exact-length incremental
generator is correct but leaves many prompt-length groups below the configured
batch size. This evaluation confirms that RC-KV inference remains a material
systems limitation independent of training throughput.

## 6. Decision

1. Retain strict per-head cache as the stronger RC-KV implementation: it
   significantly recovers shared-cache public accuracy and improves PPL.
2. Do not claim an overall capability win over the plain Transformer. Public
   V2 is a statistical tie, while synthetic Reason S600 remains lower.
3. Do not use GSM8K/MATH-500 as the primary selector at the present 140M/5B
   scale; all three models are at the floor.
4. The next cache-capacity ablation is strict per-head d64 on the same 5B
   DP-C2048-U1 contract. It tests whether removing the remaining 2x per-head
   compression improves the uneven reasoning profile.

Raw JSON and JSONL artifacts remain under each run's
`capability_v2/step_00075000/` directory and are intentionally not duplicated
in Git.
