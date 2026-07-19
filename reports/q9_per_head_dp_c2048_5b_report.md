# Q9 Strict Per-Head DP-C2048 5B Result

**Date:** 2026-07-19 to 2026-07-20

**Branch:** `rc-kv-per-head-cache`

**W&B:** `mio_nora/brian-sphere-llm/glshdyh4`

## 1. Run Contract

Q9 isolates the strict per-head cache layout on the accepted CPBC-DP C2048-U1
path. It uses eight free-route blocks, at most 16 route steps, BF16, global
batch 32 on two B200 GPUs, the balanced 5B corpus, and the legacy validation
contract. The per-head key and value cache dimensions are both 32. The prefix
compiler uses the tested recompute implementation, which is mathematically
equivalent to the shared run's incremental exact compiler.

The strict per-head model has 140,308,105 parameters, exactly matching the
shared-cache RC-KV model. The plain Transformer baseline has 137,841,408
parameters, so both RC-KV models are 1.79% larger than the baseline.

## 2. Training Completion

Q9 completed all 76,294 optimizer steps and 5,000,003,584 training tokens. It
started at `2026-07-19 17:23 JST` and finished at `2026-07-20 01:56 JST`.

| Measurement | Result |
| --- | ---: |
| End-to-end runtime | 8 h 32 min 20 s |
| Median logged global throughput | 170,711 token/s |
| Peak allocation per rank | 74,278 MiB |
| Final validation loss | **1.3615** |
| Final validation PPL | **3.9022** |
| Final validation block entropy | 0.9874 |
| Final validation path diversity | 0.9922 |
| Missing DDP gradient parameters | 0 |

No OOM, non-finite loss, NCCL failure, or route collapse occurred. Six
model-only checkpoints were retained at 15k, 30k, 45k, 60k, 75k, and final.

## 3. Checkpoint Matrix

All capability values use the fixed reasoning S600 and public S600 contracts.
The public suite contains 200 PIQA, 200 HellaSwag, and 200 ARC-Easy examples.

| Step | Val loss | PPL | Reason exact | Teacher acc | Public avg | Reason block entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 1.6444 | 5.1778 | 67.67% | 89.04% | 36.67% | 0.6896 |
| 30,000 | 1.4970 | 4.4683 | 74.17% | 92.17% | 37.17% | 0.8222 |
| 45,000 | 1.4319 | 4.1866 | 71.33% | 92.65% | 38.17% | 0.8712 |
| 60,000 | 1.3943 | 4.0321 | 76.67% | 92.75% | 38.67% | 0.8748 |
| 75,000 | 1.3636 | 3.9104 | **79.50%** | 93.50% | **40.67%** | 0.8737 |
| 76,294 | **1.3615** | **3.9022** | 77.83% | **94.64%** | 39.50% | **0.8866** |

The 75k checkpoint is the best capability checkpoint because it leads both
reasoning exact and public average. The final checkpoint is best for language
modeling and teacher-forced token prediction, but both exact generation and
public accuracy regress after 75k.

At the 75k public peak, PIQA is 57.5%, HellaSwag is 30.0%, and ARC-Easy is
34.5%. At final they are 55.5%, 26.5%, and 36.5%, respectively.

## 4. Matched Comparisons

The following table reports final PPL and the best capability value over the
same six checkpoint boundaries. Capability peaks may occur at different steps.

| Model | Parameters | Final PPL | Best reason exact | Best teacher acc | Best public avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain Transformer | 137.84M | 3.9491 | **87.67%** | **96.68%** | 39.00% |
| Shared-cache RC-KV | 140.31M | 3.9535 | 82.17% | 94.84% | 40.17% |
| Strict per-head RC-KV | 140.31M | **3.9022** | 79.50% | 94.64% | **40.67%** |

Against shared-cache RC-KV, strict per-head improves final PPL by 0.0513
(1.30%) and best public by 0.50 percentage points, but trails best reasoning by
2.67 points and best teacher accuracy by 0.20 points. The public difference is
only three examples out of 600 and is therefore directional, not statistically
resolved.

Against the plain Transformer, strict per-head improves final PPL by 0.0469
(1.19%) and best public by 1.67 points, but trails best reasoning by 8.17 points
and teacher accuracy by 2.04 points. This is not an overall capability win over
the baseline.

## 5. Capability Shape

### 5.1 Plain Baseline Versus Latest Per-Head DP

Each cell below is `plain baseline / strict per-head DP`, evaluated on exactly
the same generated samples and checkpoint step:

| Step | Overall | Arithmetic | Copy | Reverse | Rewrite |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 73.50 / 67.67 | 38.67 / 17.33 | 98.67 / 97.33 | 91.33 / **98.67** | 65.33 / 57.33 |
| 30,000 | 77.67 / 74.17 | 50.67 / 27.33 | 96.00 / 89.33 | 87.33 / **93.33** | 76.67 / **86.67** |
| 45,000 | 80.50 / 71.33 | 52.67 / 36.00 | 98.00 / 84.00 | 88.00 / 85.33 | 83.33 / 80.00 |
| 60,000 | 84.83 / 76.67 | 56.67 / 33.33 | 100.00 / 96.67 | 94.67 / **95.33** | 88.00 / 81.33 |
| 75,000 | 87.67 / 79.50 | 69.33 / 40.00 | 100.00 / 97.33 | 99.33 / 93.33 | 82.00 / **87.33** |
| 76,294 | 84.00 / 77.83 | 66.00 / 49.33 | 91.33 / **100.00** | 94.00 / 72.67 | 84.67 / **89.33** |

The plain baseline wins overall reasoning at every boundary. The per-head DP
model sometimes wins an individual family--early reverse, several rewrite
boundaries, and final copy--but never converts those gains into a higher
aggregate exact score. Arithmetic is the persistent deficit: per-head DP trails
the baseline by 16.67 to 29.33 percentage points at every checkpoint.

This gap is systematic rather than an unfavorable PPL checkpoint. From 30k
onward per-head DP has lower validation PPL than the baseline, while reasoning
exact remains 3.50 to 9.17 points lower:

| Step | Baseline PPL | Per-head DP PPL | PPL delta | Baseline reason | Per-head DP reason | Reason delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | **5.0902** | 5.1778 | +0.0876 | **73.50%** | 67.67% | -5.83 pp |
| 30,000 | 4.4897 | **4.4683** | -0.0214 | **77.67%** | 74.17% | -3.50 pp |
| 45,000 | 4.2283 | **4.1866** | -0.0417 | **80.50%** | 71.33% | -9.17 pp |
| 60,000 | 4.0682 | **4.0321** | -0.0361 | **84.83%** | 76.67% | -8.17 pp |
| 75,000 | 3.9499 | **3.9104** | -0.0395 | **87.67%** | 79.50% | -8.17 pp |
| 76,294 | 3.9491 | **3.9022** | -0.0469 | **84.00%** | 77.83% | -6.17 pp |

Teacher-forced results support the same distinction. At final, per-head DP
beats the baseline on copy and rewrite token accuracy, but arithmetic remains
82.06% versus 87.61%. The arithmetic gap is therefore a real next-token task
gap, not only autoregressive exact-match amplification. In reverse, by contrast,
the final teacher gap is only 1.15 points while exact is 21.33 points lower, so
long-sequence error compounding explains most of that deficit.

### 5.2 Evolution Within Per-Head DP

The overall reasoning curve hides strong movement between task families:

| Step | Overall | Arithmetic | Copy | Reverse | Rewrite |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 67.67% | 17.33% | 97.33% | **98.67%** | 57.33% |
| 30,000 | 74.17% | 27.33% | 89.33% | 93.33% | 86.67% |
| 45,000 | 71.33% | 36.00% | 84.00% | 85.33% | 80.00% |
| 60,000 | 76.67% | 33.33% | 96.67% | **95.33%** | 81.33% |
| 75,000 | **79.50%** | 40.00% | 97.33% | 93.33% | 87.33% |
| 76,294 | 77.83% | **49.33%** | **100.00%** | 72.67% | **89.33%** |

From 75k to final, arithmetic gains 14 correct samples, copy gains four, and
rewrite gains three. Reverse alone loses 31 samples, producing the net
10-sample overall regression. The final checkpoint is therefore not globally
weaker; it shifts capability away from reverse while improving every other
family.

Difficulty aggregates are also non-monotonic:

| Step | Easy | Medium | Hard |
| ---: | ---: | ---: | ---: |
| 15,000 | 69.00% | 71.50% | 62.50% |
| 30,000 | 77.00% | 74.50% | 71.00% |
| 45,000 | 79.50% | 72.00% | 62.50% |
| 60,000 | 89.00% | 65.00% | **76.00%** |
| 75,000 | 94.00% | **77.50%** | 67.00% |
| 76,294 | **97.00%** | 74.00% | 62.50% |

Easy performance improves almost monotonically, while medium and hard
performance oscillate. In arithmetic specifically, easy rises from 30% to
94%, but final medium and hard remain only 34% and 20%. The run learns the
simple template reliably without establishing equally stable long-composition
behavior.

The final reverse regression is length-sensitive:

| Reverse subset | 75k exact | Final exact | 75k teacher | Final teacher | Answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 100% | 98% | 100% | 99.75% | 8 |
| Medium | 100% | 76% | 100% | 98.50% | 16 |
| Hard | 80% | 44% | 99.38% | 97.69% | 32 |

For reverse-hard, the sequence-level change is almost fully explained by
token-error compounding: `0.99375^32 = 81.8%` and `0.976875^32 = 47.3%`, close
to the observed 80% and 44% exact rates. A 1.69-point teacher-token regression
therefore becomes a 36-point exact regression. This is exposure sensitivity,
not evidence of a global reasoning collapse.

Q9 also keeps the learning rate at `3e-4` through the final update. The last
1,294 steps are therefore still capable of moving task-specific behavior
materially instead of gently converging under a decay tail. This is a plausible
source of the late family rebalancing, although the current checkpoints do not
identify causality.

### 5.3 Shared-DP Attribution

At the selected 75k checkpoint, the reasoning-family comparison against the
shared-cache model is not uniformly worse:

| Family | Shared cache | Strict per-head | Difference |
| --- | ---: | ---: | ---: |
| Arithmetic | 48.67% | 40.00% | -8.67 pp |
| Copy | 100.00% | 97.33% | -2.67 pp |
| Reverse | 84.00% | **93.33%** | +9.33 pp |
| Rewrite | **96.00%** | 87.33% | -8.67 pp |

Strict head isolation therefore changes the learned capability profile rather
than producing a uniform gain. It strongly favors reverse at this checkpoint
while losing arithmetic and rewrite exactness. Teacher accuracy continuing to
rise as exact generation fluctuates also indicates sequence-level error
propagation that PPL cannot expose.

The three-way 75k comparison identifies which differences predate per-head
cache:

| Family | Plain baseline | Shared DP | Per-head DP |
| --- | ---: | ---: | ---: |
| Arithmetic | **69.33%** | 48.67% | 40.00% |
| Copy | **100.00%** | **100.00%** | 97.33% |
| Reverse | **99.33%** | 84.00% | 93.33% |
| Rewrite | 82.00% | **96.00%** | 87.33% |

Most of the arithmetic deficit is already present in shared DP relative to the
plain Transformer; strict per-head isolation adds another 8.67-point loss.
Shared DP strongly favors rewrite, while per-head gives back 8.67 points of that
gain and recovers 9.33 points of reverse. Per-head is therefore a second-stage
redistribution inside an existing DP/RC-KV capability profile, not the sole
source of the baseline reasoning gap.

## 6. Routing and Systems Cost

Validation routing remains broad, but reasoning routing is more concentrated
than in the shared-cache run. The per-head reasoning block entropy rises from
0.6896 at 15k to 0.8866 at final, compared with 0.9499 for shared cache at
final. This is not path collapse, but it is a domain-specific concentration
signal that may contribute to the uneven capability profile.

For reverse-hard specifically, 75k to final leaves mean route length fixed at
15 and leaves cache key/value weight entropy effectively unchanged
(`1.640/1.835` to `1.633/1.835`). Block entropy declines modestly from 0.9005
to 0.8805 and route entropy from 1.7816 to 1.6859. All checkpoint evaluations
have zero routing noise and zero random-route overrides. The evidence therefore
rules out stochastic evaluation and cache-weight collapse; it shows mild route
concentration, but does not establish that concentration as the cause of the
reverse-token regression.

| System metric | Plain baseline | Shared RC-KV | Strict per-head RC-KV |
| --- | ---: | ---: | ---: |
| Training throughput | 618,738 tok/s | 219,409 tok/s | 170,711 tok/s |
| End-to-end runtime | 2 h 25 min | 6 h 48 min | 8 h 32 min |
| Peak allocation/rank | 32.9 GiB | 48.3 GiB | 72.5 GiB |

Strict per-head is 22.2% slower than shared cache by median throughput, takes
25.7% longer end to end, and uses about 50% more peak allocated memory. It
remains about 3.62x slower than the plain Transformer. The quality signal must
therefore be weighed against a substantial systems cost.

## 7. Decision

The 250M pilot must not be used as an architecture ranking test. It remains
useful for OOM, gradient, numerical, throughput, and catastrophic-collapse
screening, but it did not predict the direction of the 5B PPL and public
results.

Strict per-head d32 is retained as a real scale-dependent RC-KV candidate. It
establishes the best validation PPL and best public S600 point among the current
RC-KV runs at fixed RC-KV parameter count. It is not promoted to the default
because reasoning remains below shared cache and the plain baseline, while
training memory and runtime are materially worse. Use checkpoint 75k for
capability evaluation and final for LM/teacher-forced analysis.

The complete checkpoint table is uploaded to the original W&B run under
`benchmark_backfill/*`. Raw reports remain under the Q9 run directory's
`benchmarks/` folder.
