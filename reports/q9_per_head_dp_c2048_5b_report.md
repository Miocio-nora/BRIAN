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

## 6. Routing and Systems Cost

Validation routing remains broad, but reasoning routing is more concentrated
than in the shared-cache run. The per-head reasoning block entropy rises from
0.6896 at 15k to 0.8866 at final, compared with 0.9499 for shared cache at
final. This is not path collapse, but it is a domain-specific concentration
signal that may contribute to the uneven capability profile.

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
