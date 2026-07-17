# RC-KV Triton DP-C2048 5B Run

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**W&B:** `mio_nora/brian-sphere-llm/30sjr3ng`

## 1. Run Contract

The formal run uses CPBC with Depth-Prefix Visibility (CPBC-DP), chunk size
2,048, and detach horizon `U=1`. The model has eight free-route blocks, a
maximum of 16 route steps, incremental exact prefix compilation, and the
optional Triton fused RC-KV reader. It ran on two B200 GPUs with global batch
32, local batch 16, BF16 training, the balanced 5B training corpus, and the
legacy validation contract. The Triton path changes execution only; it does not
change the RC-KV model definition or routing semantics.

## 2. Training Efficiency

| Measurement | Result |
| --- | ---: |
| Stable median global throughput | 219,409 token/s |
| Typical steady range | 211k-216k token/s |
| Matched baseline throughput | 618,738 token/s |
| Remaining baseline gap | 2.82x |
| First-to-final optimizer step | 6 h 47 min 36 s |
| Peak allocated memory per rank | about 48.3 GiB |
| Observed device memory per rank | about 56.2 GiB |

The first step included about 26.6 seconds of compilation. Periodic validation,
checkpointing, and benchmark launches are excluded from the stable-throughput
median. This run improves the previous accepted C2048 implementation from a
4.10x to a 2.82x gap against the matched baseline.

## 3. Checkpoint Matrix

All checkpoints use the same legacy validation, reasoning S600, and public
S600 contracts. Public S600 is 200 PIQA, 200 HellaSwag, and 200 ARC-Easy
examples.

| Step | Val loss | PPL | Reason exact | Teacher acc | Block entropy | Path diversity | Public avg | PIQA | HellaSwag | ARC-Easy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15,000 | 1.6456 | 5.1844 | 0.7150 | 0.9088 | 0.8155 | 1.0000 | 0.3667 | 0.525 | 0.260 | 0.315 |
| 30,000 | 1.5127 | 4.5388 | 0.7600 | 0.9164 | 0.8256 | 1.0000 | 0.3667 | 0.530 | 0.235 | 0.335 |
| 45,000 | 1.4449 | 4.2414 | 0.7567 | 0.9304 | 0.9196 | 1.0000 | 0.3983 | 0.580 | 0.275 | 0.340 |
| 60,000 | 1.4073 | 4.0850 | 0.7050 | 0.9237 | 0.9443 | 1.0000 | **0.4017** | 0.570 | 0.260 | **0.375** |
| 75,000 | 1.3791 | 3.9713 | **0.8217** | **0.9484** | **0.9552** | 1.0000 | 0.3900 | 0.570 | 0.265 | 0.335 |
| 76,294 | **1.3746** | **3.9535** | 0.7967 | 0.9375 | 0.9499 | 1.0000 | 0.3917 | 0.570 | **0.270** | 0.335 |

Validation loss and PPL improve monotonically, while capability metrics do not.
Step 75,000 is the best balanced checkpoint because it leads both reasoning
metrics and remains close to the best public score. Step 60,000 is the best
public checkpoint. The final checkpoint is best only by validation loss and
PPL, and reasoning regresses after 75,000. Checkpoint selection must therefore
not use final PPL alone.

Normalized block entropy rises from 0.8155 to roughly 0.95, so the run does not
show block-usage collapse. The reported `route_path_diversity=1.0` is not a
useful cross-sample diversity statistic in this evaluator because reasoning
examples are processed one at a time.

## 4. Matched Baseline Comparison

The matched baseline uses the same balanced 5B corpus, global batch 32, two
B200 GPUs, legacy validation, and S600 benchmark definitions. It has 137.84M
parameters versus 140.31M for RC-KV, so RC-KV is 1.79% larger.

| Metric | Baseline | RC-KV | Difference |
| --- | ---: | ---: | ---: |
| Step-75k validation loss | **1.3737** | 1.3791 | +0.0054 |
| Step-75k PPL | **3.9499** | 3.9713 | +0.0214 |
| Step-75k reasoning exact | **0.8767** | 0.8217 | -5.50 pp |
| Step-75k teacher accuracy | **0.9668** | 0.9484 | -1.84 pp |
| Step-75k public average | 0.3883 | **0.3900** | +0.17 pp |
| Best public average | 0.3900 (76,294) | **0.4017 (60k)** | +1.17 pp |
| Matched training throughput | **618,738 token/s** | 219,409 token/s | 2.82x slower |
| First-to-final step wall time | **2 h 25 min** | 6 h 48 min | 2.80x longer |
| Peak allocated memory per rank | **32.9 GiB** | 48.3 GiB | 1.47x higher |

Final validation is effectively tied: RC-KV is worse by 0.0011 loss and 0.0044
PPL. Public S600 is also statistically unresolved at this sample count. At the
best public points, RC-KV leads by seven correct answers out of 600; at aligned
step 60k it leads by eight. In contrast, the reasoning gap is material. At step
75k, the baseline alone solves 71 examples that RC-KV misses, while RC-KV alone
solves 38 examples that the baseline misses.

The current result therefore does not establish an overall capability win over
the fixed Transformer. It establishes near-baseline language modeling and
public-task quality under free routing and shared compressed memory, with a
remaining reasoning, training-efficiency, memory, and inference-efficiency
cost.

## 5. Benchmark Recovery

Legacy validation ran every 5,000 steps and was uploaded under `eval/*`. The
original post-checkpoint reasoning and public subprocesses failed because the
benchmark loader reconstructed the old `BrianRouteCore` whenever
`architecture=brian_route_core`, even when the checkpoint had
`bdre_shared_kv=true`. The resulting state-dict load rejected
`bdre_projections.*` parameters.

Commit `1ad976d` makes the loader construct `BrianBDRERouteCore` with
`BDREConfig` for BDRE checkpoints and adds a regression test. All six target
checkpoints were then evaluated successfully. The recovered history is uploaded
to the original W&B run under `benchmark_backfill/*`, with the complete matrix
in `benchmark_backfill/checkpoint_matrix`. The original failed return codes are
retained as audit history.

## 6. Inference Efficiency Limitation

The current benchmark workers use only about 2.1 GiB and 28-32% of one B200.
This is an evaluator limitation, not evidence that the Triton training path is
idle:

- reasoning evaluates 600 examples serially and greedy generation recomputes
  the complete prefix for every output token;
- public S600 evaluates each example and answer choice separately, also at
  effective batch size one;
- FP32 reference evaluation intentionally falls back to the static Flex reader
  because the custom Triton reader currently targets BF16 training;
- each checkpoint starts a new process, reconstructs the model, and rebuilds
  compilation state.

Observed reasoning time is about 593-671 seconds per checkpoint; public S600 is
about 186-195 seconds. The next exact-preserving inference work is length-bucket
batching for teacher-forced/public requests, batched incremental generation
with per-sample RC-KV state, and persistent evaluator workers that swap
checkpoint weights. A BF16 Triton inference mode should remain separate from
the FP32 reference until numerical equivalence is quantified.

## 7. Conclusion

The formal DP-C2048 run completed the full 5B budget in under seven hours and
reduced the matched baseline training gap from 4.10x to 2.82x. It maintains
healthy aggregate block usage and reaches its strongest reasoning result at
75,000 steps. Its non-monotonic capability metrics confirm that validation PPL
alone is insufficient for checkpoint selection. RC-KV matches baseline
validation and public S600 within the current resolution but remains 5.5
percentage points behind on reasoning exact. Training throughput is now usable;
the next performance bottleneck is the serial, non-incremental benchmark
execution path.
