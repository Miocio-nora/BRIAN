# Q1 CPBC Depth Visibility 250M Launch

**Date:** 2026-07-17

**Branch:** `rc-kv-triton-reader`

**GPUs:** physical GPU 0-1 only, DDP2

## 1. Question

Q1 isolates whether completed-history Full-Bank visibility improves RC-KV over
Depth-Prefix visibility. It does not compare C128 with the completed DP-C2048
5B run because that would confound visibility, chunk size, and gradient
horizon.

| Arm | Visibility | Chunk | U | Global batch | Token budget |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1-DP | depth prefix | 128 | 1 | 32 | 250,019,840 |
| Q1-FB | full bank | 128 | 1 | 32 | 250,019,840 |
| Control | plain Transformer | n/a | n/a | 32 | 250,019,840 |

Each run uses 3,815 optimizer steps with 65,536 tokens per step. The two RC-KV
arms share data, token order, seed, optimizer, constant `3e-4` learning rate,
router curriculum, slow-noise schedule, loss weights, model dimensions, Triton
reader settings, and initial parameters. Only Depth Visibility Policy and the
required FB completed-bank compiler differ.

## 2. Initialization and Config Audit

The resolved DP and FB state contracts contain 164,884,105 parameter/buffer
elements. With `torch.manual_seed(1)`, both complete initial state dictionaries
have the same SHA256:

```text
3c5b0e823a9bee97564636f63f8130c464d54a45b627c504fd63e644f6643bbf
```

Automated configuration tests reject changes to token budget, data, seed,
optimizer, routing, loss, batch, DDP, chunk, or U between the two arms.

## 3. Checkpoint Contract

Legacy validation, retained checkpoints, routing/cache visualizations, and the
full benchmark suite run at steps 1,272, 2,544, and 3,815. Reasoning uses the
strict incremental S600 profile; public S600 retains the legacy scorer. Fast
numeric profiles are not checkpoint-selection authorities.

Artifacts are written below each run directory rather than into the tracked
`reports/package_a_benchmarks` directory. Four retained checkpoints are enough
for the three planned evaluation points.

## 4. DDP2 Smoke

Both 20-step smokes completed on physical GPU 0-1 in the `brian-sphere`
environment.

| Arm | Last-10 median throughput | Last-10 range | Peak allocated/rank | Final finite loss |
| --- | ---: | ---: | ---: | ---: |
| Q1-DP | 40,525 token/s | 36,568-40,589 | 8,267 MiB | 44.1528 |
| Q1-FB | 39,992 token/s | 36,303-40,286 | 8,462 MiB | 52.2145 |

The smoke losses are not quality evidence. DP and FB intentionally diverge once
completed chunk history exists.

At the measured rates, pure training is approximately 1 h 43 min for DP and
1 h 44 min for FB. With three checkpoint benchmark stages and the inexpensive
baseline control, the serial GPU 0-1 queue is expected to take about 4.5-5 h.

## 5. Queue

The queue runs DP, then FB, then the matched baseline and stops immediately if
an arm fails:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_q1_cpbc_visibility_250m.sh
```

GPU 2-3 are not visible to the queue.

All formal arms set `resume: true`. Re-running the queue resumes an interrupted
arm from `checkpoint_latest` and skips training for an arm already at step
3,815.

## 6. Decision Rule

No arm wins on PPL alone. The comparison table must include validation/PPL,
reasoning exact and teacher accuracy, public average and task breakdown, route
entropy and coverage, route length, compiler K/V entropy, writer/last-step
mass, cache norms, throughput, and memory at all three checkpoints.

If the capability and route/cache evidence is mixed, Q1 is inconclusive. In
that case both arms must be extended rather than selecting a winner from the
final validation loss.
