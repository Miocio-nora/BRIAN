# RC-KV Deep Acceleration Report

Date: 2026-07-16

## 1. Scope and Safety

This work accelerates the existing CPBC-DP RC-KV training implementation. It
does not change the RC-KV cache definition, route policy, causal mask, depth
visibility policy, global batch, TBPTT boundary, or model parameterization.

Development was isolated in:

```text
branch:   rc-kv-deep-acceleration
worktree: /nvmesv/dredvpn009/projects/brian-rc-kv-accel
base:     589d46f
```

The existing CPBC run on GPUs 0-1 was not stopped or modified. Profiling and
acceptance used only GPUs 2-3.

## 2. Accepted Implementation

The accepted backend is additive:

```yaml
execution:
  mode: synchronous_prefix
  attention_backend: shared_padded_explicit
  dispatch: grouped_mm
```

The old `legacy_cuda_scan` and `grouped_host` modes remain available as
reference paths with unchanged execution semantics.

### 2.1 Grouped expert GEMMs

Active tokens are sorted once by route block. The implementation stacks the
corresponding block matrices and uses `torch.nn.functional.grouped_mm` for:

- position adapters;
- attention QKV;
- canonical Key and Value writers;
- attention output projections;
- FFN gate/up and down projections.

FFN gate and up are emitted by one grouped GEMM and split afterward. RMSNorm
retains per-block weights by gathering the weight associated with each routed
token. CUDA autocast matrices are explicitly prepared as BF16 because
`grouped_mm` does not inherit `F.linear` autocast behavior automatically.

The implementation falls back to per-group matrix multiplication when the
CUDA BF16 grouped-MM API is unavailable.

### 2.2 Narrow reader-cache gather

The previous shared-padded reader path selected all route steps and all reader
blocks for each active batch, then sliced one `(reader_step, reader_block)`.
The accepted path directly gathers only the requested cache slice. The cache
contents, causal mask, and reader projection are unchanged.

### 2.3 Conditional diagnostics

BDRE compiler entropy and last-step metrics are now optional. They are skipped
when `summarize_routing=false`; cache tensors and compile weights remain
identical. The optimized long-run config records a full routing summary every
10 optimizer steps:

```yaml
routing:
  summary_interval: 10
```

This does not disable the terminal monitor. Its detached process can still
read one real route every step, while detailed histograms and transition
tables are emitted every 10 steps.

## 3. Correctness Acceptance

The following checks passed:

- CPU output, loss, and gradient equivalence against `grouped_host`;
- mixed per-token routing and all-OUT edge cases;
- suffix invariance and full/streamed/incremental invariance tests;
- BF16 CUDA output and loss equality with finite gradients;
- three-step R125 DDP2 training at local/global batch 16/32;
- manual DDP gradient sync: 135 globally used parameters, 0 locally missing;
- legacy validation and two rank-local checkpoint states.

Grouped GEMM can change BF16 backward accumulation order. The accepted CUDA
check produced identical logits and loss; sampled gradient differences were
within BF16 rounding scale. A bitwise-identical optimizer trajectory is not
claimed.

## 4. Performance

All single-GPU measurements use one NVIDIA B200, R125 BF16, sequence length
2,048, CPBC-DP chunk 512, U=1, one warmup, and repeated backward passes.

### 4.1 Single B200

| Shape | Original median | Accepted median | Change | Original peak allocated | Accepted peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| BS4 | 4,816 tok/s | 6,891 tok/s | +43.1% | 18,015 MiB | 18,237 MiB |
| BS16 | 11,658 tok/s | 15,167 tok/s | +30.1% | 120,500 MiB | 120,947 MiB |

The formal local-BS16 shape gains about 30% throughput while peak allocated
memory rises by about 0.4%.

### 4.2 DDP2 formal shape

The accepted smoke preserves global batch 32 and 65,536 tokens per optimizer
step.

| Metric | Existing CPBC-DP DDP2 | Accepted grouped-MM DDP2 |
| --- | ---: | ---: |
| Steady global throughput | 15,996-16,555 tok/s | 28,638-30,609 tok/s |
| Mean steady throughput | about 16,276 tok/s | about 29,624 tok/s |
| Mean improvement | - | +82.0% |
| Steady step time | 3.96-4.10 s | 2.14-2.29 s |
| Peak allocated per rank | 123.7 GiB | 124.5 GiB |
| Pure 5B training estimate | 3.56 days | 1.95 days |

The 5B estimate excludes evaluation, public benchmarks, checkpoint I/O, and
W&B. A practical run should be budgeted above 1.95 days.

### 4.3 Profiler evidence

At BS2, sequence/chunk 512:

| Event | Original | Accepted | Change |
| --- | ---: | ---: | ---: |
| CUDA kernel launches | 28,580 | 20,051 | -29.8% |
| `aten::mm` calls | 3,376 | 496 | -85.3% |
| `aten::copy_` calls | 8,637 | 6,285 | -27.2% |
| Self CUDA time | 109.65 ms | 81.59 ms | -25.6% |

The largest remaining CUDA category is `aten::copy_`, at roughly 24% of self
CUDA time. Host action dispatch and many small elementwise routing kernels also
remain visible.

## 5. Rejected Prototype

A second prototype fused all active `(reader_block, batch)` attention groups.
It reached about 7,600 tok/s at BS4, but raised allocated memory from about
18.2 GiB to 21.9 GiB and OOMed at formal BS16. It also introduced additional
BF16 reader-gradient accumulation differences.

That prototype was removed from the accepted code. It must not be used as a
formal training configuration without a memory-bounded implementation.

## 6. Configurations

Accepted model:

```text
configs/model/brian_r125_bdre_cpbc_dp_c512_grouped_mm.yaml
```

Accepted DDP2 train config:

```text
configs/train/cpbc_r125_5b_dp_u1_c512_grouped_mm_ddp2_legacyval.yaml
```

Three-step acceptance smoke:

```text
configs/train/smoke_cpbc_r125_5b_dp_u1_c512_grouped_mm_ddp2_legacyval.yaml
```

Profiler:

```text
scripts/profile_bdre_tbptt.py
```

## 7. Reproduction

Single-GPU formal-shape benchmark:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
python scripts/benchmark_bdre_tbptt.py \
  --config configs/train/cpbc_r125_5b_dp_u1_c512_grouped_mm_ddp2_legacyval.yaml \
  --batch-size 16 --sequence-length 2048 --chunk-size 512 \
  --detach-interval-chunks 1 --warmup-steps 1 --repeats 5
```

DDP2 smoke:

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/smoke_cpbc_r125_5b_dp_u1_c512_grouped_mm_ddp2_legacyval.yaml
```

## 8. Remaining Work

The next acceleration work should target the remaining host synchronization
and copy-heavy reader path, while preserving the accepted backend as the
reference. Reasonable candidates are:

1. GPU-resident action packing with bounded reader padding;
2. a fused routed RMSNorm kernel;
3. memory-bounded reader attention batching;
4. reducing router constraint and cache-compile elementwise launches.

None of these is required to use the accepted grouped-MM configuration.
