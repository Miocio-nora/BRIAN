# Sparse Varlen Routing B 方案工作报告

更新时间：2026-06-23 22:36 JST

## 目标

在保证现有资产安全的前提下，为非 Global BRIAN route-core 增加 `sparse_varlen` block 执行后端，用于替代当前 A 方案 `sparse` 中 selected-query padded SDPA 的冗余计算。

核心验收标准不是 GPU utilization，而是：

- 语义等价：`full_sequence`、`sparse`、`sparse_varlen` 在同一路由选择下输出一致。
- 因果安全：替换 suffix 不影响 prefix logits。
- 训练可用：CUDA bf16 前向和 backward 正常，无 NaN/Inf。
- DDP 可用：2 卡 eval/training smoke 正常。
- 性能有效：显著超过 A 方案约 66-67k tokens/s；低于 150k tokens/s 视为工程负结果。

## 资产隔离

- Git branch：`sparse-varlen-routing`
- Base commit：`71bbe8a`
- Conda env：`brian-sparse-varlen`
- Source env：`brian-sphere`
- 当前运行中的 0/1 GPU global 实验不改动。
- smoke 输出目录固定为 `tmp/runs_sparse_varlen_smoke`。
- smoke 默认 `wandb.enabled: false`。
- 现有未跟踪 reports/tmp 文件不清理、不改写。

## 当前环境

- PyTorch：`2.11.0+cu128`
- CUDA runtime：`12.8`
- Triton：`3.6.0`
- FlexAttention：可 import `torch.nn.attention.flex_attention`
- `flash_attn`：当前环境未安装

结论：B 方案优先使用 FlexAttention packed-flat 后端；只有当 DDP/bf16/性能 gate 失败时，才考虑同接口下替换为自定义 Triton kernel。

## 实现设计

新增配置：

- `route_block_execution: sparse_varlen`
- model config：`configs/model/brian_r125_sphere16_no_location_bias_sparse_varlen.yaml`
- smoke train config：`configs/train/smoke_sparse_varlen_r125_5b_ddp2_legacyval.yaml`

执行路径：

- `full_sequence`：保持原实现。
- `sparse`：保持当前 A 方案，selected query + padded SDPA。
- `sparse_varlen`：新增 B 方案，selected queries packed 成单个 Q 序列，K/V 展平成 `batch * seq_len`，由 FlexAttention block mask 约束 `same_batch && key_pos <= query_pos`。

v1 限制：

- 只用于非 Global 路径。
- 与 `global_kv`、`attention_global_kv`、`parallel_passing` 不混用。
- 当前 r125 dropout 为 0；若训练态 attention dropout 非 0，`sparse_varlen` 显式拒绝。

## 已完成验收

### 单元测试

命令：

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
```

结果：

```text
11 passed
```

覆盖内容：

- top-1 selected block：`full_sequence` vs `sparse`/`sparse_varlen` allclose。
- top-2 weighted fusion：`full_sequence` vs `sparse`/`sparse_varlen` allclose。
- empty batch rows for action。
- CUDA bf16 autocast 前向。
- CUDA backward：完整 `BrianRouteCore(route_block_execution="sparse_varlen")` 对 logits mean 反传，qkv 梯度 finite。
- suffix invariance：替换 suffix 不改变 prefix logits。

注意：CPU FlexAttention 不支持 backward，因此 CPU 等价测试使用 `torch.no_grad()`；真正训练路径由 CUDA backward 和 DDP smoke 覆盖。

### 配置测试

命令：

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
21 passed
```

### DDP eval consistency

单卡命令：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python scripts/check_ddp_eval_consistency.py --config configs/train/smoke_sparse_varlen_r125_5b_ddp2_legacyval.yaml --split val_legacy --batch-size 2 --max-batches 1 --output tmp/sparse_varlen_single_eval_consistency.json
```

2 卡命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m torch.distributed.run --nproc_per_node=2 scripts/check_ddp_eval_consistency.py --config configs/train/smoke_sparse_varlen_r125_5b_ddp2_legacyval.yaml --split val_legacy --batch-size 2 --max-batches 1 --output tmp/sparse_varlen_ddp_eval_consistency.json
```

关键指标完全一致：

| Metric | Single | DDP2 | Diff |
| --- | ---: | ---: | ---: |
| validation_loss | 114.93109893798828 | 114.93109893798828 | 0.0 |
| perplexity | 485165195.4097903 | 485165195.4097903 | 0.0 |
| route_entropy | 1.9681446552276611 | 1.9681446552276611 | 0.0 |
| route_imitation_accuracy | 0.9375 | 0.9375 | 0.0 |
| average_route_steps | 16.0 | 16.0 | 0.0 |
| block_load_entropy | 1.965482473373413 | 1.965482473373413 | 0.0 |

### DDP train smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m torch.distributed.run --nproc_per_node=2 scripts/train.py --config configs/train/smoke_sparse_varlen_r125_5b_ddp2_legacyval.yaml
```

输出目录：

```text
tmp/runs_sparse_varlen_smoke/smoke_sparse_varlen_r125_5b_ddp2_legacyval
```

结果：

| Item | Value |
| --- | ---: |
| train steps | 100 |
| final loss | 23.1985 |
| final lm_loss | 23.1564 |
| final tokens/s | 24,944 |
| all-step mean tokens/s | 23,064.98 |
| last-20 mean tokens/s | 23,979.30 |
| last-20 mean step time | 2.7357 s |
| max CUDA memory allocated | ~18.0 GB/rank |
| eval step 50 validation_loss | 46.0241 |
| eval step 100 validation_loss | 28.9867 |

稳定性结论：

- 训练完成，无 OOM。
- 没有 NaN/Inf。
- DDP + bf16 + backward 可用。
- checkpoint 写出：`checkpoint_latest` 和 `checkpoint_step_00000100`。

性能结论：

- B 方案当前实现明显未过性能 gate。
- last-20 mean 约 23.98k tokens/s，低于 A 方案约 66-67k tokens/s，也远低于 no-go 阈值 150k tokens/s。
- 当前 FlexAttention packed-flat 原型功能正确，但工程性能不可作为正式训练后端。

## 待完成验收

- 若继续推进性能，下一步不是直接正式训练，而是替换 B 后端为自定义 Triton varlen/persistent selected attention，或者重审当前 full-sequence 路径为什么反而更快。

## 当前结论

`sparse_varlen` 的语义、因果性、CUDA backward、DDP eval、DDP train smoke 均通过；但吞吐严重失败。当前实现应保留为受测原型和 correctness reference，不建议进入正式 Package A 训练，也不建议替代当前 `sparse` A 方案。

## 2026-06-23 追加优化检查

### 新证据：per-block microbenchmark

在 B200/cu128、r125 block shape (`B=4, S=2048, D=768, H=12`) 上，单个 route block 的局部计时显示：

| Path | Forward mean | Backward mean | 结论 |
| --- | ---: | ---: | --- |
| full dense block + select | ~0.95 ms | ~2.85 ms | 最快/最稳定 |
| current padded sparse A | ~1.35 ms | ~3.39 ms | 比 dense 慢 |
| FlexAttention varlen B | ~7.5 ms | ~10.2 ms | 明显不可用 |

不同 selected density 从 0.1% 到 50% 的测试也没有发现 Flex B 划算区间；dense fused SDPA/GEMM 在 B200 上太强，selected sparse 的小 kernel、mask 构造、scatter/gather 开销吃掉了理论 FLOPs 优势。

### 代码调整

`route_block_execution: sparse_varlen` 现在支持 backend 选择：

- `BRIAN_SPARSE_VARLEN_BACKEND=flex`：强制使用原 FlexAttention packed-varlen reference backend。
- `BRIAN_SPARSE_VARLEN_BACKEND=dense`：使用 dense full block + selected gather。
- `BRIAN_SPARSE_VARLEN_BACKEND=dense_compiled`：实验性编译 dense block 后 selected gather。
- `auto`：CPU 走 `flex` 以保留 correctness reference；CUDA 默认走 `dense`，避免 Flex 性能灾难。

新增对照配置：

- `configs/train/smoke_sparse_varlen_auto_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_full_sequence_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_sparse_varlen_auto_ddp_static_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_sparse_varlen_auto_compiled_r125_5b_ddp2_legacyval.yaml`

### 追加测试

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
12 passed
21 passed
```

新增覆盖：

- `dense` backend 强制模式下与 `full_sequence` top-1 输出 allclose。
- `dense_compiled` backend 保留为实验选项，但不作为默认。

### 追加 smoke 对照

所有 run 都在 `CUDA_VISIBLE_DEVICES=2,3` 上执行。注意：当时 GPU 2/3 同时存在外部 eval 任务，各占约 30GB 显存，因此绝对吞吐低于空卡状态；但同一时间窗口内的相对比较仍有参考价值。

| Run | Backend | Steps | Mean tokens/s | Last-20 tokens/s | Final tokens/s | Max tokens/s | Last eval tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke_sparse_varlen_r125_5b_ddp2_legacyval` | Flex varlen | 100 | 23,064.98 | 23,979.30 | 24,944 | 25,171 | 14,171 |
| `smoke_sparse_varlen_auto_r125_5b_ddp2_legacyval` | auto dense | 100 | 53,991.95 | 56,511.40 | 45,302 | 79,000 | 28,911 |
| `smoke_full_sequence_r125_5b_ddp2_legacyval` | full sequence | 100 | 63,575.38 | 68,269.20 | 81,808 | 87,721 | 24,536 |
| `smoke_sparse_varlen_auto_ddp_static_r125_5b_ddp2_legacyval` | auto dense, DDP static | 100 | 56,071.68 | 48,705.30 | 58,551 | 83,730 | 24,879 |
| `smoke_sparse_varlen_auto_compiled_r125_5b_ddp2_legacyval` | dense compiled | 100 | 63,118.33 | 62,493.65 | 53,527 | 97,883 | 19,440 |

### Updated Conclusion

Correctness 已经比较充分：Flex reference、dense fallback、CUDA bf16/backward、DDP eval、DDP train 都跑通。

性能结论仍然没有达成目标：

- Flex varlen 原方案从性能上失败。
- CUDA auto dense fallback 能把 Flex 的 24k 提升到约 54-56k mean/last20，但只是恢复到 A/full-sequence 同级别，不能证明真正 sparse 加速。
- full_sequence 在同一窗口仍更强，last20 约 68k，说明当前 selected sparse 路径不是瓶颈解。
- `ddp_find_unused_parameters=false` 没有改善，且对未来 router collapse 场景不安全。
- `dense_compiled` 有单点 97.9k，但均值不稳定，不能作为默认。

下一步若必须达到预期加速，不能再只替换 attention API；需要做 grouped expert execution：

- 用 `torch._grouped_mm` / Triton grouped GEMM 合并多个 route block 的 QKV/out/FFN linear。
- attention 需要按 expert 分组后批处理，避免每个 block 一个 Python/kernel 小调用。
- 目标是把 16 route steps * 8 route blocks 的碎片化执行降到少量 grouped kernels。

当前提交可作为 correctness-safe 原型和性能负结果记录；尚不能标记为“达到预期加速”。

## 2026-06-23 Grouped Expert Execution 追加检查

### 实现

新增显式实验后端：

- `route_block_execution: grouped_dense`
- model config：`configs/model/brian_r125_sphere16_no_location_bias_grouped_dense.yaml`
- smoke config：`configs/train/smoke_grouped_dense_r125_5b_ddp2_legacyval.yaml`

该后端在一个 route step 内将 8 个 route block 作为 expert 维度一起执行：

- position adapter、RMSNorm、QKV、attention out、FFN linear 使用 stacked expert weights。
- attention 把 expert 维度并入 batch，一次调用 SDPA。
- CUDA 下使用 `torch.compile(dynamic=False)` 编译 grouped route-block forward。
- top-1/top-2 fusion 语义保持和 `full_sequence` 一致。
- 不支持 global/attention_global/parallel_passing/activation_checkpointing 路径。

### Correctness

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
16 passed
21 passed
```

新增覆盖：

- `grouped_dense` top-1 allclose。
- `grouped_dense` top-2 weighted fusion allclose。
- `grouped_dense` CUDA bf16/backward finite。
- config/model_stats validation。

### Microbenchmark

单 route step、8 route blocks、`B=4,S=2048,D=768,H=12`：

| Path | Forward mean | Backward mean | 结论 |
| --- | ---: | ---: | --- |
| loop 8 dense blocks | 6.47 ms | 17.09 ms | 当前 full_sequence 的基本形态 |
| grouped 8 dense experts | 5.46 ms | 14.91 ms | eager 有 12-16% 改善 |
| compiled grouped 8 dense experts | 2.07 ms | 7.38 ms | microbench 很强 |

但接入完整训练后，没有复现 microbench 的大幅优势。

### DDP Smoke

空 GPU 2/3 上执行：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m torch.distributed.run --nproc_per_node=2 scripts/train.py --config configs/train/smoke_grouped_dense_r125_5b_ddp2_legacyval.yaml
```

结果：

| Run | Steps | Mean tokens/s | Last-20 tokens/s | Final tokens/s | Max memory/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `smoke_grouped_dense_r125_5b_ddp2_legacyval` | 100 | ~79.7k last-50 | ~79.6k | ~79.8k | ~44GB |

同一空卡窗口重跑 full sequence：

| Run | Steps | Mean tokens/s | Last-20 tokens/s | Final tokens/s | Max memory/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `smoke_full_sequence_r125_5b_ddp2_legacyval` | 100 | 86.0k all-step | 87.0k | 87.9k | ~28GB |

结论：

- `grouped_dense` 正确，但完整训练吞吐低于 `full_sequence`。
- 它的显存明显更高，因为每个 route step materialize `E x B x S x D` expert activations。
- microbench 的 compiled grouped 优势没有转化为 end-to-end 优势，主要被 route scatter/fusion、compile graph 边界、显存带宽和更大的 activation footprint 抵消。

### Grouped Sparse-Padded Prototype

还测试了未接入代码的 grouped sparse-padded 原型：把 8 个 block 的 selected-Q attention 合成一次 batched SDPA，K/V 仍 full sequence。

单 step 结果：

| Path | Forward mean | Backward mean |
| --- | ---: | ---: |
| loop sparse A over 8 blocks | 10.73 ms | 23.45 ms |
| loop dense 8 blocks + select | 7.62 ms | 20.89 ms |
| grouped sparse-padded | 8.67 ms | 29.12 ms |

结论：

- grouped sparse-padded forward 仍慢于 dense-select。
- backward 明显更差。
- padding/scatter/gather 和更复杂反传图抵消了 selected-Q 的理论 FLOPs 节省。

### Current State

本轮把 grouped expert execution 做到了 correctness-safe 的实验后端，但它没有达到预期加速，也不应作为默认正式训练路径。

截至目前的可靠判断：

- PyTorch 级别的 selected/varlen/grouped 组合都没有超过 dense full-sequence baseline。
- 预期加速若仍要实现，需要更底层的 fused Triton/CUDA kernel，目标是同时融合 expert GEMM、selected attention、scatter/fusion，而不是用多个 PyTorch op 拼装。
- 当前分支保留 `sparse_varlen` 和 `grouped_dense` 作为可复现实验后端和负结果依据。

## 2026-06-23 Training Summary / DDP Overhead 检查

### 代码调整

新增训练期 routing summary 开关：

- `BrianRouteCore.forward(..., summarize_routing=True)`
- train config `routing.summary_interval`
- 默认 `summary_interval: 1`，保持旧行为。
- `summary_interval: 0` 只关闭训练 forward 的 routing summary；eval 仍保留 summary，用于验证和报告。

新增 smoke 配置：

- `configs/train/smoke_full_sequence_fastlog_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_full_sequence_fastlog_nounused_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_r125_5b_ddp2_legacyval.yaml`

`nounused` 版本额外设置 `ddp_find_unused_parameters: false`。这只用于 `full_sequence` / `grouped_dense` 的性能 smoke，因为这两条路径每步都会触达所有 route block 参数；不推广到真正 sparse/free collapse 风险路径。

### 测试

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
16 passed
21 passed
```

### Fastlog / Nounused Smoke

均使用空 GPU 2/3、`max_steps=100`、`wandb.enabled=false`、`checkpoint_benchmarks.enabled=false`。

| Run | Summary interval | DDP unused check | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Final max memory/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `smoke_full_sequence_fastlog_r125_5b_ddp2_legacyval` | 0 | true | 87,527.55 | 87,569.12 | 89,055 | ~27.9GB |
| `smoke_full_sequence_fastlog_nounused_r125_5b_ddp2_legacyval` | 0 | false | 88,284.25 | 87,979.70 | 89,385 | ~28.0GB |
| `smoke_grouped_dense_fastlog_r125_5b_ddp2_legacyval` | 0 | true | 79,231.45 | 79,043.04 | 79,065 | ~44.0GB |
| `smoke_grouped_dense_fastlog_nounused_r125_5b_ddp2_legacyval` | 0 | false | 80,093.90 | 80,104.20 | 79,847 | ~44.0GB |

### 结论

- 每步 routing summary 不是主要吞吐瓶颈：关掉后 full-sequence 仍约 87.5k，与之前空卡约 87.0k 基本一致。
- `ddp_find_unused_parameters=false` 只带来小幅提升：full-sequence last20 +0.9%，grouped-dense last20 +1.1%。
- `grouped_dense` 在更公平的 fastlog/nounused 对照下仍低于 `full_sequence`：80.1k vs 88.3k，约慢 9.3%，且显存高约 16GB/rank。
- 因此当前 B 方案实现正确、资产隔离良好，但没有达到预期加速。后续如果继续追求 route execution 加速，应该转向更底层 fused kernel 或重新定义训练 forward 语义，不能继续依赖 PyTorch op 级别拼装。

## 2026-06-23 Grouped Dense Weight Cache / Gather 优化

### 代码调整

本轮继续优化 `route_block_execution: grouped_dense`，仍不改变 routing 语义和模型参数结构。

新增优化：

- 每次 model forward 只 stack 一次 route-block expert weights，在 16 个 route steps 内复用。
- `grouped_dense` top-1 输出从逐 action mask/scatter 改为一次 `gather`。
- top-2 weighted fusion 输出改为向量化 `gather + weighted sum`。
- 新增 `float32_matmul_precision` train config 选项，默认不设置；smoke 里可显式设为 `high` 以启用 TF32 matmul policy。

新增 smoke 配置：

- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_tf32_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_full_sequence_fastlog_nounused_tf32_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_full_sequence_fastlog_tf32_router1_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_routedcompile_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_routedcompile_r125_5b_ddp2_legacyval.yaml`

### 测试

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
16 passed
21 passed
```

### Smoke 结果

均使用空 GPU 2/3、`max_steps=100`、`batch_size=4`、`gradient_accumulation_steps=4`、`summary_interval=0`、`ddp_find_unused_parameters=false`。

| Run | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Final max memory/rank |
| --- | ---: | ---: | ---: | ---: |
| `smoke_full_sequence_fastlog_nounused_r125_5b_ddp2_legacyval` | 88,284.25 | 87,979.70 | 89,385 | ~28.0GB |
| `smoke_full_sequence_fastlog_nounused_tf32_r125_5b_ddp2_legacyval` | 88,076.85 | 87,568.58 | 89,526 | ~28.0GB |
| `smoke_grouped_dense_fastlog_nounused_r125_5b_ddp2_legacyval` | 80,093.90 | 80,104.20 | 79,847 | ~44.0GB |
| `smoke_grouped_dense_fastlog_nounused_weightcache_r125_5b_ddp2_legacyval` | 83,401.50 | 83,573.12 | 83,162 | ~44.3GB |
| `smoke_grouped_dense_fastlog_nounused_weightcache_tf32_r125_5b_ddp2_legacyval` | 84,154.50 | 84,189.24 | 83,862 | ~44.3GB |
| `smoke_grouped_dense_fastlog_nounused_weightcache_gather_r125_5b_ddp2_legacyval` | 92,861.35 | 92,878.02 | 92,466 | ~49.8GB |
| `smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_r125_5b_ddp2_legacyval` | 93,026.25 | 93,116.70 | 92,353 | ~49.8GB |

### 结论

- 一次 forward 内复用 stacked expert weights 有效：grouped-dense last20 从 80.1k 提升到 83.4k，约 +4.1%。
- TF32 policy 对 grouped-dense 有小幅帮助：83.4k 到 84.2k，约 +0.9%；对 full-sequence 没有稳定提升。
- 向量化 gather/fusion 是本轮关键收益：83.4k 到 92.9k，约 +11.3%。
- 当前最佳 grouped-dense smoke：93.0k last20，高于 full-sequence TF32 88.1k，约 +5.6%。
- 代价：训练峰值显存从 full-sequence ~28.0GB/rank 增至 grouped gather ~49.8GB/rank。

当前判断：

- `grouped_dense` 已经从性能负结果推进到小幅正结果；这是目前唯一超过 dense full-sequence baseline 的 PyTorch-level route execution 后端。
- 这仍不是 150k tokens/s 级别的预期加速；若最终目标仍是大幅加速，需要继续减少 `E x B x S x D` materialization 或转向 fused grouped GEMM/attention kernel。
- 上表前 100 step 的 schedule 中 `router_probability=0`，因此主要覆盖 top-1/fixed target 路径；top-2 weighted fusion 另见下方 router1 smoke。

### Router1 / Top-2 Weighted Fusion Smoke

为了覆盖 schedule 后期 `router_probability>0` 的路径，新增 `router1` smoke：从 step 1 起强制 `router_probability=1.0`，触发 router/top-2 weighted fusion。

注意：`full_sequence` 在自由路由/top-2 下可能不会每步触达所有 route block 参数，因此不能设置 `ddp_find_unused_parameters=false`。一次错误的 `full_sequence + router1 + nounused` 试跑在 step 51 后 DDP 挂住；正确 baseline 使用 `find_unused_parameters=true`。`grouped_dense` 每步计算全部 experts，因此仍可安全使用 `ddp_find_unused_parameters=false`。

| Run | DDP unused check | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Final max memory/rank |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke_full_sequence_fastlog_tf32_router1_r125_5b_ddp2_legacyval` | true | 64,640.70 | 64,599.00 | 64,063 | ~41.1GB |
| `smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_r125_5b_ddp2_legacyval` | false | 79,427.50 | 78,969.18 | 79,853 | ~50.8GB |

router1/top-2 结论：

- grouped gather 在 top-2 weighted fusion 路径上领先更明显：79.4k vs 64.6k，约 +22.9%。
- `grouped_dense` 的优势来自每 step 一次性计算全部 experts 并向量化 gather/fusion；full-sequence 在自由路由下只能按实际被选 block 逐 action 执行，DDP 还必须开启 unused-parameter 检查。
- 代价仍是显存：router1 grouped gather 约 50.8GB/rank，full-sequence router1 约 41.1GB/rank。

### Routed-Level Compile

在上一版基础上继续把 `expert forward + gather/fusion` 放进同一个 compiled routed wrapper，而不是只 compile grouped expert forward。这个改动不改变数学语义，主要目的：

- 让 Inductor 看到 grouped expert 输出的消费者，减少 graph 边界。
- top-1 / `router_probability=0` 路径用 Python bool 跳过 `torch.any(use_weighted_fusion)` 的 per-step 同步。
- top-2 / router1 路径也把 weighted gather/fusion 纳入 compiled wrapper。

| Run | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Final max memory/rank |
| --- | ---: | ---: | ---: | ---: |
| grouped gather TF32 top-1 | 93,026.25 | 93,116.70 | 92,353 | ~49.8GB |
| grouped gather TF32 routedcompile top-1 | 95,803.05 | 96,052.32 | 95,056 | ~46.7GB |
| grouped gather TF32 router1 | 79,427.50 | 78,969.18 | 79,853 | ~50.8GB |
| grouped gather TF32 routedcompile router1 | 90,850.30 | 85,987.16 | 81,686 | ~47.4GB |

routedcompile 结论：

- top-1 路径稳定提升：95.8k vs 93.0k，约 +3.0%，同时显存从 ~49.8GB/rank 降到 ~46.7GB/rank。
- router1/top-2 路径 last20 提升明显：90.9k vs 79.4k，约 +14.4%；但 last50 和 final 波动较大，需要更长 run 判断稳态。
- 目前最佳 top-1 smoke 距 150k 仍有明显距离；继续优化的主方向仍是减少或融合 `E x B x S x D` expert activation materialization。

## 2026-06-23 Plan Execution Status / Safety Note

本轮按“保证已有资产稳定，在此基础上尝试 B 方案，并充分验收”的原则处理：

- `sparse_varlen` packed-varlen B 方案已经实现并验收，功能正确但吞吐失败。
- `grouped_dense` 作为后续 PyTorch-level grouped expert B 方向已经实现并验收，当前最佳 top-1 smoke 达到约 95.8k tokens/s，router1/top-2 last20 达到约 90.9k tokens/s。
- 所有正式保留路径仍通过 `tests/test_sparse_route_block_execution.py` 和 `tests/test_config_inventory.py`。
- 当前实现通过独立 branch、独立 conda env、独立 smoke output root 隔离；没有改动 global-kv 正式配置和历史 checkpoint。

### Rejected: `max-autotune` Compile Mode

曾尝试在 `grouped_dense` routedcompile 上开启 `torch.compile(mode="max-autotune")`。结果在 step 50 eval 阶段触发 CUDAGraph 输出复用错误：

```text
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

这个结果说明 `max-autotune` 不能作为默认或推荐实验路径进入主线；相关临时配置和环境开关已从工作树移除。报告保留该负结果，避免后续误把它当作可继承优化。

### Current Decision

当前 B 方案已经达成“可安全运行、可复现、比 full-sequence 有小幅加速”的阶段，但尚未达成 150k tokens/s 级别预期。继续追求大幅加速时，不应再优先堆叠 PyTorch compile/autotune 开关；更合理的下一步是新增独立 fused-kernel prototype，目标是减少 `E x B x S x D` activation materialization，并把 expert GEMM、attention、gather/fusion 的边界继续下沉。

## 2026-06-23 B200 Microbatch Scaling

上一节的最佳 grouped-dense routedcompile 使用 `batch_size=4, gradient_accumulation_steps=4`，虽然 global batch 是 32，但每个 optimizer step 有 4 个 microbatches/rank。B200 显存明显未被用满，因此测试保持 global batch 32 不变，把 microbatch 放大、accumulation 减少。

新增正收益配置：

- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_routedcompile_b8acc2_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_routedcompile_b16acc1_r125_5b_ddp2_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_routedcompile_b16acc1_r125_5b_ddp2_legacyval.yaml`

top-1 / scheduled target path：

| Run | Batch/Accum | Effective batch | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Step time last20 | Peak memory/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped routedcompile | 4/4 | 32 | 95,803.05 | 96,052.32 | 95,056 | 0.6841s | ~46.7GB |
| grouped routedcompile b8acc2 | 8/2 | 32 | 112,267.00 | 112,326.12 | 112,843 | 0.5838s | ~88.0GB |
| grouped routedcompile b16acc1 | 16/1 | 32 | 121,582.75 | 121,900.78 | 122,015 | 0.5390s | ~170.0GB |

router1 / top-2 weighted fusion path：

| Run | Batch/Accum | Effective batch | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Step time last20 | Peak memory/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped routedcompile router1 | 4/4 | 32 | 90,850.30 | 85,987.16 | 81,686 | 0.7289s | ~47.4GB |
| grouped routedcompile router1 b16acc1 | 16/1 | 32 | 105,242.75 | 104,468.28 | 100,292 | 0.6236s | ~173.0GB |

结论：

- batch scaling 是本轮最有效的非 kernel 优化：top-1 从 95.8k 提到 121.6k，约 +26.9%；router1/top-2 从 90.9k 提到 105.2k，约 +15.8%。
- `b16acc1` 已接近 B200 单卡显存上限，继续在 2 卡上靠 batch 放大逼近 150k 的空间很小。
- `b8acc2` 是更稳妥的 B200 配置，显存约 88GB/rank；`b16acc1` 是吞吐优先配置，显存约 170GB/rank。
- 即便使用 b16acc1，2 卡总吞吐仍未达到 150k gate；若不改变 kernel/activation materialization，下一步只能靠 4 卡总吞吐或 fused kernel 继续推进。

负结果：

- `hybrid grouped_mm` 只在 standalone QKV microbench 有小幅收益，完整 top-1 smoke 为 95.68k last20，低于当前 einsum/routedcompile 95.80k；未进入主线实现。
- `ddp_static_graph + gradient_as_bucket_view` 在 b16acc1 上无实质收益：121.54k last20 vs 121.58k；训练器保留可选 DDP 开关，但不推荐作为当前性能路径。

## 2026-06-23 DDP4 Validation Prep

为了验证 150k gate，不再继续在 2 卡上挤 batch。下一步更合理的是用 0-3 四卡保持 global batch 不变：

- 2 卡 b16acc1：`world_size=2, batch_size=16, grad_accum=1`，effective batch = 32。
- 4 卡 b8acc1：`world_size=4, batch_size=8, grad_accum=1`，effective batch = 32。

新增待跑配置：

- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_routedcompile_b8acc1_r125_5b_ddp4_legacyval.yaml`
- `configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_routedcompile_b8acc1_r125_5b_ddp4_legacyval.yaml`

验证命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m torch.distributed.run --nproc_per_node=4 scripts/train.py --config configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_routedcompile_b8acc1_r125_5b_ddp4_legacyval.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m torch.distributed.run --nproc_per_node=4 scripts/train.py --config configs/train/smoke_grouped_dense_fastlog_nounused_weightcache_gather_tf32_router1_routedcompile_b8acc1_r125_5b_ddp4_legacyval.yaml
```

验收标准：

- top-1 DDP4 last20 tokens/s > 150k。
- router1/top-2 DDP4 last20 tokens/s > 150k。
- 每 rank peak memory 应接近 b8acc2 的单 microbatch footprint，预计显著低于 b16acc1；实测为准。

当前 0/1 GPU 仍被其他实验占用，因此本节只完成配置和测试准备，尚未把四卡吞吐写成结果。

## 2026-06-23 Training Route Info Fast Path

训练配置中 `routing.summary_interval: 0` 会关闭 routing summary，但旧 forward 仍然每个 route step 记录 top-k actions、top-k weights、weighted-fusion flags、exit flags、position norms、random-route/self-recur counts 等诊断字段。这些字段不参与 loss，只服务 eval/report/visualization。

本轮改动：

- 当 `summarize_routing=False` 且不收集 router-space 时，只保留 loss 必需字段：
  - `route_logits`
  - `route_probs`
  - `selected_actions`
  - `route_targets`
  - `location_distance`
- eval、routing summary、route path visualization、router-space collection 仍保留完整诊断字段。
- 新增测试 `test_summarize_routing_false_keeps_loss_fields_without_diagnostics` 覆盖精简路径可以正常 backward，且诊断字段为空。

测试：

```bash
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sparse-varlen/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
17 passed
21 passed
```

b16acc1 对照：

| Run | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Peak memory/rank |
| --- | ---: | ---: | ---: | ---: |
| b16acc1 before | 121,582.75 | 121,900.78 | 122,015 | ~170.0GB |
| b16acc1 fast route-info | 121,610.80 | 122,041.92 | 121,827 | ~169.7GB |

随后继续清理 0 权重辅助 loss：当 loss weight 为 0 时，不再计算对应 raw component。当前 smoke 中 `route` schedule lambda 为 0，`balance=0`，`transition_diversity=0`，因此可以跳过 route imitation、balance、transition diversity 的计算。日志里这些 0 权重 component 现在记录为 0，表示对 total loss 的贡献为 0，而不是 raw diagnostic value。

fastloss 对照：

| Run | Last-20 tokens/s | Last-50 tokens/s | Final tokens/s | Peak memory/rank |
| --- | ---: | ---: | ---: | ---: |
| b16acc1 before | 121,582.75 | 121,900.78 | 122,015 | ~170.0GB |
| b16acc1 fast route-info | 121,610.80 | 122,041.92 | 121,827 | ~169.7GB |
| b16acc1 fast route-info + zero-weight loss skip | 122,046.75 | 122,396.20 | 121,973 | ~169.7GB |

结论：训练路径清理有小幅正收益，约 +0.4% last20；仍不是接近 150k 的关键路径。当前 gate 仍需要 DDP4 实测或 fused kernel。
