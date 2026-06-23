# Sparse Varlen Routing B 方案工作报告

更新时间：2026-06-23 20:16 JST

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
