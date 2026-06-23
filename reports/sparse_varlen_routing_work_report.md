# Sparse Varlen Routing B 方案工作报告

更新时间：2026-06-23 19:17 JST

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
