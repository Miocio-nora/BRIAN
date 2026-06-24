# Pure Factorized Global KV 工作报告

更新时间：2026-06-24 02:27 JST

## 目标

新增一个不破坏现有资产的 Global KV 实验版本，用于验证新的 pure global attention 思路：

- 不替换旧 `global_kv` hidden-state cache。
- 不替换旧 `attention_global_kv` 的 `summary` / `token_compressed` 模式。
- 新增独立 `attention_global_kv_mode: pure_factorized`。
- first version 先实现可训练原型和 smoke 验证，不把它作为正式实验默认路径。

## 设计

新模式位于 attention 层内部，不再把 local K/V 与 global K/V 拼接，而是让 attention 的 K/V 全部来自压缩 global code。

当前实现：

- `WkA/WvA`：把 attention input hidden 映射到压缩 global key/value code。
- `WkA/WvA` 在 route blocks 间共享；各 block 仍保留自己的 attention Q/out/FFN。
- `WkB`：以 head 为单位的 key-read 参数，融合到 query 侧：
  - `q_code = q @ WkB`
  - `score = q_code @ global_key_code.T`
- `WvB`：以 head 为单位的 value-read 参数，放在 attention 聚合之后：
  - `value_code_out = attn @ global_value_code`
  - `value_head = value_code_out @ WvB`
- global pool 使用 compressed code，shape 为 token-shaped rank-5 cache，但 head 维固定为 1。
- cache 使用 latest-token-only 写入：
  - 每个 token 只有一个 latest global KV slot。
  - 当前 step 只覆盖 `valid=True` 的 token。
  - `valid=False` 的 token 保留上一版 latest KV。

## Training 语义

first version 采用 step-synchronous training 语义：

- 每个 attention forward 读取上一轮 global pool。
- 同时使用当前 step 的 compressed causal KV 作为 bootstrap。
- 当前 step 结束后，把 selected token 的 compressed KV 写回 latest pool。
- causal mask 保证 token 不能看未来 sequence position。

这不是最终 inference-perfect 版本。它的优点是能保持 full-sequence training；风险是和严格 token-by-token inference 的 final-latest KV 语义存在差异。后续必须专门做 full-forward vs incremental diagnostics。

## 代码改动

新增/修改：

- `src/brian_sphere_llm/model/llama_backbone.py`
  - 新增 `attention_global_kv_mode: pure_factorized`。
  - 新增 pure factorized compressed attention path。
  - 用 SDPA 执行 compressed global attention，避免 materialize `[B,H,S,K]` scores。
  - 最后一个 token 的 global mass metrics 仍保留。
- `src/brian_sphere_llm/memory/attention_global_cache.py`
  - 新增 `latest_token_only` cache mode。
  - 写入时只覆盖 valid token，其余 token 保留旧 latest KV。
- `src/brian_sphere_llm/model/brian_model.py`
  - 允许 `pure_factorized` mode。
  - pure mode 下 attention global cache 使用 headless `n_heads=1`。
  - route blocks 间共享 `WkA/WvA` writer modules。
- `configs/model/brian_r125_sphere16_no_location_bias_attention_global_kv_pure_factorized.yaml`
- `configs/train/corrected_attention_global_kv_pure_factorized_r125_5b_smoke.yaml`
- `tests/test_sparse_route_block_execution.py`

## 已验证

单元测试：

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
```

结果：

```text
19 passed
```

新增覆盖：

- `pure_factorized` suffix invariance。
- route blocks 共享 `WkA/WvA` writer modules。
- latest-token-only cache 只覆盖 valid token。
- config/model_stats 支持 `attention_global_kv_mode: pure_factorized`。

配置测试：

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_config_inventory.py -q
```

结果：

```text
21 passed
```

编译检查：

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m compileall -q src scripts
```

结果：通过。

## GPU Smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python scripts/train.py --config configs/train/corrected_attention_global_kv_pure_factorized_r125_5b_smoke.yaml
```

结果：

- run dir: `runs/smoke_corrected_attention_global_kv_pure_factorized_r125_5b`
- max steps: 3
- batch size: 1
- train/eval/checkpoint/routing_report 均完成。
- train peak memory: ~5.2GB。
- eval peak memory: ~3.0GB。
- `attention_global_kv_slots_mean`: 1.0。
- `attention_global_kv_write_count_mean`: 1920.0。
- `attention_global_kv_last_token_mass`: ~0.111。

第一次 smoke 使用继承 batch size 32，在 R125 sequence length 下 OOM。原因是最初手写 full attention score matrix。随后主 attention 路径改成 SDPA，并把 smoke batch 降为 1。当前 smoke 是功能性验证，不代表正式 batch/throughput 能力。

## 当前限制

- 还没有做 full-forward vs incremental 一致性诊断。
- 还没有做 global strength/off/swap intervention。
- pure mode 当前没有 local K/V fallback，因此训练初期 loss 很高是预期风险。
- RoPE 暂未直接作用到 compressed global key code；目前依赖 causal mask 和 token/position hidden 表达，后续需要评估是否需要额外 global position encoding。
- 当前 smoke batch=1，只证明路径可运行；正式训练配置需要重新定 batch、checkpoint benchmark、保留策略。

## 下一步建议

1. 做 P0 diagnostics：suffix invariance 已覆盖，下一步补 full-forward vs incremental。
2. 加一个 short-run 1k/5k 配置，先看 loss 是否能下降、global mass 是否健康。
3. 如果 full/incremental mismatch 明显，再决定是否改 chunkwise/token-scan training。
4. 如果 loss 能下降，再做 public S600 checkpoint sweep，避免只看 val loss。

## 2026-06-24 追加：selected-query 执行优化

### 动机

初版 pure factorized global 在 tokenwise routing 下仍然沿用 attention global 的 full-sequence block 执行：

- 某个 block 只要被任意 token 选中，就对完整 `[B, S]` sequence 计算 attention/FFN。
- 之后再用 token mask 取出 routed token 的输出和 write KV。
- 这和非 global sparse 优化前的问题一致，会保留大量对 loss 没贡献的 activation。

实测初版：

| Batch | Result | Train speed after warmup | Train peak memory |
| --- | --- | ---: | ---: |
| 1 | pass | ~20.5k tokens/s | ~5.2GB |
| 8 | pass | ~17.6k tokens/s | ~90-93GB |
| 16 | OOM | n/a | B200 183GB 打满 |

### 实现

新增 pure factorized 专用 selected-query path：

- `CausalSelfAttention.forward_selected_attention_global`
- `TransformerBlock.forward_selected_attention_global`
- `RouteBlock.forward_selected_attention_global`
- `BrianRouteCore._apply_routed_blocks_with_pure_factorized_attention_global_selected`

语义保持：

- Q/attention/FFN 只对当前 block 实际 routed token 计算。
- 当前 step 的 compressed key/value code 仍对完整 sequence 计算，作为 selected query 的 causal KV pool。
- previous latest global pool 的 causal mask 保持不变。
- top-1 与 top-2 weighted fusion 均走 selected path。
- `attention_global_kv_last_token_mass` 仍按旧 full-sequence 口径计算 last-token metric，但只做轻量 metric 计算，不回退到 full attention。

### 等价性验证

新增测试：

- `test_pure_factorized_attention_global_selected_matches_full_sequence_reference`
- `test_pure_factorized_attention_global_selected_matches_full_sequence_weighted_fusion`

验证内容：

- 同一权重、同一 hidden/position、同一 attention global state 下，新 selected-query path 与旧 full-sequence reference 的 `hidden/write_key/write_value/write_valid` 逐项 `allclose(atol=1e-5, rtol=1e-5)`。
- 覆盖 top-1 和 top-2 weighted fusion。

额外验证：

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_config_inventory.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m compileall -q src/brian_sphere_llm/model
```

结果：

```text
21 passed
21 passed
compileall passed
```

CUDA bf16 backward smoke：

```text
pure_factorized_cuda_backward_ok
```

### B200 smoke 性能

配置基础：

- `configs/train/corrected_attention_global_kv_pure_factorized_r125_5b_smoke.yaml`
- sequence length: 2048
- max steps: 3
- single GPU: GPU4
- route mode: 当前 smoke 下 `weighted_fusion_ratio=0.0`，因此主要测 top-1 tokenwise routing。

| Batch | Result | Step 2 | Step 3 | Warmup 后均值 | Train peak memory | Eval speed | Eval peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | pass | 16,298 tok/s | 17,167 tok/s | 16,733 tok/s | 53.6GB | 25,931 tok/s | 11.4GB |
| 16 | pass | 14,586 tok/s | 14,954 tok/s | 14,770 tok/s | 126.4GB | 23,245 tok/s | 20.6GB |

结论：

- 显存优化有效：bs=8 从约 90-93GB 降到约 52-54GB；bs=16 从 OOM 变为可运行，峰值约 126GB。
- 速度没有明显改善：selected-query 减少了无用 activation 和 mask 尺寸，但引入 per-block gather/scatter/index_add，小 batch/短 smoke 下吞吐仍在 15-17k tokens/s 区间。
- 当前版本可以作为 bs=16 的可训练原型；若要进一步提速，下一步应优先做 route-step/block 级 compile 或 activation checkpointing/自定义 fused scatter，而不是再改变 global KV 语义。

## 2026-06-24 追加：top1_fast 执行路径与更底层优化判断

### 目标

在不改变现有 pure factorized Attention Global KV 语义的前提下，继续检查是否能像非 Global grouped/gather 路径那样，把 routing 执行做得更底层。

本轮先实现低风险 opt-in fast path，目的不是替换默认配置，而是回答两个问题：

- 外层 weighted-fusion/index-add 是否是当前速度瓶颈？
- 是否值得继续推进 grouped-selected 或 fused kernel 级别实现？

### 代码调整

新增配置字段：

- `attention_global_route_execution: selected`
- `attention_global_route_execution: top1_fast`

默认仍为 `selected`，保持现有配置行为不变。`top1_fast` 只在以下条件同时满足时启用：

- `attention_global_kv: true`
- `attention_global_kv_mode: pure_factorized`
- 当前 route step 没有 top-2 weighted fusion

新增文件：

- `configs/model/brian_r125_sphere16_no_location_bias_attention_global_kv_pure_factorized_top1_fast.yaml`
- `configs/train/corrected_attention_global_kv_pure_factorized_top1_fast_r125_5b_smoke.yaml`

新增实现：

- `BrianRouteConfig.attention_global_route_execution`
- `BrianRouteCore._apply_routed_blocks_with_pure_factorized_attention_global_top1_fast`

`top1_fast` 与 `selected` 的主要差异：

- 跳过 weighted-fusion 通用分支。
- 用 flat `index_copy_` / `index_fill_` 回填 selected token。
- write key/value 的 invalid slots 不再初始化为 0；这些位置由 `write_valid=false` 屏蔽，语义上不可见。
- 外层不再用 `torch.any(...).item()` 判断是否 top-1，避免每个 route step 的 GPU/CPU 同步。

### 验证

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_config_inventory.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m compileall src/brian_sphere_llm
```

结果：

```text
22 passed
21 passed
compileall passed
```

新增等价测试：

- `test_pure_factorized_attention_global_top1_fast_matches_selected`

等价口径：

- hidden exact/allclose。
- write_valid exact。
- write key/value 只比较 `write_valid=true` 的 payload；invalid slots 在 fast path 中未初始化，但不会进入 cache 语义。

CUDA bf16 smoke：通过。`top1_fast` 可完成 forward/backward，梯度 finite。

### 同机 B200 smoke 对照

配置基础：

- sequence length: 2048
- max steps: 3
- single GPU: GPU4
- `weighted_fusion_ratio=0.0`
- 对照使用同一代码状态下的默认 `selected` 与新 `top1_fast`

| Mode | Batch | Step 2 | Step 3 | Warmup 后均值 | Train peak memory | Eval speed | Eval peak memory | Val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 8 | 16,194 tok/s | 17,021 tok/s | 16,608 tok/s | 52.3GB | 25,789 tok/s | 11.1GB | 100.080 |
| top1_fast | 8 | 16,415 tok/s | 17,262 tok/s | 16,838 tok/s | 53.1GB | 25,237 tok/s | 11.1GB | 100.083 |
| selected | 16 | 13,578 tok/s | 14,951 tok/s | 14,264 tok/s | 123.4GB | 23,298 tok/s | 20.2GB | 99.175 |
| top1_fast | 16 | 13,583 tok/s | 15,027 tok/s | 14,305 tok/s | 124.9GB | 23,352 tok/s | 20.2GB | 99.167 |

### 结论

`top1_fast` 是 correctness-safe 的 opt-in 实验路径，但不是足够强的性能方案：

- bs=8 约 +1.4%。
- bs=16 约 +0.3%。
- 训练显存没有下降，反而略高。

这说明当前瓶颈不主要在外层 weighted-fusion/index-add/scatter，而在 per-block selected attention 仍然碎片化：

- 每个 route step 仍按 8 个 route blocks 循环。
- 每个 block 都要做 full-sequence position bias、RMSNorm、global K/V write projection。
- `forward_selected_attention_global` 内部仍按 block 构造 padded selected-query SDPA。
- selected-query path 仍有动态 selected count/padding 相关调度成本。

### 更底层优化是否可行

可行，但不能直接照搬非 Global `grouped_dense` 的全 expert 输出方案。

不建议的方案：

- 对 pure Global 直接计算 `E x B x S x D` 全 expert 输出再 gather。这样会把 Q/attention/FFN 也放大到 8 experts，global attention 的 K 长度本来就更大，显存和速度风险都高。

更合理的下一步：

- `grouped-selected attention global`：把 8 个 experts 的 selected-query attention 合并成 packed/grouped 调用，只对被选 token 做 Q/attention/FFN。
- grouped K/V write：把 8 个 block 的 position bias、RMSNorm、global key/value write projection 合并成 expert 维度，减少小 kernel 和 Python loop。
- 如果 PyTorch grouped/padded 版本仍不够快，再做 Triton/CUDA fused kernel，把 expert GEMM、selected attention、write payload scatter 融合。

当前判断：

- `top1_fast` 可以保留为安全实验后端。
- 它不应作为正式训练默认。
- 若继续追求 Global 训练速度，下一轮应直接实现 grouped-selected/fused path，并以 selected 当前结果作为 baseline；继续微调 top1_fast 没有足够收益。

## 2026-06-24 追加：active-pair grouped-selected B 方案

### 目标

在 `top1_fast` 证明外层 scatter 不是主瓶颈后，继续实现真正的 B 路径：把 pure factorized Attention Global KV 的 top-1 selected route step 按真实 `(expert, batch)` active pair 分组执行，避免原 `selected` 路径中按 action 循环造成的 full-batch padding 浪费。

### 实现

新增配置：

- `attention_global_route_execution: grouped_selected`

新增正式配置：

- `configs/model/brian_r125_sphere16_no_location_bias_attention_global_kv_pure_factorized_grouped_selected.yaml`
- `configs/train/corrected_attention_global_kv_pure_factorized_grouped_selected_r125_5b_smoke.yaml`

执行条件：

- 只支持 `attention_global_kv_mode: pure_factorized`。
- 只接管 top-1 route step。
- 如果当前 step 有 top-2 / weighted fusion，自动回退到默认 `selected` 路径。
- 默认配置仍为 `selected`，现有资产不受影响。

核心变化：

- 原 `selected`：按 action 循环，每个 action 的 selected-query attention 仍 padded 到 `[batch, max_selected]`。
- 中间失败版 grouped：padded 到 `[expert, batch, max_selected]`，bs8 显存升到约 136.7GB，训练只有约 12.0k tok/s，已废弃。
- 当前 active-pair grouped：只为真实非空 `(expert, batch)` pair 建 group，固定路由下 bs8 从 64 个 padded groups 降到 8 个 active groups。
- Q/out/FFN 权重保持 expert 维度做 grouped einsum，不按 token 展开权重，避免 `N x hidden_dim x d_model` 级别显存爆炸。

语义说明：

- hidden 输出和 write key/value 的 valid payload 与 `selected` 路径等价。
- write invalid slots 未定义，但由 `write_valid=false` 屏蔽，不进入 cache 语义。
- `attention_global_kv_*` metrics 在 grouped 路径按 active `(expert,batch)` group 记录；它用于诊断日志，不参与 loss。

### 验证

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_config_inventory.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m compileall src/brian_sphere_llm
```

结果：

```text
23 passed
21 passed
compileall passed
```

新增测试：

- `test_pure_factorized_attention_global_grouped_selected_matches_selected`

CUDA bf16 backward smoke：

```text
grouped_selected_active_cuda_backward_ok
```

### B200 smoke 性能

配置基础：

- sequence length: 2048
- max steps: 3
- single GPU: GPU4
- `weighted_fusion_ratio=0.0`
- 对照使用同一代码状态下的默认 `selected` 与新 `grouped_selected`

| Mode | Batch | Step 2 | Step 3 | Warmup 后均值 | Train peak memory | Eval speed | Eval peak memory | Val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 8 | 16,334 tok/s | 17,177 tok/s | 16,756 tok/s | 52.3GB | 25,962 tok/s | 11.1GB | 100.075 |
| grouped_selected | 8 | 44,836 tok/s | 45,667 tok/s | 45,252 tok/s | 30.2GB | 58,458 tok/s | 11.4GB | 100.090 |
| selected | 16 | 14,579 tok/s | 14,910 tok/s | 14,744 tok/s | 123.4GB | 23,232 tok/s | 20.2GB | 99.161 |
| grouped_selected | 16 | 50,780 tok/s | 51,475 tok/s | 51,128 tok/s | 58.4GB | 63,226 tok/s | 20.4GB | 99.166 |

相对提升：

- bs=8 train +170.1%，train peak memory -42.4%，eval +125.2%。
- bs=16 train +246.8%，train peak memory -52.7%，eval +172.2%。

### 结论

`grouped_selected` 是目前第一个对 pure factorized Attention Global KV 明确有效的 B 加速路径：

- 它同时提升吞吐并降低训练显存。
- bs16 从 selected 的约 123GB 降到约 58GB，后续正式训练的 batch/并行余量明显更好。
- 该路径仍是 opt-in，不改变默认 `selected` 行为。

限制：

- 目前只加速 top-1 step；top-2 weighted fusion 仍回退。
- 后续如果要覆盖 router/top-2 后期训练，需要实现 grouped weighted-fusion scatter，或者在 route schedule 中明确 top-1/global fast path 的使用窗口。
- 如果继续追求更高速度，下一步应考虑把 active-pair grouped attention/FFN 做 `torch.compile` 或 Triton fused kernel，而不是再回到全 `[expert,batch]` padding。

## 2026-06-24 追加：cache-only true Global KV 语义

### 动机

前一版 `grouped_selected` 虽然已经加速明显，但它仍然在每个 active `(expert,batch)` group 内现算 current full-sequence K/V，并把：

```text
previous cache + current sequence K/V
```

一起作为 attention 的 read pool。

这不符合我们现在确认的目标语义。新的目标是：

```text
route step t 只读 cache_{t-1}
route step t 计算 routed block 输出
route step t 结束后只更新 selected token，得到 cache_t
```

也就是说 Global KV 必须是真 cache，而不是每个 step 临时拼一个 current full-sequence pool。

### 实现

新增配置：

- `attention_global_route_execution: cache_only`

新增正式配置：

- `configs/model/brian_r125_sphere16_no_location_bias_attention_global_kv_pure_factorized_cache_only.yaml`
- `configs/train/corrected_attention_global_kv_pure_factorized_cache_only_r125_5b_smoke.yaml`

新语义：

- routing loop 前：用 shared A writer 对 pre-route hidden 初始化一次 full-sequence latest-token cache。
- route step t：block query 只 attend `cache_{t-1}`。
- route step t 结束：只对本 step routed selected token 的 updated hidden 计算 shared A code，并写回 latest-token cache。
- 不再在 active group 内生成 current full-sequence K/V。
- A writer 不经过 block-specific position adapter / block RMSNorm；block 差异保留在 Q/read/out/FFN 侧。

当前限制：

- `cache_only` 当前只支持 top-1，因此配置强制 `top_k: 1` 和 `later_top_k: 1`。
- top-2 weighted fusion 的 cache-only 版本需要后续单独实现。

### 验证

```bash
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_sparse_route_block_execution.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m pytest tests/test_config_inventory.py -q
PYTHONPATH=src /home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere/bin/python -m compileall src/brian_sphere_llm
```

结果：

```text
24 passed
21 passed
compileall passed
```

新增语义测试：

- `test_pure_factorized_attention_global_cache_only_ignores_current_unselected_hidden`

该测试固定 cache，修改当前 step 中未 selected token 的 hidden，验证 selected token 输出和 write payload 不变，从而证明 cache-only path 不读取 current full-sequence K/V。

CUDA bf16 backward smoke：

```text
cache_only_cuda_backward_ok
```

### B200 smoke 性能

配置基础：

- sequence length: 2048
- max steps: 3
- single GPU: GPU4
- batch 8 / 16
- 对照使用同一代码状态下的 `selected`、`grouped_selected`、`cache_only`

| Mode | Batch | Step 2 | Step 3 | Warmup 后均值 | Train peak memory | Eval speed | Eval peak memory | Val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 8 | 16,304 tok/s | 17,048 tok/s | 16,676 tok/s | 52.3GB | 25,570 tok/s | 11.1GB | 100.096 |
| grouped_selected | 8 | 43,479 tok/s | 45,663 tok/s | 44,571 tok/s | 30.2GB | 58,152 tok/s | 11.4GB | 100.087 |
| cache_only | 8 | 58,191 tok/s | 64,854 tok/s | 61,522 tok/s | 28.6GB | 78,438 tok/s | 11.4GB | 101.024 |
| selected | 16 | 13,507 tok/s | 14,970 tok/s | 14,238 tok/s | 123.4GB | 23,274 tok/s | 20.2GB | 99.172 |
| grouped_selected | 16 | 51,709 tok/s | 52,185 tok/s | 51,947 tok/s | 58.4GB | 63,979 tok/s | 20.4GB | 99.169 |
| cache_only | 16 | 75,951 tok/s | 77,278 tok/s | 76,614 tok/s | 55.3GB | 86,606 tok/s | 20.4GB | 99.788 |

相对 `selected`：

- bs=8：train +268.9%，train peak memory -45.3%，eval +206.8%。
- bs=16：train +438.1%，train peak memory -55.2%，eval +272.1%。

相对 `grouped_selected`：

- bs=8：train +38.0%，train peak memory -5.2%，eval +34.9%。
- bs=16：train +47.5%，train peak memory -5.3%，eval +35.4%。

### 结论

`cache_only` 达到了这轮预期：

- 语义上符合 true Global KV：step 内只读旧 cache，step 末更新 cache。
- 速度上明显超过 `grouped_selected`。
- 显存进一步下降，bs16 训练峰值约 55.3GB。
- 这是目前最适合作为下一轮 Global KV 正式训练的实现。

需要注意：`cache_only` 是语义变化，因此不能直接与旧 `selected/grouped_selected` 的 validation loss 数值逐项等价比较。它更接近原始构想，但需要后续通过长训和 benchmark 判断能力曲线。
