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
