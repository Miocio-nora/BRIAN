# Attention-Level Global KV Design

Date: 2026-06-19

## Purpose

This branch implements the user's intended Transformer-attention Global KV path as
a new mechanism, without replacing the existing hidden-summary `global_kv`
implementation.

The existing `global_kv` path writes compressed hidden-state codes and reads them
back through `GlobalReadAdapter`. The new `attention_global_kv` path writes
attention-level key/value memory and lets route-pool attention attend to that
memory directly.

## Scope

The first implementation is route-only:

- pre blocks use normal local causal attention;
- route-pool blocks can attend to attention-level global K/V memory;
- post blocks use normal local causal attention;
- parallel passing is rejected for this mode until the branch memory semantics are
  defined.

This matches the current research question: whether free routing benefits from a
shared memory across routed block steps, not whether every fixed Transformer
layer should share K/V.

## Memory Object

`CanonicalAttentionGlobalKVCache` stores:

```text
keys:   [batch, heads, slots, head_dim]
values: [batch, heads, slots, head_dim]
valid:  [batch, slots]
```

Each route block call writes one compressed attention memory slot per sample by
mean-pooling the block attention K/V over the sequence length:

```text
write_key   = mean(local_attention_key over tokens)
write_value = mean(local_attention_value over tokens)
```

This is intentionally compressed: it is not a full raw token-level cache. It
keeps the memory budget controlled while still giving later route steps direct
K/V access inside attention.

## Retention

The cache uses the same simple first-stage policy as the canonical Global KV
plan:

```text
M_global = M_sink + M_window
```

- `attention_global_sink_slots`: first slots retained permanently.
- `attention_global_window_slots`: most recent slots retained as a sliding
  window.

The R125 5B config currently uses sink 4 and window 32.

## Attention Path

Inside a route-pool block, self-attention builds:

```text
K_all = concat(global_keys, local_keys)
V_all = concat(global_values, local_values)
```

The attention mask allows every token to read valid global slots while local
tokens remain causal. A learnable scalar `attention_global_logit_bias` is added
to global slots before attention. It starts negative by default
(`attention_global_logit_bias_init: -4.0`) so the model does not over-read
untrained memory at initialization.

## Metrics

Training summaries now expose:

- `attention_global_kv_slots_mean`
- `attention_global_kv_slots_max`
- `attention_global_kv_write_count_mean`
- `attention_global_kv_logit_bias_mean`
- `attention_global_kv_last_token_mass`
- `attention_global_kv_sink_last_token_mass`
- `attention_global_kv_window_last_token_mass`

The attention-mass metrics are a last-token approximation used as a cheap
diagnostic during training.

## Configs

Main validation config:

```text
configs/train/corrected_attention_global_kv_r125_5b.yaml
```

Smoke config:

```text
configs/train/corrected_attention_global_kv_r125_5b_smoke.yaml
```

Experiment manifest:

```text
configs/experiments/route_core_attention_global_kv_corrected_r125_5b.yaml
```

## Routing Robustness Follow-Up

The initial hidden-summary Global KV run reached strong intermediate loss but
ended in a router collapse: the router saturated into one internal block and
repeated self-routing. The attention-level Global KV run is healthier at early
checkpoints, but it uses the same no-noise route policy and can still form a
small number of repeated path templates.

The follow-up configs add three concrete safeguards:

- `logit_noise_std`, `logit_noise_decay_steps`, and `logit_noise_min_std` add
  slow-decaying Gaussian noise to route logits during training only.
- `random_route_probability`, `random_route_decay_steps`, and
  `random_route_min_probability` override selected internal routes during
  training, forcing the model to tolerate arbitrary routed block execution. This
  is independent of router sampling and disables weighted top-2 fusion for those
  overridden examples.
- `routing.constraints.self_recur_max_consecutive` is a hard cap. Once a sample
  has selected the same internal block too many times in a row, that block is
  masked for the next route decision.

The prepared attention-level follow-up is:

```text
configs/train/corrected_attention_global_kv_r125_5b_slow_noise.yaml
```

A no-router-position attention variant is also available:

```text
configs/train/corrected_attention_global_kv_r125_5b_slow_noise_no_router_position.yaml
```

It keeps position injection inside route blocks but removes position from the
router input. This isolates whether the router is using previous block location
as a shortcut path-state variable instead of scoring from hidden-state content.

## Current Limitations

- Writes one memory token per routed block call only.
- Does not store full token-level K/V.
- Does not support parallel passing yet.
- Does not yet include long-context benchmark evidence.
- Attention-mass logging is approximate and intentionally cheap.

These limits are acceptable for the first implementation because they isolate
the core question: can route-pool blocks use shared attention-level K/V memory
without disturbing the corrected route-core setup.
