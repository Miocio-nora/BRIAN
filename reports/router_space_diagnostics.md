# Router Space Diagnostics

Date: 2026-06-19

## Purpose

Route path histograms tell us what blocks were selected, but not why the router
selected them. Router-space diagnostics visualize the router's own scoring
space so we can inspect block expert separation, sample embedding drift, and
single-block domination.

## Space Definition

For each routed decision, the router input is:

```text
[mean(hidden over tokens); current block-position state]
```

The diagnostic projects the router MLP hidden state:

```text
routing_embedding = SiLU(W_in [pooled_hidden; position])
```

The block experts are the rows of the router scoring layer:

```text
logits = routing_embedding @ W_score.T + bias
```

Each row of `W_score` is plotted as the expert vector for one action: route
blocks plus `OUT`.

## What To Watch

- Selected-action domination: one action receives almost every selected sample.
- Raw-top domination: the router's scoring layer itself always prefers one
  action before location bias or constraints.
- Effective entropy collapse: probabilities become near one-hot.
- Dead selected actions: blocks never selected in the sampled routing space.
- Self-recur ratio: repeated selection of the same internal block on consecutive
  steps.
- Expert geometry: very close or highly aligned expert vectors can make routing
  fragile.

## Commands

Generate a router-space report for a completed or running run's latest
checkpoint:

```bash
python scripts/eval.py \
  --config configs/eval/router_space_visualization.yaml \
  --run runs/corrected_global_kv_r125_5b_compressed \
  --checkpoint checkpoint_latest \
  --max-batches 1
```

The default output is:

```text
<run_dir>/router_space_visualization.html
<run_dir>/router_space_visualization.json
```

Training-time upload is controlled by:

```yaml
router_space_visualization:
  enabled: true
  interval: 2500
  max_points: 2048
  upload_to_wandb: true
```

The current Global KV validation configs enable this diagnostic for future
starts. Already-running processes do not reload this setting.

## Current Old Hidden Global KV Example

The collapsed latest checkpoint of
`runs/corrected_global_kv_r125_5b_compressed` reports:

```text
raw_top_action_counts: block 6 = 512 / 512
selected_action_counts: block 6 = 480 / 512, OUT = 32 / 512
effective_entropy_mean: ~0
self_recur_ratio: 0.933
```

This confirms the failure is router saturation and self-recurrent collapse, not
ordinary validation overfitting.
