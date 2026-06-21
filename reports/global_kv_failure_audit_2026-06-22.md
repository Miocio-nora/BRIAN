# Global KV Failure Audit - 2026-06-22

This audit investigates why the balanced Global KV runs reached very low validation loss while public/synthetic benchmark quality collapsed.

Runs audited:

- Hidden-state Global KV: `runs/corrected_global_kv_r125_5b_balanced_slow_noise`
- Attention-level Global KV: `runs/corrected_attention_global_kv_r125_5b_balanced_slow_noise`

## Executive Conclusion

The Global KV series does show real abnormalities. The failure is not explained by a simple route collapse at the first benchmark collapse point.

The main failure mode is global-memory shortcut overfitting:

- The hidden-state Global KV path becomes too strong: at 45k, `global_to_local_read_ratio` reaches `4.70`, while reasoning exact match is already `0.000`.
- The hidden-state `global_read.query` matrix norm explodes relative to the write projection: query/write spectral ratio rises from `14.5x` at 5k to `39.6x` at 45k and `66-73x` later.
- The attention Global KV path learns to open the global prefix: mean `global_logit_bias` moves from `-4` initialization to positive by 30k, with last-token global attention mass around `0.37-0.40` after 45k.
- Disabling or emptying the global channel at failed checkpoints does not recover benchmark quality, which means the model has co-adapted to the global path instead of retaining a healthy local fallback.
- The hidden-state implementation also has a concrete write-mask risk: it writes hidden global cache entries after route steps even for samples that have already exited, unlike the attention Global KV path, which filters `~exited` and `selected != out_action`.

## Benchmark Timeline

Balanced hidden-state Global KV:

| Step | PPL | Reason exact | Teacher acc | Public avg | Global/local read | Gate | Sink mass | Window mass | Route entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15k | 11.332 | 0.735 | 0.918 | 0.372 | 0.429 | 0.300 | 0.998 | 0.002 | 1.624 |
| 30k | 9.372 | 0.622 | 0.914 | 0.362 | 0.826 | 0.452 | 0.924 | 0.076 | 1.730 |
| 45k | 6.523 | 0.000 | 0.275 | 0.358 | 4.701 | 0.825 | 0.317 | 0.683 | 1.876 |
| 60k | 4.613 | 0.000 | 0.151 | 0.342 | 4.255 | 0.810 | 0.304 | 0.696 | 1.750 |
| 75k | 44.271 | 0.008 | 0.409 | 0.322 | 3.346 | 0.770 | 0.580 | 0.420 | ~0.000 |

Balanced attention Global KV:

| Step | PPL | Reason exact | Teacher acc | Public avg | Global attn mass | Bias mean | Sink mass | Window mass | Route entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15k | 11.249 | 0.715 | 0.918 | 0.372 | 0.076 | -0.871 | 0.064 | 0.012 | 1.595 |
| 30k | 9.030 | 0.553 | 0.885 | 0.365 | 0.252 | 0.339 | 0.202 | 0.050 | 1.696 |
| 45k | 7.303 | 0.347 | 0.874 | 0.373 | 0.377 | 0.522 | 0.288 | 0.090 | 1.745 |
| 60k | 4.450 | 0.023 | 0.376 | 0.340 | 0.386 | 0.620 | 0.296 | 0.090 | 1.357 |
| 75k | 3.496 | 0.003 | 0.335 | 0.323 | 0.396 | 0.553 | 0.303 | 0.093 | 1.263 |

Loss/PPL keeps improving while reasoning quality collapses. This confirms that checkpoint selection cannot rely on validation loss alone for Global KV.

## Implementation Findings

### Hidden-State Global KV

Current behavior:

- `GlobalWriteAdapter` pools hidden states by sequence mean, projects to `global_code_dim=64`, and normalizes the code to unit norm.
- `GlobalReadAdapter` pools hidden states by sequence mean, projects a query, softmaxes over cached codes, maps the selected code back to hidden size, and adds `sigmoid(gate) * read_hidden` to every token.
- Global read happens before every route decision and again after the exit block.
- Global write happens after every routed step.

Concrete risk:

- Hidden Global KV writes after every routed step without filtering already exited samples.
- Attention Global KV does filter writes with `write_valid = write_valid & ~exited & (selected != self.out_action)`.
- This asymmetry can pollute hidden Global KV with stale/exited hidden states, especially after hard exit.

### Attention Global KV

Current behavior:

- Attention Global KV is attention-level in the sense that route-block attention directly attends to cached K/V prefix slots.
- Each write is still a compressed sequence summary: `key_summary = k.mean(dim=2)` and `value_summary = v.mean(dim=2)`, so it is not full token-level KV reuse.
- Global slots are concatenated before local causal K/V.
- A learned per-block scalar `global_logit_bias` controls global-prefix preference.

Concrete risk:

- `global_logit_bias` is the only attention-specific global control and it is unconstrained after initialization.
- Once the bias turns positive, the model can route a large fraction of attention through a tiny set of compressed global slots.

## Parameter Diagnostics

Hidden Global KV parameter norms:

| Step | Gate | Read query spec | Write spec | Query/write | Gate x read-out spec | Emb row norm |
|---:|---:|---:|---:|---:|---:|---:|
| 5k | 0.082 | 15.324 | 1.056 | 14.5x | 0.352 | 23.837 |
| 15k | 0.300 | 24.569 | 1.091 | 22.5x | 1.086 | 17.663 |
| 30k | 0.452 | 21.759 | 1.168 | 18.6x | 1.256 | 11.274 |
| 45k | 0.825 | 51.409 | 1.299 | 39.6x | 2.794 | 7.216 |
| 60k | 0.810 | 80.617 | 1.210 | 66.6x | 2.217 | 4.681 |
| 75k | 0.770 | 85.953 | 1.174 | 73.2x | 1.960 | 3.236 |
| final | 0.762 | 82.684 | 1.204 | 68.7x | 1.896 | 3.138 |

The write projection and read-out projection are not the main explosion. The abnormal parameter is the read query projection, combined with a large gate.

Attention Global KV parameter norms:

| Step | Bias mean | Bias min | Bias max | Global attn mass | Emb row norm |
|---:|---:|---:|---:|---:|---:|
| 5k | -2.901 | -3.265 | -2.397 | 0.001 | 23.837 |
| 15k | -0.870 | -1.475 | 0.057 | 0.076 | 17.663 |
| 30k | 0.337 | -0.568 | 0.962 | 0.252 | 11.273 |
| 45k | 0.696 | -0.118 | 1.857 | 0.377 | 7.247 |
| 60k | 0.666 | 0.109 | 1.325 | 0.386 | 4.731 |
| 75k | 0.579 | 0.141 | 0.967 | 0.396 | 3.174 |
| final | 0.575 | 0.142 | 0.945 | 0.369 | 3.073 |

The embedding norm drop is not Global KV-specific. A 5B baseline and non-global route-core runs also end near row norm `3.1`, likely due to the high global `weight_decay=0.1`. It can still amplify the relative impact of global reads, but it is not itself the global-specific bug.

## Targeted Ablations

Same checkpoints, same 600-sample synthetic reasoning suite.

| Run/checkpoint | Intervention | Exact | Teacher acc | Interpretation |
|---|---|---:|---:|---|
| hidden 45k | normal | 0.000 | 0.275 | Failed checkpoint. |
| hidden 45k | gate forced near zero | 0.002 | 0.168 | Turning off global does not recover; local path has co-adapted. |
| hidden 45k | sink-only | 0.000 | 0.014 | Recent window was carrying most remaining usable signal. |
| hidden 45k | last-4 window only | 0.000 | 0.272 | Very close to normal; early sink is not the main cause here. |
| attention 60k | normal | 0.023 | 0.376 | Failed checkpoint. |
| attention 60k | global bias forced to `-20` | 0.000 | 0.000 | Model is dependent on global attention by this stage. |
| attention 60k | sink setting changed but all slots retained | 0.023 | 0.376 | Not a meaningful causal change because actual slots are below full retention capacity. |
| attention 60k | last-4 window only | 0.015 | 0.308 | Removing early slots does not improve quality. |

The sink slots are heavily attended in the attention run, but the collapse is not fixed by removing early/sink retention. The more general issue is over-reliance on compressed global memory.

## Output Pattern

At successful early checkpoints, synthetic tasks mostly emit structured numeric/symbolic answers. At failed checkpoints, outputs shift to natural-language fragments such as repeated FineWeb-like phrases. This is a qualitative sign of conditional task failure, not only benchmark noise.

Examples:

- Hidden 15k: reverse/copy/rewrite samples are often exact.
- Hidden 45k: generated answers include fragments like `physical pione physical...`, `along6 crucial discovery...`.
- Attention 45k: still partially structured.
- Attention 60k: often emits broad natural-language fragments despite lower validation loss.

## What This Means

The Global KV mechanism has potential because early checkpoints are strong:

- Hidden 15k: reasoning exact `0.735`, teacher acc `0.918`.
- Attention 15k: reasoning exact `0.715`, teacher acc `0.918`.

But the current unconstrained Global KV path is unstable. It gives the model an easier training-loss route than robust conditional reasoning. The failure is especially dangerous because validation loss improves through the collapse.

## Recommended Fixes Before More Long Runs

1. Fix hidden Global KV write masking.
   - Do not write hidden global cache entries for already exited samples.
   - Do not write when `selected == out_action`.

2. Add explicit global strength control.
   - Hidden: cap or scale `global_read_gate`, for example max gate `0.20-0.30`.
   - Hidden: add a penalty on `global_to_local_read_ratio`, target initially below `0.5-1.0`.
   - Hidden: regularize or normalize `global_read.query`; consider spectral norm, query norm clamp, or query/code temperature.
   - Attention: add global attention mass penalty or bias cap.
   - Attention: keep `global_logit_bias` upper-bounded during early training, e.g. do not allow it to become positive until benchmarks are stable.

3. Make global memory less shortcut-like.
   - Add global slot dropout during training.
   - Randomly disable global reads on a small fraction of steps, so the local path remains viable.
   - Consider stop-gradient or delayed global write/read warmup.
   - For attention Global KV, replace single sequence-mean K/V summaries with a more faithful token/top-token summary if we want true attention-level reuse.

4. Keep benchmark-driven checkpoint selection.
   - Use the 600-sample synthetic reasoning suite and public benchmark during training.
   - Track `global_to_local_read_ratio`, hidden query norm, attention global mass, and global bias in checkpoint reports.
   - Select checkpoints by benchmark quality, not validation loss.

5. Next experiments should be short and diagnostic.
   - Hidden Global KV: write-mask fix + gate cap + ratio penalty.
   - Attention Global KV: bias cap/mass penalty + slot dropout.
   - Stop before 45k/60k unless benchmark curves remain stable.

## Artifacts

Diagnostic outputs:

- `reports/global_kv_diagnostics/parameter_norms.json`
- `reports/global_kv_diagnostics/hidden45_gateoff_reasoning_s600.json`
- `reports/global_kv_diagnostics/hidden45_sinkonly_reasoning_s600.json`
- `reports/global_kv_diagnostics/hidden45_last4_reasoning_s600.json`
- `reports/global_kv_diagnostics/attention60_biasoff_reasoning_s600.json`
- `reports/global_kv_diagnostics/attention60_last4_reasoning_s600.json`

The earlier `dropfirst4` diagnostic files are equivalent to almost empty-cache tests because that monkey patch deleted slots after each write. They should not be used as sink-specific evidence.
