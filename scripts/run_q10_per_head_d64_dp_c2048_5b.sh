#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${BRIAN_ENV_DIR:-/nvmesv/dredvpn009/anaconda3/envs/brian-sphere}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"

printf '[%s] starting Q10 per-head d64 DP-C2048 5B on CUDA_VISIBLE_DEVICES=%s\n' \
  "$(date --iso-8601=seconds)" "${CUDA_VISIBLE_DEVICES}"
"${ENV_DIR}/bin/torchrun" --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2_legacyval.yaml
printf '[%s] completed Q10 per-head d64 DP-C2048 5B\n' "$(date --iso-8601=seconds)"
