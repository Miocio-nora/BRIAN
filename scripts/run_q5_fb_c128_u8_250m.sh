#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${BRIAN_ENV_DIR:-/nvmesv/dredvpn009/anaconda3/envs/brian-sphere}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"

cd "${ROOT_DIR}"

printf '[%s] starting FB C128/U8 250M on CUDA_VISIBLE_DEVICES=%s\n' \
  "$(date --iso-8601=seconds)" "${CUDA_VISIBLE_DEVICES}"
"${ENV_DIR}/bin/torchrun" --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/train/q5_cpbc_r125_250m_fb_u8_c128_triton_ddp2_legacyval.yaml
printf '[%s] completed FB C128/U8 250M\n' "$(date --iso-8601=seconds)"
