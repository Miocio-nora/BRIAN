#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${BRIAN_ENV_DIR:-/home/dredvpn009/Flash_Storage/anaconda3/envs/brian-sphere}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"

cd "${ROOT_DIR}"

run_arm() {
  local label="$1"
  local config="$2"
  printf '\n[%s] starting %s on CUDA_VISIBLE_DEVICES=%s\n' "$(date --iso-8601=seconds)" "${label}" "${CUDA_VISIBLE_DEVICES}"
  "${ENV_DIR}/bin/torchrun" --standalone --nproc_per_node=2 \
    scripts/train.py --config "${config}"
  printf '[%s] completed %s\n' "$(date --iso-8601=seconds)" "${label}"
}

run_arm \
  "Q1 CPBC-DP C128/U1 250M" \
  "configs/train/q1_cpbc_r125_250m_dp_u1_c128_triton_ddp2_legacyval.yaml"
run_arm \
  "Q1 CPBC-FB C128/U1 250M" \
  "configs/train/q1_cpbc_r125_250m_fb_u1_c128_triton_ddp2_legacyval.yaml"
run_arm \
  "Q1 matched Transformer baseline 250M" \
  "configs/train/q1_baseline_r125_250m_ddp2_legacyval.yaml"
