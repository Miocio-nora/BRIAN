#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${BRIAN_ENV_DIR:-/nvmesv/dredvpn009/anaconda3/envs/brian-sphere}"
RUN_DIR="${Q10_RUN_DIR:-runs/q10_cpbc_r125_5b_dp_u1_c2048_per_head_d64_ddp2_legacyval}"
OUT_DIR="${RUN_DIR}/benchmarks"

REASONING_GPU="${REASONING_GPU:-0}"
PUBLIC_GPU="${PUBLIC_GPU:-1}"
CHECKPOINT_STEPS=(00015000 00030000 00045000 00060000 00075000 00076294)

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

run_reasoning_matrix() {
  local gpu="$1"
  local step checkpoint output
  for step in "${CHECKPOINT_STEPS[@]}"; do
    checkpoint="checkpoint_step_${step}"
    output="${OUT_DIR}/step${step}_reasoning_s600.json"
    if [[ -s "${output}" ]]; then
      printf '[%s] skip existing reasoning %s\n' "$(date --iso-8601=seconds)" "${checkpoint}"
      continue
    fi
    printf '[%s] reasoning %s on GPU %s\n' "$(date --iso-8601=seconds)" "${checkpoint}" "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_DIR}/bin/python" scripts/eval.py \
      --config configs/eval/reasoning_eval_s600.yaml \
      --run "${RUN_DIR}" \
      --checkpoint "${checkpoint}" \
      --generation-mode batched_incremental \
      --teacher-mode reference \
      --batch-size 64 \
      --output "${output}"
  done
}

run_public_matrix() {
  local gpu="$1"
  local step checkpoint output samples
  for step in "${CHECKPOINT_STEPS[@]}"; do
    checkpoint="checkpoint_step_${step}"
    output="${OUT_DIR}/step${step}_public_s600.json"
    samples="${OUT_DIR}/step${step}_public_s600_samples.jsonl"
    if [[ -s "${output}" && -s "${samples}" ]]; then
      printf '[%s] skip existing public %s\n' "$(date --iso-8601=seconds)" "${checkpoint}"
      continue
    fi
    printf '[%s] public %s on GPU %s\n' "$(date --iso-8601=seconds)" "${checkpoint}" "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_DIR}/bin/python" scripts/public_benchmark.py \
      --config configs/eval/public_benchmark_s600.yaml \
      --run "${RUN_DIR}" \
      --checkpoint "${checkpoint}" \
      --batch-size 1 \
      --output "${output}" \
      --samples-output "${samples}"
  done
}

run_reasoning_matrix "${REASONING_GPU}" >"${OUT_DIR}/reasoning_matrix.log" 2>&1 &
pids=("$!")
run_public_matrix "${PUBLIC_GPU}" >"${OUT_DIR}/public_matrix.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

if [[ "${status}" -ne 0 ]]; then
  printf '[%s] Q10 benchmark matrix failed with status %s\n' "$(date --iso-8601=seconds)" "${status}" >&2
  exit "${status}"
fi

printf '[%s] completed Q10 reasoning/public S600 checkpoint matrix\n' "$(date --iso-8601=seconds)"
