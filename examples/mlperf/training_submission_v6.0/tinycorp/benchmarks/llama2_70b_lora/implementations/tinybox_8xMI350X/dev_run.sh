#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../../../../" && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export DEV="${DEV:-AMD}"
export EMULATE="${EMULATE:-AMD_CDNA4}"
export CHECK_OOB="${CHECK_OOB:-0}"
export REWRITE_STACK_LIMIT="${REWRITE_STACK_LIMIT:-5000000}"
export HCQDEV_WAIT_TIMEOUT_MS="${HCQDEV_WAIT_TIMEOUT_MS:-240000}"
export DEVICE_IN_FUNCTION_BUG="${DEVICE_IN_FUNCTION_BUG:-1}"

export DEBUG="${DEBUG:-0}"
export HK_FLASH_ATTENTION="${HK_FLASH_ATTENTION:-1}"
export ALL2ALL="${ALL2ALL:-1}"
export USE_ATOMICS="${USE_ATOMICS:-0}"
export ASM_GEMM="${ASM_GEMM:-1}"
export WQKV="${WQKV:-1}"
export OFFLOAD_OPTIM="${OFFLOAD_OPTIM:-1}"

export DEFAULT_FLOAT="${DEFAULT_FLOAT:-bfloat16}"
export OPTIM_DTYPE="${OPTIM_DTYPE:-bfloat16}"
export DP="${DP:-1}"
export MP="${MP:-8}"
export BS="${BS:-1}"
export EVAL_BS="${EVAL_BS:-1}"
export GRADIENT_ACC_STEPS="${GRADIENT_ACC_STEPS:-1}"
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama2_70b_lora"
export DATASET_PATH="${DATASET_PATH:-/raid/datasets/scrolls_gov_report_8k}"
if [[ ! ${MODEL_PATH+x} ]]; then export MODEL_PATH="/raid/weights/llama2-70b-fused-qkv-mlperf"; fi
if [[ -z "${FAKEDATA:-}" && -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must point to converted Llama2 70B weights for a real benchmark run" >&2
  exit 1
fi
export SEQLEN="${SEQLEN:-8192}"

export EVAL_TARGET="${EVAL_TARGET:-0.925}"
export EVAL_FREQ="${EVAL_FREQ:-48}"
export LR="${LR:-4e-4}"
export END_LR="${END_LR:-0.0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export MAX_STEPS="${MAX_STEPS:-1024}"

export LLAMA_LORA_RANK="${LLAMA_LORA_RANK:-16}"
export LLAMA_LORA_ALPHA="${LLAMA_LORA_ALPHA:-32}"
export LLAMA_LORA_DROPOUT="${LLAMA_LORA_DROPOUT:-0.1}"

export SEED="${SEED:-1234}"
export DATA_SEED="${DATA_SEED:-${SEED}}"

export JITBEAM="${JITBEAM:-3}"
export BEAM_UOPS_MAX="${BEAM_UOPS_MAX:-6000}"
export BEAM_UPCAST_MAX="${BEAM_UPCAST_MAX:-256}"
export BEAM_LOCAL_MAX="${BEAM_LOCAL_MAX:-1024}"
export BEAM_MIN_PROGRESS="${BEAM_MIN_PROGRESS:-5}"
export BEAM_PADTO="${BEAM_PADTO:-1}"

python3 "${REPO_ROOT}/examples/mlperf/model_train.py"
