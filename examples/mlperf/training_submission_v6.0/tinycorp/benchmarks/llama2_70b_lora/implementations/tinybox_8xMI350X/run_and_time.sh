#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOGMLPERF=1
export MODEL="llama2_70b_lora"
export SUBMISSION_PLATFORM="tinybox_8xMI350X"
export SEED="${SEED:-1234}"
export DATA_SEED="${SEED}"

DATETIME="$(date "+%m%d%H%M")"
LOGFILE="${LOGFILE:-llama2_70b_lora_8xMI350x_${DATETIME}_${SEED}.log}"

# init / beam search warmup on a reduced layer count. Keep benchmark identity and sequence length intact.
env -u RUNMLPERF -u LOAD_CKPT -u RESUME_CKPT MODEL_PATH="" TOKENIZER_PATH="" FAKEDATA=1 BENCHMARK=10 INITMLPERF=1 LLAMA_LAYERS=2 \
  "${SCRIPT_DIR}/dev_run.sh" | tee "${LOGFILE}"

env -u INITMLPERF -u FAKEDATA -u BENCHMARK -u LLAMA_LAYERS -u FULL_LAYERS -u TRAIN_ON_VAL -u SMALL -u LOAD_CKPT -u RESUME_CKPT -u SAMPLES -u EVAL_SAMPLES \
  RUNMLPERF=1 DATASET_PATH="${DATASET_PATH:-/raid/datasets/scrolls_gov_report_8k}" MODEL_PATH="${MODEL_PATH:-/raid/weights/llama2-70b-fused-qkv-mlperf}" \
  MP=8 DP=1 BS=1 EVAL_BS=1 GRADIENT_ACC_STEPS=1 SEQLEN=8192 MAX_STEPS=1024 EVAL_FREQ=48 EVAL_TARGET=0.925 LR=4e-4 END_LR=0.0 WARMUP_STEPS=0 \
  LLAMA_LORA_RANK=16 LLAMA_LORA_ALPHA=32 LLAMA_LORA_DROPOUT=0.1 "${SCRIPT_DIR}/dev_run.sh" | tee -a "${LOGFILE}"
