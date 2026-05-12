#!/usr/bin/env bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="."
export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export MODEL="llama2_70b_lora"
export NULL_ALLOW_COPYOUT=1
export OFFLOAD_OPTIM=${OFFLOAD_OPTIM:-0}
export MASTER_WEIGHTS=${MASTER_WEIGHTS:-1}
export LORA=1

export HK_FLASH_ATTENTION=${HK_FLASH_ATTENTION:-1}
export LATE_ALLREDUCE=${LATE_ALLREDUCE:-0}
export USE_ATOMICS=${USE_ATOMICS:-1}
export ASM_GEMM=${ASM_GEMM:-1}
export ALLREDUCE_CAST=${ALLREDUCE_CAST:-1}
export FAST_CE=${FAST_CE:-1}
export FUSED_INPUT_QUANTIZE=${FUSED_INPUT_QUANTIZE:-1}
export FUSED_ADD_NORM_MUL_QUANTIZE=${FUSED_ADD_NORM_MUL_QUANTIZE:-1}
export FUSED_SILU_W13=${FUSED_SILU_W13:-1}
export FUSED_PAD_GRAD_ACCUM=${FUSED_PAD_GRAD_ACCUM:-1}

export DEV="${DEV:-AMD}"
export BS="${BS:-8}"
export MP="${MP:-1}"
export DP="${DP:-1}"
export FSDP="${FSDP:-1}"
export VIZ=${VIZ:-0}
export JITBEAM=${JITBEAM:-3}
export WANDB=${WANDB:-0}
export WANDB_PROJ='MLPerf-llama2_70b_lora'
export GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-8}

export ALL2ALL=${ALL2ALL:-1}
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

export GBS=$((BS * GRADIENT_ACC_STEPS))

export LR=${LR:-4e-4}
export END_LR=${END_LR:-0}
export MAX_STEPS=${MAX_STEPS:-1024}
export WARMUP_STEPS=${WARMUP_STEPS:-1}
export SAMPLES=${SAMPLES:-$((MAX_STEPS * GBS))}

export EVAL_BS=${EVAL_BS:-8}
export EVAL_FREQ=${EVAL_FREQ:-384}

export ADAM_BETA_1=${ADAM_BETA_1:-0.9}
export ADAM_BETA_2=${ADAM_BETA_2:-0.999}
export ADAM_EPSILON=${ADAM_EPSILON:-1e-8}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.3}

export EVAL_TARGET="${EVAL_TARGET:-0.925}"
export LOAD_MODEL="${LOAD_MODEL:-0}"
export FAKEDATA="${FAKEDATA:-1}"
export EVAL_SAMPLES=173
export CPU_DISK_LOAD=1

python3 -u examples/mlperf/model_train.py