#!/usr/bin/env bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="."
export DEV=AMD
export CHECK_OOB=0
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000
export DEVICE_IN_FUNCTION_BUG=1

export HK_FLASH_ATTENTION=1
export ALL2ALL=1
export LATE_ALLREDUCE=0
export USE_ATOMICS=1
export ASM_GEMM=1
export MASTER_WEIGHTS=1
export ALLREDUCE_CAST=1
export FAST_CE=1
export FUSED_INPUT_QUANTIZE=1
export FUSED_ADD_NORM_MUL_QUANTIZE=1
export FUSED_SILU_W13=1
export FUSED_PAD_GRAD_ACCUM=1

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export DP=1 MP=1 FSDP=8 BS=8 EVAL_BS=8 GRADIENT_ACC_STEPS=1
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama2_70b_lora"
export BASEDIR="/raid/datasets/c4-llama2-70b-lora/"
export MODEL_PATH="/raid/weights/c4-llama2-70b-lora/"
export EVAL_TARGET=0.925 EVAL_FREQ=384

export LR="4e-4" END_LR=0
# WARMUP_SAMPLES=4096 MAX_STEPS=1200000

export WARMUP_STEPS=1 
export SAMPLES=$((MAX_STEPS * GBS))

export SEED=$RANDOM
export DATA_SEED=$SEED

export JITBEAM=3
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

export ADAM_BETA_1="0.9" ADAM_BETA_2="0.999" ADAM_EPSILON="1e-8" WEIGHT_DECAY="1e-4" MAX_GRAD_NORM="0.3"

export LOGMLPERF=1

DATETIME=$(date "+%m%d%H%M")
LOGFILE="llama31_70b_lora_8xMI350x_${DATETIME}_${SEED}.log"

# beam
FAKEDATA=1 BENCHMARK=10 INITMLPERF=1 LLAMA_LAYERS=2 python3 examples/mlperf/model_train.py | tee "$LOGFILE"

# run
RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a "$LOGFILE"
