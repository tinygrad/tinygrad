#!/usr/bin/env bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="."
export ROCM_PATH="/opt/rocm-7.1.1"
export PATH="$ROCM_PATH/bin:$PATH"
export DEV=AMD
export CHECK_OOB=0
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000
export DEVICE_IN_FUNCTION_BUG=1
export SCACHE=2

export DEBUG=0
export HK_FLASH_ATTENTION=1
export FP8_FA=1
export ALL2ALL=1
export LATE_ALLREDUCE=0
export USE_ATOMICS=1
export ASM_GEMM=1
export WQKV=1
export MASTER_WEIGHTS=1
export MXFP4=1
export ALLREDUCE_CAST=1
export FAST_CE=1
export FUSED_SILU_W13=1
export SPLIT_W13=0
export OFFLOAD_OPTIM=0

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="float32"
export DP=8 MP=1 BS=16 EVAL_BS=8 GRADIENT_ACC_STEPS=2
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama3"
export SUBMISSION_PLATFORM="tinybox_8xMI350X"
export CONFIG_FILENAME="${BASH_SOURCE[0]}"
export BASEDIR="/raid/datasets/c4-8b/"
export SMALL=1
export LLAMA3_SIZE=8B
export EVAL_TARGET=3.3 EVAL_FREQ=12288
export LR="1e-3" END_LR="1e-4" WARMUP_SAMPLES=2048 MAX_STEPS=1200000
export WARMUP_STEPS=$((WARMUP_SAMPLES / GBS))
export SAMPLES=$((MAX_STEPS * GBS))
export SEQLEN=8192

export SEED=$RANDOM
export DATA_SEED=$SEED

export JITBEAM=3
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0

export LOGMLPERF=1

DATETIME=$(date "+%m%d%H%M")
LOGFILE="llama31_8b_8xMI350x_${DATETIME}_${SEED}.log"

python3 extra/amdpci/hive_reset.py

# beam
FAKEDATA=1 BENCHMARK=10 INITMLPERF=1 python3 examples/mlperf/model_train.py | tee "$LOGFILE"

# run
RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a "$LOGFILE"
