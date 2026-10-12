#!/usr/bin/env bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

# Run serve.py on the other node: PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667
# This node opens its GPUs directly.
export NODES="tinyamd3 tinyamd4"
export REMOTE="local,$(echo "$NODES" | tr ' ' '\n' | grep -vx "$(hostname)" | head -1):6667"
export DEV=PCI+AMD REMOTE_TIMEOUT=600 ALLREDUCE_NODE_NDEVS=8
export ALLREDUCE_NODE_LEADER=1 RDMA_DUPLEX=1 RDMA_POST_AHEAD=1 COPY_PREHOLD=1000
export LLAMA_STEP_GROUP=1

export PYTHONPATH="."
export ROCM_PATH="/opt/rocm-7.1.1"
export PATH="$ROCM_PATH/bin:$PATH"
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
export DP=16 MP=1 BS=32 EVAL_BS=16 GRADIENT_ACC_STEPS=1
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama3"
export SUBMISSION_PLATFORM="tinybox_2x8xMI350X"
export CONFIG_FILENAME="${BASH_SOURCE[0]}"
export BASEDIR="/raid/datasets/c4-8b/"
export SMALL=1
export LLAMA3_SIZE=8B
export EVAL_TARGET=3.3 EVAL_FREQ=12288
export LR="1e-3" END_LR="1e-4" WARMUP_SAMPLES=2048 MAX_STEPS=1200000
export WARMUP_STEPS=$((WARMUP_SAMPLES / GBS))
export SAMPLES=300000
export SEQLEN=8192

export SEED=$RANDOM
export DATA_SEED=$SEED

export JITBEAM=3 BEAM_STOP_US=12
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0

export LOGMLPERF=1 PYTHONUNBUFFERED=1

# Fresh kernel and compiler caches for each run.
export CACHEDB="$(mktemp -d)/cache.db"
export AMD_COMGR_CACHE_DIR="$(dirname "$CACHEDB")/comgr"

DATETIME=$(date "+%m%d%H%M")
LOGFILE="llama31_8b_2x8xMI350X_${DATETIME}_${SEED}.log"

echo "cache $CACHEDB, seed $SEED, log $LOGFILE" | tee "$LOGFILE"

# Capture on fake batches and restore initial state, then build dataset indices before RUN_START.
INITMLPERF=1 RUNMLPERF=1 PRE_RUN_CAPTURE=1 LLAMA_JIT_NO_WARMUP=1 \
  python3 examples/mlperf/model_train.py 2>&1 | tee -a "$LOGFILE"
