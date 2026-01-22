#!/usr/bin/env bash

export PYTHONPATH="." AMD=1
export IGNORE_OOB=1
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000

export DEBUG=${DEBUG:-0}
export FLASH_ATTENTION=1

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export DP=8 BS=8 EVAL_BS=8

export MODEL="llama3"
export BASEDIR="/raid/datasets/c4-8b/"
export SMALL=1
export LLAMA3_SIZE=${LLAMA3_SIZE:-"8b"}
export EVAL_TARGET=3.3 EVAL_FREQ=12288
export LR="1e-3" END_LR="1e-4" WARMUP_STEPS=1024 MAX_STEPS=1200000
export SAMPLES=$((MAX_STEPS * BS))

export SEED=5760

export JITBEAM=3
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5

python3 examples/mlperf/model_train.py
