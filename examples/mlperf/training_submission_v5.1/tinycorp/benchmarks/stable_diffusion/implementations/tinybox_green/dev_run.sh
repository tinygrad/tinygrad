#!/bin/bash

# NOTE: The current settings are sufficient to test whether this tinygrad implementation matches the mlperf reference implementation.
#   The approach is to test I/O with each component, ensuring tinygrad/ref implementations match.
#   The focus is on correctness, not speed - I use CPU because my local GPU isn't big enough for holding all the training params.
# TODO: next steps: train to convergence on one big NVIDIA GPU (cloud rented), then on tinybox

# dependencies
# pip install tqdm

export PYTHONPATH="." NV=1

#export MODEL="bert"
export MODEL="stable_diffusion"

#export DEFAULT_FLOAT="HALF" SUM_DTYPE="HALF" GPUS=6 BS=96 EVAL_BS=96
export BS=1 EVAL_BS=1

#export FUSE_ARANGE=1 FUSE_ARANGE_UINT=0

#export BEAM=8 BEAM_UOPS_MAX=10000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
#export IGNORE_JIT_FIRST_BEAM=1

#export BASEDIR="/raid/datasets/wiki"
# TODO: this will change when training on a tinybox
export BASEDIR="/home/hooved/train-sd/training/stable_diffusion/datasets"

#export WANDB=1 PARALLEL=0
export PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py