#!/bin/bash

# NOTE: The current settings are sufficient to test whether this tinygrad implementation matches the mlperf reference implementation.
#   The approach is to test I/O with each component, ensuring tinygrad/ref implementations match.
#   The focus is on correctness, not speed - I use CPU because my local GPU isn't big enough for holding all the training params.
# TODO: next steps: train to convergence on one big NVIDIA GPU (cloud rented), then on tinybox

# *** dependencies
#pip install tqdm
#pip install numpy

### to match mlperf reference clip tokenizer behavior
#pip install ftfy
#pip install regex

### to use mlperf reference dataloader
#pip install webdataset
#pip install torch # for torch.utils.data.DataLoader, which webdataset depends on

#export PYTHONPATH="." NV=1
export PYTHONPATH="."
export MODEL="stable_diffusion"
export DEFAULT_FLOAT="HALF" BS=1 EVAL_BS=1

# TODO: this will change when training on a tinybox
export BASEDIR="/home/hooved/train-sd/training/stable_diffusion"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints"

#export WANDB=1 PARALLEL=0
export PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py