#!/bin/bash

# NOTE: The current settings are sufficient to test whether this tinygrad implementation matches the mlperf reference implementation.
#   The approach is to test I/O with each component, ensuring tinygrad/ref implementations match.
#   The focus is on correctness, not speed - I use CPU because my local GPU isn't big enough for holding all the training params.
# TODO: next steps: train to convergence on one big NVIDIA GPU (cloud rented), then on tinybox

# *** dependencies
#pip install tqdm
#pip install numpy

## to match mlperf reference clip tokenizer behavior
#pip install ftfy
#pip install regex

## to use mlperf reference dataloader
#pip install webdataset
#pip install torch # for torch.utils.data.DataLoader, which webdataset depends on

export PYTHONPATH="." NV=1

export MODEL="stable_diffusion"

# https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#benchmark_specific_rules
# Checkpoint must be collected every 512,000 images. CEIL(512000 / global_batch_size) if 512000 is not divisible by GBS.
export BS=1 EVAL_BS=512000

# TODO: this will change when training on a tinybox
export BASEDIR="/home/hooved/train-sd/training/stable_diffusion"

#export WANDB=1 PARALLEL=0
export PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py