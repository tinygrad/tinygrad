#!/bin/bash

# NOTE: The current settings are sufficient to test whether this tinygrad implementation matches the mlperf reference implementation.
#   The approach is to test I/O with each component, ensuring tinygrad/ref implementations match.
#   The focus is on correctness, not speed - I use CPU because my local GPU isn't big enough for holding all the training params.
# TODO: next steps: train to convergence on one big NVIDIA GPU (cloud rented), then on tinybox

# *** dependencies
#pip install tqdm
#pip install numpy

#### to match mlperf reference clip tokenizer behavior
#pip install ftfy
#pip install regex

## PIL is for validation step: preprocessing the generated images before clip vision encoder encodes the image
#pip install pillow

## for inception, calculating the frechet distance, which uses scipy.linalg
#pip install scipy

#### to use mlperf reference dataloader
#pip install webdataset
#pip install torch # for torch.utils.data.DataLoader, which webdataset depends on
source venv/bin/activate
export DEBUG=2

export PYTHONPATH="."
export MODEL="stable_diffusion"
export BS=1 EVAL_BS=1

export BASEDIR="/home/hooved/stable_diffusion"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints"
mkdir -p $UNET_CKPTDIR

#export WANDB=1 PARALLEL=0
export PARALLEL=0

RUNMLPERF=1 python3 examples/mlperf/model_train.py