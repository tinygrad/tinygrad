#!/bin/bash

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
#export DEBUG=2
export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1

#export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_red_${DATETIME}_${SEED}.log"
export HCQDEV_WAIT_TIMEOUT_MS=300000
#export BEAM_TIMEOUT_SEC=20

export PYTHONPATH="."
export MODEL="stable_diffusion"
export GPUS=6 BS=12 EVAL_BS=36
#export GPUS=1 BS=1 EVAL_BS=1

#export RUN_EVAL=1
#export EVAL_ONLY=1
#export EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/08051713/run_eval"
#export EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/08051713/run_eval_2"

export BASEDIR="/home/hooved/stable_diffusion"
export DATADIR="/raid/datasets/stable_diffusion"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR

export WANDB=1
export PARALLEL=32

RUNMLPERF=1 python3 examples/mlperf/model_train.py