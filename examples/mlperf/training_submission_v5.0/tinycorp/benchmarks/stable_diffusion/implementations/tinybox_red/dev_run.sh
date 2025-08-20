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
#pip install --index-url https://download.pytorch.org/whl/cpu torch # for torch.utils.data.DataLoader, which webdataset depends on
#pip install webdataset
source venv/bin/activate
#export DEBUG=2
#export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
#export IGNORE_JIT_FIRST_BEAM=1

#export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_red_${DATETIME}_${SEED}.log"
export HCQDEV_WAIT_TIMEOUT_MS=300000
#export BEAM_TIMEOUT_SEC=20

export PYTHONPATH="."
export MODEL="stable_diffusion"
#export GPUS=8 BS=248
export GPUS=6

#export EVAL_BS=192
# use separate BS for the various jits in eval to maximize throughput
export CONTEXT_BS=6
export DENOISE_BS=6
export DECODE_BS=6
export INCEPTION_BS=6
export CLIP_BS=6

export RUN_EVAL=1
export EVAL_OVERFIT_SET=1
#export EVAL_INTERVAL=4000
export EVAL_ONLY=1
export EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/08142156/run_eval_88000"

export BASEDIR="/home/hooved/stable_diffusion"
export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR

export WANDB=1
#export PARALLEL=16

RUNMLPERF=1 python3 examples/mlperf/model_train.py