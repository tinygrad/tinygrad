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
apt list --installed | grep amdgpu
#export DEBUG=2
export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1

export BASEDIR="/home/hooved/stable_diffusion"
#export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_red_${DATETIME}_${SEED}.log"
export HCQDEV_WAIT_TIMEOUT_MS=300000
#export BEAM_TIMEOUT_SEC=20

export PYTHONPATH="."
export MODEL="stable_diffusion"

#export RESUME_CKPTDIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09022307"
#export RESUME_ITR=1524
#export BACKUP_INTERVAL=2065
#export BACKUP_INTERVAL=413
#export BACKUP_INTERVAL=640
export BACKUP_INTERVAL=762

# mi300x
# use separate BS for the various jits in eval to maximize throughput
#export JIT=3 # eval takes ~80% longer, but doesn't crash with Bus error

export AMD_LLVM=0 # bf16 seems to require this
export GPUS=8 BS=512
export GRAD_ACC_STEPS=4
export CONTEXT_BS=816
export DENOISE_BS=600
export DECODE_BS=384
export INCEPTION_BS=560
export CLIP_BS=240

# tinybox red
#export GPUS=6 BS=12
#export CONTEXT_BS=600
#export DENOISE_BS=96
#export DECODE_BS=90
#export INCEPTION_BS=480
#export CLIP_BS=84
####export DENOISE_BS=144
####export DECODE_BS=138

export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR
#export RUN_EVAL=1
#export EVAL_ONLY=1
#export EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09022307/run_eval_762"
#export KEEP_EVAL_CACHE=1
#export EVAL_OVERFIT_SET=1
#export EVAL_INTERVAL=2065
#export LIMIT_EVAL_SAMPLES=600

# mi300x
export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"

# tinybox red
#export DATADIR="/home/hooved/stable_diffusion/datasets"
#export CKPTDIR="/home/hooved/stable_diffusion/checkpoints"

#export WANDB=1
#export PARALLEL=4
#export PARALLEL=0
#export PARALLEL=4

#EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09022307/run_eval_762" RUNMLPERF=1 python3 examples/mlperf/model_train.py
RUNMLPERF=1 python3 examples/mlperf/model_train.py