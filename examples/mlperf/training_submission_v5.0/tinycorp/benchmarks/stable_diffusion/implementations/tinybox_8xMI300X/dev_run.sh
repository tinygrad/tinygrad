#!/bin/bash

source venv/bin/activate
# dependencies
#pip install tqdm, numpy, ftfy, regex, pillow, scipy, wandb
# webdataset depends on torch.utils.data.DataLoader
#pip install --index-url https://download.pytorch.org/whl/cpu torch
#pip install webdataset
pip list
apt list --installed | grep amdgpu
rocm-smi --version
modinfo amdgpu | grep version

export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 IGNORE_JIT_FIRST_BEAM=1 HCQDEV_WAIT_TIMEOUT_MS=300000
HCQDEV_WAIT_TIMEOUT_MS=300000
export AMD_LLVM=0 # bf16 seems to require this

export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"
export MODEL="stable_diffusion" PYTHONPATH="."

export GPUS=8 BS=304
export CONTEXT_BS=816 DENOISE_BS=600 DECODE_BS=384 INCEPTION_BS=560 CLIP_BS=240
export WANDB=1
export PARALLEL=0

DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_mi300x_${DATETIME}.log"
export UNET_CKPTDIR="$HOME/stable_diffusion/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR
#export RESUME_CKPTDIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09100305"
#export RESUME_ITR=15240

sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setperfdeterminism 1500
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setpoweroverdrive 400
TOTAL_CKPTS=10 LEARNING_RATE="2.5e-7" RUNMLPERF=1 python3 examples/mlperf/model_train.py && \
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setpoweroverdrive 750 && \
EVAL_CKPT_DIR=$UNET_CKPTDIR python3 examples/mlperf/model_eval.py
