#!/bin/bash
for i in {0..7}; do sudo rocm-smi -d $i --setperfdeterminism 1500; done
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setpoweroverdrive 750

# dependencies
#pip install tqdm
#pip install numpy
#pip install ftfy
#pip install regex
#pip install pillow
#pip install scipy
# webdataset depends on torch.utils.data.DataLoader
#pip install --index-url https://download.pytorch.org/whl/cpu torch
#pip install webdataset
source venv/bin/activate
pip list
apt list --installed | grep amdgpu

export BEAM=5 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 IGNORE_JIT_FIRST_BEAM=1 HCQDEV_WAIT_TIMEOUT_MS=300000
export AMD_LLVM=0 # bf16 seems to require this

export BASEDIR="~/stable_diffusion"
export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"
export MODEL="stable_diffusion" PYTHONPATH="."

# set these if resuming from checkpoint
#export RESUME_CKPTDIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09090228"
#export RESUME_ITR=5334
export GPUS=8 BS=304

# use separate BS for the jits in eval to maximize throughput
#export RUN_EVAL=1 EVAL_ONLY=1 CONTEXT_BS=816 DENOISE_BS=600 DECODE_BS=384 INCEPTION_BS=560 CLIP_BS=240

export WANDB=1
export PARALLEL=0

export TOTAL_CKPTS=6

DATETIME=$(date "+%m%d%H%M")
#LOGFILE="sd_red_${DATETIME}_${SEED}.log"
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR
LEARNING_RATE="6.25e-8" RUNMLPERF=1 python3 examples/mlperf/model_train.py

sleep 120

DATETIME=$(date "+%m%d%H%M")
export UNET_CKPTDIR="${BASEDIR}/checkpoints/training_checkpoints/${DATETIME}"
mkdir -p $UNET_CKPTDIR
LEARNING_RATE="2.5e-7" RUNMLPERF=1 python3 examples/mlperf/model_train.py
#EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09130207/run_eval_2" RUNMLPERF=1 python3 examples/mlperf/model_train.py