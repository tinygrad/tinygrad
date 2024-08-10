#!/bin/bash

export TINY_TYPE='tinygreen'
export PYTHONPATH='.'
export MODEL='retinanet'
export DEFAULT_FLOAT='HALF' GPUS=6 BS=96 BS_EVAL=36
export SYNCBN=0
export LAZYCACHE=0
export TRAIN_BEAM=3 IGNORE_JIT_FIRST_BEAM=1 BEAM_UOPS_MAX=1500 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=10 BEAM_PADTO=0
export DATAPATH='/raid/datasets/open-images'
export PART_BATCH=1
export WANDB=1
export WANDB_RUN_ID=$(date +%Y%m%d%H%M%S%N)

# Download dataset if missing
python3 extra/datasets/openimages.py

TRAIN_ONLY=1 python3 examples/mlperf/model_train.py
export SYNCBN=1
export LOAD_BN=0
EVAL_ONLY=1 SHIFT=1 CHKPT_PATH="ckpts/retinanet_6x${TINY_TYPE}_B${BS}_S${LOAD_BN}_E${SHIFT}.safe" python3 examples/mlperf/model_train.py &
sleep 30
EVAL_ONLY=1 SHIFT=2 CHKPT_PATH="ckpts/retinanet_6x${TINY_TYPE}_B${BS}_S${LOAD_BN}_E${SHIFT}.safe" python3 examples/mlperf/model_train.py &
sleep 30
EVAL_ONLY=1 SHIFT=3 CHKPT_PATH="ckpts/retinanet_6x${TINY_TYPE}_B${BS}_S${LOAD_BN}_E${SHIFT}.safe" python3 examples/mlperf/model_train.py &
sleep 30
EVAL_ONLY=1 SHIFT=4 CHKPT_PATH="ckpts/retinanet_6x${TINY_TYPE}_B${BS}_S${LOAD_BN}_E${SHIFT}.safe" python3 examples/mlperf/model_train.py &

wait

echo "RETINA SCRIPT COMPLETE"