#!/bin/bash

export TINY_TYPE='tinyred'
export PYTHONPATH='.'
export MODEL='retinanet'
export DEFAULT_FLOAT='HALF' GPUS=6 BS=96 BS_EVAL=36
export LAZYCACHE=0
export TRAIN_BEAM=3 IGNORE_JIT_FIRST_BEAM=1 BEAM_UOPS_MAX=1500 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=10 BEAM_PADTO=0
export DATAPATH='/raid/datasets/open-images'
export PART_BATCH=1
export WANDB=1

# Download dataset if missing
python3 extra/datasets/openimages.py

TRAIN_ONLY=1 python3 examples/mlperf/model_train.py

EVAL_ONLY=1 CHKPT_PATH='ckpts/retinanet_6xtinyred_B96_E1.safe' python3 examples/mlperf/model_train.py
EVAL_ONLY=1 CHKPT_PATH='ckpts/retinanet_6xtinyred_B96_E2.safe' python3 examples/mlperf/model_train.py
EVAL_ONLY=1 CHKPT_PATH='ckpts/retinanet_6xtinyred_B96_E3.safe' python3 examples/mlperf/model_train.py
EVAL_ONLY=1 CHKPT_PATH='ckpts/retinanet_6xtinyred_B96_E4.safe' python3 examples/mlperf/model_train.py