#!/bin/bash

export PYTHONPATH='.'
export MODEL='retinanet'
export DEFAULT_FLOAT='HALF' GPUS=6 BS=96 BS_EVAL=36
export LAZYCACHE=0
export TRAIN_BEAM=3 IGNORE_JIT_FIRST_BEAM=1 BEAM_UOPS_MAX=1500 BEAM_UPCAST_MAX=64 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=10 BEAM_PADTO=0
export DATAPATH='/raid/datasets/open-images'
export WANDB=1

CUDA=1 TRAIN_ONLY=1 python3 examples/mlperf/model_train.py

CUDA=1 EVAL_ONLY=1 python3 examples/mlperf/model_train.py