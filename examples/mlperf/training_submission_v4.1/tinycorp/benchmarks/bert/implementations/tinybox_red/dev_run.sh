#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=54 EVAL_BS=6

export BEAM=4
export BASEDIR="/raid/datasets/wiki"

export WANDB=1

python3 examples/mlperf/model_train.py