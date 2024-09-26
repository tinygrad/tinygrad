#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=54 EVAL_BS=6

export BEAM=4
export IGNORE_JIT_FIRST_BEAM=1 BEAM_UOPS_MAX=2000 BEAM_UPCAST_MAX=96 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0
export BASEDIR="/raid/datasets/wiki"

export BENCHMARK=10 DEBUG=2

python3 examples/mlperf/model_train.py
