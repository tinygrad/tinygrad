#!/bin/bash

export PYTHONPATH="."
export MODEL="bert"
export SUBMISSION_PLATFORM="tinybox_red"
export DEFAULT_FLOAT="HALF" GPUS=6 BS=84 EVAL_BS=6

export BEAM=4
export BASEDIR="/raid/datasets/wiki"

echo "TODO: DISABLING DROPOUT - UNSET FOR REAL SUBMISSION RUN"
export DISABLE_DROPOUT=1 # TODO: Unset flag for real submission run.

# pip install -e ".[mlperf]"
export LOGMLPERF=1

export SEED=$RANDOM
DATETIME=$(date "+%m%d%H%M")
LOGFILE="bert_green_${DATETIME}_${SEED}.log"

# init
BENCHMARK=10 INITMLPERF=1 python3 examples/mlperf/model_train.py | tee $LOGFILE

# run
RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a $LOGFILE
