#!/bin/bash
set -e

export BENCHMARK=${BENCHMARK:-5}
export EVAL_BS=0
export FP8_FA=${FP8_FA:-1}
VIZ=${VIZ:--1} FULL_LAYERS=1 DEBUG=${DEBUG:--0} examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_beam.sh
[ "$BENCHMARK" -le 3 ] || [[ $DEV == NULL* ]] || python -m tinygrad.viz.cli -s AMD -t --interval "train @ 2" "train @ 3"
