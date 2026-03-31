#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
VIZ=${VIZ:--1} examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama405b/implementations/tinybox_8xMI350X/dev_run.sh
extra/viz/cli.py --profile -s "${DEV:-AMD}" | head -23
