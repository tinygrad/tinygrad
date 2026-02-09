#!/bin/bash
export BENCHMARK=5
export EVAL_BS=0
export VIZ=${VIZ:--1}
examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama8b/implementations/tinybox_8xMI350X/dev_run.sh
PYTHONPATH="." extra/viz/cli.py --profile --device "AMD" --top 20
