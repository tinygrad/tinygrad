#!/usr/bin/env bash
# timed run: model init loads the graphs dev_beam.sh saved (still before run_start), then trains on the real dataset to
# log perplexity 3.34
set -e
source "$(dirname "$0")/common.sh"
export RUNMLPERF=1 BENCHMARK=0 BEAM_CACHE_ONLY=1
export GPTOSS_JIT_LOAD=${GPTOSS_JIT_LOAD:-$GPTOSS_JIT_PKL}
bash "$(dirname "$0")/setup.sh" down "$(hostname)" # this box's gpus are opened directly: fresh hive, before the run's clock
exec bash "$(dirname "$0")/../tinybox_8xMI350X/dev_run.sh"
