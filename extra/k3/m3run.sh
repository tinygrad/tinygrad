#!/bin/bash
# usage: m3run.sh name prompt_file|- [ENV=VAL ...]: one engine run of ~/k3/tg-merge under the gpu lock, log ~/k3/m3_name.log
name=$1 pf=$2; shift 2
prompt=""; [ "$pf" != "-" ] && prompt="$(cat "$pf")"
cd ${TG:-~/k3/tg-merge}
flock ~/k3/gpu.lock env GUESS=$HOME/k3/guess-v1 COMGR_PATH=$HOME/libamd_comgr.so COMGR_3_PATH=$HOME/libamd_comgr.so PYTHONPATH=. VFIO=1 \
  DEV="AMD:HIP;CPU:LLVM" "$@" timeout 1800 ../venv/bin/python -u extra/k3/engine.py ~/k3/trunk-q4_k-pr.gguf "$prompt" 2>&1 | grep --line-buffered -vE "Warn|frozen" > ~/k3/m3_$name.log
echo "== $name"; grep -E "ms/token|top-1|speculative" ~/k3/m3_$name.log | tail -2 | cut -c1-330
