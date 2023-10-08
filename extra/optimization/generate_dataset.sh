#!/bin/bash
export LOGOPS=/tmp/ops
rm $LOGOPS

# generate many kernels
STEPS=3 python3 examples/hlb_cifar10.py
python3 examples/stable_diffusion.py --noshow
python3 examples/llama.py --prompt "hello" --count 5
python3 examples/gpt2.py --count 5
python3 examples/mlperf/model_spec.py
python3 examples/yolov8.py ./test/models/efficientnet/Chicken.jpg
openpilot/go.sh
BIG=1 MPS=1 pytest test/

# sort and uniq
sort -u /tmp/ops > /tmp/sops
ls -lh /tmp/ops /tmp/sops
