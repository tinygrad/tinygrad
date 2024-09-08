#!/bin/bash
export LOGOPS=/tmp/sops
rm $LOGOPS
test/external/process_replay/reset.py

RUN_PROCESS_REPLAY=1 python3 -m pytest -n=auto test/ --ignore=test/unit --durations=20
extra/optimization/generate_dataset.py
