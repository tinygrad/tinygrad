#!/bin/bash

HEAD=$(git rev-parse --abbrev-ref HEAD)
python test/external/process_replay/reset.py
RUN_PROCESS_REPLAY=1 python test/test_ops.py TestOps.test_add
git checkout master
ASSERT_PROCESS_REPLAY=1 python test/external/process_replay/process_replay.py
git checkout $HEAD
