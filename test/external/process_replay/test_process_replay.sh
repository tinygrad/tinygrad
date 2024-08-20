#!/bin/bash

# should assert
sed -i 's/temp/temp1/g' ./tinygrad/codegen/kernel.py
COMPARE_SCHEDULE=0 ASSERT_PROCESS_REPLAY=1 python3 test/external/process_replay/process_replay.py &> /dev/null
if [[ $? -eq 0 ]]; then
  echo "PROCESS REPLAY IS WRONG."
  exit 1
fi
# should NOT assert
git stash > /dev/null
COMPARE_SCHEDULE=0 ASSERT_PROCESS_REPLAY=1 python3 test/external/process_replay/process_replay.py &> /dev/null
