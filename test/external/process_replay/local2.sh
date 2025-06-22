#!/bin/bash

# local.sh without git stuff
set -e
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command to process replay>"
  exit 1
fi
python test/external/process_replay/reset.py
CAPTURE_PROCESS_REPLAY=1 "$@"
ASSERT_PROCESS_REPLAY=${ASSERT_PROCESS_REPLAY:-1} python test/external/process_replay/process_replay.py
