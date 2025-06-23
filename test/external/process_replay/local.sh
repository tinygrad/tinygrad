#!/bin/bash

set -e
HEAD=$(git rev-parse --abbrev-ref HEAD)
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command to process replay>"
  exit 1
fi

# reset the exisitng replay cache
python test/external/process_replay/reset.py

# run the command just passed in with CAPTURE_PROCESS_REPLAY=1
CAPTURE_PROCESS_REPLAY=1 "$@"

# checkout master
GIT=${GIT:-1}
if [ "$GIT" -eq 1 ]; then
  git checkout master
  git checkout "$HEAD" -- test/external/process_replay/process_replay.py
fi

# run process replay
ASSERT_PROCESS_REPLAY=${ASSERT_PROCESS_REPLAY:-1} python test/external/process_replay/process_replay.py

# install as pr util:
# sudo ln -s "$(pwd)/test/external/process_replay/local2.sh" /usr/local/bin/pr && hash -r
# pr python test/test_ops.py TestOps.test_add
