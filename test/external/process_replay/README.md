# Process replay tests

Process replay is a tool for creating a diff of generated kernels between two commits.

Refactor and speedup PRs need a green process replay check.

Behavior change PRs can use process replay with `ASSERT_PROCESS_REPLAY=0` to check the diff is what was expected. It's also an indirect test coverage checker.

## Running locally

To run process replay locally:

(optional: clear previous process replay runs with `test/external/process_replay/reset.py`)

1. Run tests with `RUN_PROCESS_REPLAY=1` in your branch
2. Checkout master
3. Run `test/external/process_replay/process_replay.py`
