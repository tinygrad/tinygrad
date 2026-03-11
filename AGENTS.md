# tinygrad Agent Instructions

## Coding Standards

- **No code golf.** Reduce complexity and increase readability; deleting newlines does nothing to help.
- **Small diffs.** If a PR looks "complex", is a big diff, or adds lots of lines, it won't be merged. Break work into smaller commits that are individually clear wins.
- **Prerequisite refactors first.** If you can cleanly refactor to the point that a feature is a 3-line change, do that.
- **Line tradeoff matters.** A 3-line feature has a lower bar than a 300-line feature. Keep additions minimal.
- **All features must have regression tests.** API should match torch or numpy where possible.
- **Speedup claims must be benchmarked.**
- **Don't change code outside core `tinygrad/` unless it's broken.**
- **Dead code removal from `tinygrad/` is welcome.**
- **Refactors must be clear wins** and should pass process replay.
- **2-space indentation, 150 char line length** (see ruff config in pyproject.toml).

## Before Every Commit

Run the pre-commit checks (ruff, mypy, tiny tests):

```sh
python -m ruff check .
python -m mypy
python -m pytest test/test_tiny.py
```

Run the comprehensive test suite:

```sh
OMP_NUM_THREADS=1 SKIP_SLOW_TEST=1 PYTHONPATH="." python -m pytest -n=6 \
  test/backend/test_ops.py test/backend/test_schedule.py \
  test/unit/test_assign.py test/backend/test_tensor.py \
  test/backend/test_jit.py test/unit/test_schedule_cache.py \
  test/null/test_pattern_matcher.py test/null/test_uop_symbolic.py \
  test/unit/test_helpers.py
```