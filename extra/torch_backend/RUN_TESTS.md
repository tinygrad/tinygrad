## Torch Backend Test Guide

Run from repo root after activating `.venv`.

```bash
# keep torch extensions and sqlite cache inside the repo
export PYTHONPATH=.
export TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions
export CACHEDB=$PWD/.cache/cache.db
export XDG_CACHE_HOME=$PWD/.cache

# run the backend test suite
.venv/bin/python extra/torch_backend/test.py

# run targeted smoke tests while iterating
.venv/bin/python -m pytest test/test_ops.py -k diag -n 0
```

The `TORCH_EXTENSIONS_DIR` and cache vars avoid permission errors on macOS. Delete `.torch_extensions` / `.cache` if a stale build causes trouble.***
