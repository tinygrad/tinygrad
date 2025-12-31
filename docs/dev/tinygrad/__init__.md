# `__init__.py`

`tinygrad/__init__.py` is the package entry point.

It exposes key components:
- `Tensor`: The main tensor class.
- `TinyJit`: The JIT compiler decorator.
- `Device`: The device manager.
- `dtypes`: Data types.
- `GlobalCounters`: Performance counters.
- `Variable`: Symbolic variable (from UOps).

It also optionally installs a type import hook if `TYPED` env var is set.
