# Schedule

The `tinygrad/schedule/` directory handles graph scheduling transformations.

## `__init__.py`

Exports scheduling functions. (See `tinygrad/engine/schedule.py` for the main scheduler logic).

## `indexing.py`

Helpers for symbolic indexing and range calculations.

## `multi.py`

Handles multi-device scheduling logic (`Ops.MULTI`). It transforms graphs to run across multiple devices.

## `rangeify.py`

Responsible for "rangeifying" the graph, which means identifying and handling loops and ranges in the computation. This is crucial for optimizing kernels that can be expressed as loops.
