Tinygrad Design (v2)
====================

Overview
--------

Tinygrad is currently a bit bloated, and there are places where concerns should be separated but aren't. The goal is to improve code organization and separation of concerns.

tensor.py and mlops.py
----------------------

'tensor.py' and `mlops.py` contain great code. The interface going backward is as follows:

- `LazyBuffer.const`: This creates a matching size buffer.
- `LazyBuffer.contiguous`: This is not exactly elementwise.
- `LazyBuffer.e`: Elementwise operations.
- `LazyBuffer.r`: Reduce operations.
- Reshape/permute/expand/stride/shrink/pad: Movement operations.

lazy.py
-------

The `lazy.py` reordering engine has unnecessary logic for movement operations that should be removed.

view.py
-------

`view.py` contains mostly great code, except it shouldn't have rendering logic. The int type should be parameterized to avoid importing from symbolic.

LazyOp
------

`LazyOp` shouldn't have `LazyBuffers` as sources. Instead, use `LazyOp` LoadOps with a tuple of Views. This way, `LazyOp` uniquely determines the kernel, eliminating the need for replacements.

ShapeTracker
------------

`ShapeTracker` probably shouldn't exist and should be part of `LazyBuffer`. Most of the logic in `ShapeTracker` should move to `symbolic_view`, which combines view and symbolic.

This refactoring aims to improve the overall structure and separation of concerns in Tinygrad. Keep in mind that these are guidelines, and adjustments may be necessary based on the specific requirements and architecture of the codebase.
