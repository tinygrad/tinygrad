# Tensor Implementation Details

`tinygrad/tensor.py` defines the `Tensor` class, which is the user-facing API for deep learning. It wraps the `UOp` graph and provides autograd, lazy execution, and device management.

## 1. The `Tensor` Class

### 1.1 Attributes

*   **`uop` (`UOp`)**: The underlying micro-operation graph node representing this tensor's data. This is the source of truth for the tensor's value and history.
*   **`requires_grad` (`bool | None`)**:
    *   `True`: Gradients will be computed for this tensor during backward pass.
    *   `False`: Gradients are not computed.
    *   `None`: Default state. Will be set to `True` if the tensor is used in an optimizer or if `requires_grad=True` is explicitly passed.
*   **`grad` (`Tensor | None`)**: Stores the gradient of the loss with respect to this tensor. Populated after `.backward()` is called.

### 1.2 Initialization (`__init__`)

The constructor handles various input types and normalizes them into a `UOp`.

*   **Data Sources**:
    *   **`UOp`**: Wraps an existing UOp directly. Checks for dtype mismatches.
    *   **`ConstType`** (int, float, bool): Creates a constant scalar tensor.
    *   **`bytes`**: Creates a tensor from raw bytes (useful for loading weights).
    *   **`list` / `tuple`**: Recursively flattens and loads data.
    *   **`numpy.ndarray`**: Copies data from NumPy.
    *   **`pathlib.Path`**: Loads data from a file (lazy disk loading).
*   **Device & DType**: Normalizes `device` string (e.g., "GPU:0" -> "GPU") and `dtype`.
*   **Unique Consts**: `_force_unique` flag forces creation of a new UOp even for cached constants, useful for unique ids.

### 1.3 Properties

*   **`shape`**: Delegated to `self.uop.shape`.
*   **`dtype`**: Delegated to `self.uop.dtype`.
*   **`device`**: Delegated to `self.uop.device`.

## 2. Autograd Engine

The autograd system is implicit in the `UOp` graph construction and explicit in the `backward` pass.

### 2.1 Forward Pass (Graph Construction)

Operations on `Tensor` (e.g., `__add__`, `relu`, `matmul`) call `_apply_uop`.

```python
def _apply_uop(self, fxn:Callable, *x:Tensor, extra_args=(), **kwargs) -> Tensor:
    # ...
    new_uop: UOp = fxn(*[t.uop for t in (self,)+x], *extra_args, **kwargs)
    # ...
    ret = Tensor.__new__(Tensor)
    ret.uop = new_uop
    # ...
    return ret
```

*   It extracts `uop`s from input tensors.
*   Calls the `UOp` generation function (e.g., `UOp.alu`, `UOp.load`).
*   Creates a new `Tensor` wrapping the result `UOp`.
*   Propagates `requires_grad`.

### 2.2 Backward Pass (`backward`)

1.  **Topological Sort**: `self.uop.toposort()` finds all nodes in the graph leading to the loss.
2.  **Filter**: Identifies tensors that `require_grad`.
3.  **Compute Gradients**: Calls `tinygrad.gradient.compute_gradient`.
    *   This generates the backward graph (UOps) for gradients.
4.  **Accumulate**:
    *   Iterates through computed gradients.
    *   Assigns them to `.grad` attributes of corresponding Tensors.
    *   If a Tensor already has a gradient (from a previous backward pass), it accumulates (`+=`).

## 3. Data Management

### 3.1 Realization (`realize`)

Tinygrad is lazy. Computation only happens when `realize()` is called (explicitly or implicitly via `.numpy()`, `.item()`).

```python
def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    if len(to_realize:=[x for x in (self,)+lst if not x.uop.is_contiguous()]):
      run_schedule(*Tensor.schedule_with_vars(*to_realize), do_update_stats=do_update_stats)
    return self
```

*   It gathers tensors to realize.
*   Calls `schedule_with_vars` to generate an execution schedule (`list[ExecItem]`).
*   Calls `run_schedule` (in `engine/realize.py`) to execute the kernels.

### 3.2 Scheduling (`schedule_with_vars`)

This bridges the `Tensor` world and the `Engine` world.

1.  **Sink Creation**: Creates a `UOp.sink` of all target tensors.
2.  **Scheduling**: Calls `complete_create_schedule_with_vars` (in `engine/schedule.py`).
    *   This transforms the UOp graph into a linear list of `ExecItem`s.
    *   It handles "rangeifying" (loops), memory planning, and graph linearization.
3.  **Map Application**: Updates the `Tensor.uop`s to point to the realized buffers (or the new graph state).

### 3.3 Storage

*   **`_buffer()`**: Returns the underlying `Buffer` object (from `device.py`).
*   **`_data()`**: Returns a `memoryview` of the data (synchronously copies to CPU if needed).

## 4. Tensor Operations

### 4.1 Movement Ops
Implemented directly or via `_mop`.
*   `reshape`, `permute`, `expand`, `pad`, `shrink`, `flip`.
*   These manipulate the `UOp` metadata (shape, strides) without necessarily launching kernels (zero-copy views), unless a contiguous buffer is required later.

### 4.2 Elementwise Ops
Implemented via `_binop` or `alu`.
*   `add`, `sub`, `mul`, `div`, `pow`.
*   Uses `_broadcasted` to handle shape broadcasting before applying the ALU op.

### 4.3 Reduction Ops
Implemented via `_reduce`.
*   `sum`, `max`.
*   Maps to `UOp(Ops.REDUCE_AXIS, ...)`.

### 4.4 Processing Ops
High-level ops like `conv2d`, `avg_pool2d`.
*   These are implemented by decomposing them into simpler ops (`reshape`, `permute`, `pad`, `mul`, `sum` - effectively "im2col" style or Winograd).
*   Example: `conv2d` might reshape input, expand weights, and do a reduction.

## 5. Randomness (`rand`, `randn`)

Uses a counter-based PRNG (Threefry).
*   **`manual_seed`**: Sets the global seed.
*   **`_device_seeds`**: Stores seeds per device.
*   **`_threefry_random_bits`**: Generates random bits deterministically based on seed and position.
*   Random tensors are lazy; the random generation happens in the kernel execution.

## 6. Multi-Device Support (`shard`)

*   **`shard`**: splits a tensor across multiple devices.
*   Creates a `MultiLazyBuffer` (handled via `UOp` with tuple device).
*   Operations on sharded tensors dispatch to sharded UOps (`Ops.MULTI`), which the scheduler expands into per-device kernels.

## 7. Context Managers

*   **`train`**: Sets `Tensor.training`. Used by layers like Dropout and BatchNorm to change behavior.
*   **`no_grad`** (implicit): By setting `requires_grad=False` on inputs.

## 8. Helpers

*   **`argfix`**: Normalizes arguments (tuples/lists).
*   **`_metadata_wrapper`**: Captures stack traces for debugging/visualization (`TRACEMETA`).
