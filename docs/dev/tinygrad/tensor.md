# Tensor Implementation Details

`tinygrad/tensor.py` defines the `Tensor` class, the core API for tinygrad. This document details its implementation mechanics.

## 1. The `Tensor` Class

### 1.1 `__init__`
The constructor standardizes inputs into a `UOp`.
*   **`data` Handling**:
    *   **`UOp`**: Validates dtype. If `Ops.BIND` (from `Variable.bind`), it replaces the bound constant with a new `Ops.CONST` on the correct device.
    *   **`None`**: Creates a `Const` UOp (0.0). `_force_unique` uses `UOp.unique_const` (cache-breaking).
    *   **`ConstType`**: Creates a `Const` UOp.
    *   **`bytes`**: Calls `_frompy`. Creates a `PYTHON` buffer (CPU) containing the raw bytes.
    *   **`list`/`tuple`**: Calls `_frompy`. Infers dtype (bool/int/float) if `None`. Handles flattening.
    *   **`numpy.ndarray`**: Calls `_fromnp`. Allocates a `NPY` buffer reusing the array's memory (zero-copy if possible).
    *   **`pathlib.Path`**: Disk loading. Creates a `DISK` buffer (`UOp.new_buffer`).
*   **Device Normalization**: Uses `canonicalize_device`.
*   **UOp Assignment**: Ensures the final `self.uop` matches the requested device (via `copy_to_device` or `shard`).
*   **Reference Tracking**: Adds `weakref` of `self` to `all_tensors`.

### 1.2 Graph Construction (`_apply_uop`)
Generic handler for creating new Tensors from operations.
1.  **Call**: Takes a function (e.g., lambda building a `UOp` tree) and input tensors.
2.  **Execute**: Calls the function with input `uop`s.
3.  **Metadata**: If `TRACEMETA`, attaches source location to the new `UOp`.
4.  **Grad Requirement**: `ret.requires_grad` is True if *any* input requires grad.
5.  **Return**: Returns a new `Tensor` wrapping the result `UOp`.

### 1.3 `realize`
Triggers execution.
1.  **Filter**: Selects tensors that are *not* contiguous or realized.
2.  **Schedule**: Calls `Tensor.schedule_with_vars` -> `complete_create_schedule_with_vars`.
3.  **Run**: Calls `run_schedule` to execute the kernels.
4.  **Return**: Returns `self`.

### 1.4 `schedule_with_vars`
1.  **Sink**: Creates a `UOp.sink` of the tensor(s) uops.
2.  **Scheduler**: Calls `engine.schedule.complete_create_schedule_with_vars`.
    *   This linearizes the graph into `ExecItem`s.
    *   It returns `becomes_map`, `schedule`, `var_vals`.
3.  **Update (`_apply_map_to_tensors`)**:
    *   The scheduler might rewrite the graph (e.g., fusing, changing buffer pointers).
    *   `_apply_map_to_tensors` iterates over `all_tensors` (weakrefs).
    *   It updates the `uop` of any Tensor involved in the schedule to point to the *realized* (or scheduled) `UOp`.
    *   This "in-place" update is how lazy evaluation propagates results back to user objects.

## 2. Gradient Computation

### 2.1 `backward`
1.  **Toposort**: `self.uop.toposort()`. Finds the upstream graph.
2.  **Filter**: Identifies `tensors_need_grad`.
3.  **Compute**: Calls `compute_gradient`.
    *   Inputs: Loss UOp, Loss Grad (1.0), Target UOps.
    *   Output: Map of `target_uop -> grad_uop`.
4.  **Accumulate**:
    *   Iterates targets.
    *   `t.grad = t.grad + g` (if exists) else `g`.
    *   This handles branching paths summing up gradients.

## 3. Operations Breakdown

### 3.1 Movement Ops (`_mop`)
*   **`reshape`**: Calls `self._apply_uop(UOp._mop, Ops.RESHAPE, arg)`.
*   **`pad`**: Complex logic.
    *   Supports "flat" (PyTorch style) and "grouped" (NumPy style) padding.
    *   Modes: `constant` (pad with val), `reflect`, `replicate`, `circular`.
    *   `constant`: Uses `UOp.pad` and `where` (for non-zero value).
    *   `reflect/replicate`: Implemented via slicing and concatenation (`cat`).
*   **`permute`**: Calls `UOp.permute`.
*   **`expand`**: Calls `UOp.expand`.
*   **`shrink`**: Calls `UOp.shrink`.
*   **`stride`**: Handled in `_getitem` via reshape/shrink logic.

### 3.2 Slicing (`_getitem`)
Handles complex indexing: `int`, `slice`, `Tensor`, `None`, `Ellipsis`.
1.  **Normalization**: Resolves `Ellipsis` and `None` (newaxis).
2.  **Iteration**: Goes dim by dim.
    *   `slice`: Converts to `shrink` (start/end) and `stride`.
    *   `int`: Converts to `shrink` (range 1) + `reshape` (drop dim).
    *   `Tensor` (Advanced Indexing):
        *   Creates `one_hot` masks.
        *   Uses `mul` (masking) and `sum` (reduction) to extract values.
        *   This is expensive compared to basic slicing.
3.  **Optimization**: Merges consecutive slices/strides.

### 3.3 Processing (`conv2d`)
1.  **Im2Col / Winograd**:
    *   If `WINO` and conditions met, uses `_apply_winograd_matrix` (F(4x4, 3x3)).
    *   Otherwise, standard Im2Col:
        *   `pool` (with stride/dilation) creates the windows.
        *   `reshape`/`permute` aligns them.
        *   `mul` (elementwise multiplication with weights).
        *   `sum` (reduce over window).
2.  **Groups**: Handles grouped convolution by reshaping.

### 3.4 Reduction (`sum`, `max`)
*   Calls `_reduce`.
*   Resolves axis (supports tuple).
*   Calls `UOp.r` (`Ops.REDUCE_AXIS`).
*   If `keepdim=False`, adds a `reshape` to squeeze dimensions.

### 3.5 Matmul (`dot`)
*   Checks dimensions (broadcasting).
*   Reshapes inputs to align last dimensions.
*   Calls `mul` (elementwise) and `sum` (reduction).
*   Standard approach: `(M, K) * (K, N) -> (M, K, N) -> sum(1) -> (M, N)`.

## 4. Helpers

*   **`_broadcasted`**:
    *   Given two tensors/consts.
    *   Casts to common dtype.
    *   Calculates `_broadcast_shape`.
    *   Calls `expand` on both to match shape.
*   **`_to_np_dtype`**: Maps tinygrad `DType` to `numpy.dtype`.
*   **`_from_np_dtype`**: Inverse.

## 5. Randomness
*   **`Threefry`**: A counter-based PRNG used in `rand` and `randn`.
*   **State**: `_device_rng_counters` stores the offset per device.
*   **Generation**:
    *   Generates 2x 32-bit randoms.
    *   Bitcasts/masks to float mantissa.
    *   Subtracts 1.0 to get U[0, 1).
