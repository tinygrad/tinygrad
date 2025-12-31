# JIT Implementation Details

`tinygrad/engine/jit.py` implements `TinyJit`, a record-and-replay optimization decorator.

## 1. `TinyJit.__call__` Flow

The logic is controlled by `self.cnt`:

### 1.1 `cnt == 0` (Warmup)
1.  **Preparation**: Calls `_prepare_jit_inputs`.
    *   Identifies tensor inputs.
    *   Realizes unrealized inputs (JIT cannot handle purely symbolic inputs that change every time).
    *   Extracts `var_vals` (symbolic variables).
2.  **Execution**: Calls `self.fxn`.
3.  **Realize**: Calls `Tensor.realize` on the return values to ensure computation happens.
4.  **No Capture**: Just returns the result. Increment `cnt`.

### 1.2 `cnt == 1` (Capture)
1.  **Setup**:
    *   Initializes `self._jit_cache` (list).
    *   Initializes `self._buffer_replace` (map for deduplicating buffers).
2.  **Context**: Sets `BEAM` (optional) and `NO_MEMORY_PLANNER=1` (we want raw graph first).
3.  **Capture**:
    *   `capturing.append(self)`.
    *   Run `self.fxn`.
    *   Any call to `run_schedule` (in `realize.py`) sees `capturing` and appends `ExecItem`s to `self._jit_cache` instead of running them.
    *   `capturing.clear()`.
4.  **Pruning**:
    *   If `self.prune=True`:
        *   Calculates dependencies (buffers used by outputs).
        *   Filters `jit_cache` to remove kernels that don't contribute to outputs (dead code).
        *   Runs the pruned ("onetime") kernels immediately.
5.  **Optimization**:
    *   **Memory Planning**: Calls `_internal_memory_planner` on the captured schedule to assign buffers efficiently. Replaces temporary buffers in `ExecItem`s.
    *   **Input Replacement**: Builds `input_replace` map: `(jit_index, buffer_index) -> input_arg_index`.
6.  **Finalize**: Creates `CapturedJit`.

### 1.3 `cnt >= 2` (Execution)
1.  **Verify**: Checks if input arguments (names, shapes, dtypes) match the capture.
2.  **Run**: Calls `self.captured(input_buffers, var_vals)`.

## 2. `CapturedJit`

Holds the frozen execution plan.

### 2.1 `__call__`
1.  **Input Binding**:
    *   Updates `self._jit_cache[j].bufs[i]` with the new `input_buffers`.
    *   Uses `extra_view_inputs` to reconstruct view buffers (offset pointers) from base inputs.
2.  **Graphing (First Run Only)**:
    *   If `JIT < 2` (Graph enabled):
    *   Calls `apply_graph_to_jit`.
    *   Groups sequences of kernels that run on the same device into `GraphRunner` items.
    *   Replaces the list of kernels with a single `ExecItem` calling the graph.
3.  **Execution Loop**:
    *   Iterates `self._jit_cache`.
    *   Calls `ei.run(var_vals, jit=True)`.
    *   If it's a `GraphRunner`, it updates the graph node params.
    *   If it's a `CompiledRunner` (fallback), it launches the kernel.

## 3. `GraphRunner`

Abstract base for hardware graphs (CUDA Graphs, etc.).

### 3.1 `__init__`
*   **Analysis**:
    *   Identifies variables (`var_vals`) used in the graph.
    *   Identifies symbolic launch dimensions.
    *   Builds dependency maps (`w_dependency_map`, `r_dependency_map`) to handle synchronization within the graph.

### 3.2 `apply_graph_to_jit`
1.  **Batching**: Iterates through the cache.
2.  **Grouping**: Collects consecutive `CompiledRunner` items for the same device.
3.  **Flushing**: Creates a `GraphRunner` for the batch via `device.graph(batch, ...)`.

## 4. Why?
*   **Latency**: Python overhead is ~10-20us per kernel. GPU kernels can be ~1us. JIT removes Python loop.
*   **Memory**: Static planner reuses intermediate memory, reducing footprint.
*   **Graphs**: Hardware graphs reduce driver overhead (launch 100 kernels in 1 syscall).
