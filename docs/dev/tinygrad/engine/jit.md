# JIT (Just-In-Time) Implementation Details

`tinygrad/engine/jit.py` implements the JIT compiler/executor `TinyJit`.

## 1. The `TinyJit` Class

A decorator class that records and replays computation graphs.

### 1.1 Motivation
In deep learning, the same operations (layers) are executed repeatedly with the same shapes but different data.
Dispatching kernels from Python is slow (overhead). JIT allows us to:
1.  **Capture**: Record the sequence of kernels once.
2.  **Optimize**: Fuse or graph the sequence.
3.  **Replay**: Execute the sequence with low overhead.

### 1.2 Usage
```python
@TinyJit
def run(x):
  return x.relu()
```

### 1.3 Execution Flow (`__call__`)

It has a state machine based on `self.cnt`:

1.  **Cnt 0 (Warmup)**:
    *   Executes the function normally.
    *   Does NOT capture yet.
    *   Ensures lazy buffers are realized so we start with a clean slate.

2.  **Cnt 1 (Capture)**:
    *   Initializes `self._jit_cache`.
    *   Sets global `capturing` flag (consumed by `run_schedule`).
    *   Executes the function.
    *   `run_schedule` appends `ExecItem`s to `_jit_cache` instead of running them.
    *   **Pruning**: Removes independent kernels (kernels whose outputs are not used).
    *   **Optimization**:
        *   `_internal_memory_planner`: Reassigns buffers to reuse memory (since the graph is static).
        *   `apply_graph_to_jit`: Converts a list of kernels into a `GraphRunner` (e.g., CUDA Graphs).
    *   Creates a `CapturedJit` object.

3.  **Cnt > 1 (Replay)**:
    *   Calls `self.captured(input_buffers, var_vals)`.

## 2. `CapturedJit` Class

Stores the captured execution plan.

### 2.1 Inputs
*   **`input_replace`**: Map `(jit_cache_index, buffer_index) -> input_argument_index`.
    *   This allows replacing the "dummy" buffers recorded during capture with the actual input buffers passed during replay.
*   **`extra_view_inputs`**: Handles inputs that are views (offsets) of other inputs.

### 2.2 Execution
*   Updates the `bufs` in `jit_cache` with the new inputs.
*   Runs the `ExecItem`s.
    *   If using `GraphRunner`, it updates the graph node parameters instead of launching new kernels.

## 3. `GraphRunner`

Wraps hardware graph APIs (CUDA Graphs, HGS).

*   **Batched Execution**: Groups multiple `CompiledRunner` items into a single graph submission.
*   **Dependency Management**: `_access_resources` tracks read/write dependencies to insert barriers/edges in the graph.
*   **Update**: Updates pointers (buffer addresses) and scalars (launch dims, var vals) in the instantiated graph.

## 4. Why this design?

*   **Python Overhead**: Removing Python loop overhead for small kernels.
*   **Device Overhead**: Reducing kernel launch overhead (driver latency).
*   **Memory Footprint**: Static memory planning can significantly reduce peak memory usage compared to dynamic allocation.
