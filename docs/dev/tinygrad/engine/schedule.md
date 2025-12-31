# Schedule Implementation Details

`tinygrad/engine/schedule.py` converts a graph of `UOp`s (from Tensor realization) into a linear sequence of `ExecItem`s.

## 1. `complete_create_schedule_with_vars`

The main entry point.

### 1.1 Inputs
*   **`big_sink` (`UOp`)**: A `UOp.sink` containing all the tensors we want to realize.

### 1.2 Process

1.  **Graph Rewrite (Pre-sched)**:
    *   Replaces `BUFFER` ops with `LUNIQUE` (logical unique) to normalize the graph for caching.
    *   Strips `BIND` values.
    *   This allows us to cache the schedule even if buffer pointers or variable values change.

2.  **Schedule Cache**:
    *   Checks `schedule_cache` using the normalized graph key.
    *   If hit, returns the cached schedule.

3.  **Graph Transformation**:
    *   **Multi-Device**: Expands `Ops.MULTI` into multiple single-device operations (if present).
    *   **Rangeify (`get_rangeify_map`)**: Identifies loops/ranges in the graph (e.g., for Flash Attention or large reductions) and inserts `Ops.RANGE` / `Ops.ENDRANGE`.

4.  **`create_schedule`**:
    *   **Topological Sort**: Sorts the graph.
    *   **Grouping**: Identifies "Kernels".
        *   A `KERNEL` UOp effectively groups a subgraph that can be fused into a single shader/kernel.
        *   Dependencies (children) determine breaks.
    *   **Linearization**: Flattens the dependency graph into a list.

5.  **Reconstruction**:
    *   Replaces `LUNIQUE` back to `BUFFER` (allocating new buffers if needed).
    *   Allocates multi-buffers if needed.

6.  **Memory Planning**:
    *   Calls `memory_planner` to assign specific `Buffer` objects, reusing memory where possible.

## 2. Rangeify

This is a critical advanced feature. It detects patterns that should be executed as a loop inside the scheduler (rather than unrolled in the kernel or handled by Python loop).
*   Allows implementing things like "Ring Attention" or tiling large matrix multiplications without materializing huge intermediates.

## 3. Multi-Device Scheduling

Splits the graph into per-device shards.
*   Handles synchronization (copies) between devices.
*   Ensures that if you run `t.shard(...)`, the computation actually runs in parallel on all GPUs.
