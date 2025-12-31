# Schedule Implementation Details

`tinygrad/engine/schedule.py` transforms a high-level `UOp` graph into a linear schedule of kernels.

## 1. `complete_create_schedule_with_vars`

This is the scheduler entry point.

### 1.1 Pre-processing
1.  **Graph Rewrite**:
    *   `pm_pre_sched_cache`:
        *   Replaces concrete `BUFFER` UOps with `LUNIQUE` (Logical Unique).
        *   Strips `BIND` values.
    *   *Why?* To canonicalize the graph. Two calls with different buffer addresses but same logic should hit the same schedule cache.
2.  **Cache Lookup**: Checks `schedule_cache`.

### 1.2 Scheduling Pipeline (on Miss)
1.  **Type Verification**: Checks `tensor_spec`.
2.  **Multi-Device**:
    *   If `Ops.MULTI` exists, applies `get_multi_map`.
    *   Transforms single graph into sharded graph (e.g., `ADD` -> 4 `ADD`s on 4 GPUs).
3.  **Rangeify (`get_rangeify_map`)**:
    *   Detects high-level loops (e.g., in Flash Attention or big reduces).
    *   Inserts `Ops.RANGE` control flow.
4.  **`create_schedule`**: Linearizes the graph.
5.  **Caching**: Saves the result.

### 1.3 Post-processing
1.  **Re-binding**:
    *   Replaces `LUNIQUE` back to the actual `BUFFER`s (allocating if necessary).
    *   Allocates `MultiBuffer`s.
2.  **Memory Planning**:
    *   Calls `memory_planner` to assign specific memory addresses.

## 2. `create_schedule`

### 2.1 Graph Analysis (`toposort`)
1.  **Child Graph**: Builds `children` map and `in_degree` count.
    *   It treats a `KERNEL` UOp (or `SINK`) as a node.
    *   Inputs to a kernel are `AFTER` (dependency) or `BUFFER`.

### 2.2 Linearization (BFS/Toposort)
1.  **Queue**: Initialize with nodes having `in_degree == 0`.
2.  **Loop**:
    *   Pop `k` (kernel).
    *   **Expand**:
        *   Extract `buf_uops` (buffers used).
        *   Extract `ast` (the compute).
        *   Identify `bound_ranges` (if part of a loop).
    *   Append to `schedule` list.
    *   **Update Dependencies**: Decrement `in_degree` of children. Add to queue if 0.

### 2.3 Range Expansion
1.  **Loop**: Iterates the linear schedule.
2.  **State**: Tracks `in_ranges` (loop indices).
3.  **Expansion**: If inside a `RANGE`, it might clone the `ExecItem`s for the loop iterations (though often this is handled by the executor or range lowering).

## 3. Rangeify (`rangeify.py`)

This handles "Kernel Fusion" at the schedule level for loops.
*   **Goal**: Run a sequence of kernels in a loop without returning to Python.
*   **Mechanism**:
    *   Identifies `REDUCE` ops that can be split (e.g., accumulation).
    *   Replaces them with `Ops.RANGE` ... `Ops.ASSIGN` ... `Ops.END`.
