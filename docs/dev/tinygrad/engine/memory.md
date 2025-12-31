# Memory Planning Implementation Details

`tinygrad/engine/memory.py` implements memory optimization strategies.

## 1. Goal
Reduce peak memory usage and fragmentation by reusing `Buffer`s that are no longer needed.

## 2. `_internal_memory_planner`

This function takes a list of scheduled kernels (and their buffer usages) and returns a mapping `Buffer -> Buffer` (virtual -> physical).

### 2.1 Algorithm (Greedy / TLSF-like)

1.  **Liveness Analysis**:
    *   Iterates through the schedule.
    *   Records `first_appearance` and `last_appearance` for each buffer.
    *   Creates a timeline of events: `(index, ALLOC, buf)` and `(index, FREE, buf)`.

2.  **Allocation Loop**:
    *   Iterates through the timeline.
    *   **ALLOC**:
        *   Checks `reuse_buffers` (free list) for a buffer of sufficient size/properties.
        *   If found, assigns it.
        *   If not, requests a new allocation from `global_planner`.
    *   **FREE**:
        *   Adds the buffer to `reuse_buffers`.

### 2.2 Suballocation
For devices that support it (e.g., Vulkan, or via `_offset` in allocator), it can allocate one giant buffer and sub-allocate chunks from it.
*   **`TLSFAllocator`**: Two-Level Segregated Fit implementation (standard memory allocator algorithm).
*   Reduces driver allocation overhead (fewer `malloc` calls).

## 3. Integration
*   Called by `complete_create_schedule_with_vars` before returning the schedule.
*   Called by `TinyJit` to bake the memory addresses into the captured graph.
