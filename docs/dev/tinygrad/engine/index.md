# Engine

The `tinygrad/engine/` directory contains the core execution engine of tinygrad. It is responsible for scheduling, compiling (JIT), and executing the computation graph.

## `realize.py`

This file handles the execution of `ExecItem`s.
- **`Runner`**: Base class for execution runners.
- **`CompiledRunner`**: Runs compiled kernels. It manages the `ProgramSpec`, compilation (via `Device.compiler`), and execution (via `Device.runtime`).
- **`ExecItem`**: Represents a single execution unit (kernel, copy, view). It holds the AST, buffers, and metadata.
- **`run_schedule`**: Iterates through a schedule of `ExecItem`s and runs them.

## `schedule.py`

This file is responsible for creating a schedule from a sink `UOp`.
- **`create_schedule`**: Linearizes the graph into a list of kernels.
- **`complete_create_schedule_with_vars`**: The main entry point. It handles graph rewriting, caching, and calls `create_schedule` and `memory_planner`.

## `jit.py`

This file implements the Just-In-Time (JIT) compilation and execution.
- **`TinyJit`**: A decorator that captures the execution of a function and replays it. It optimizes the execution by:
    - Capturing the kernel schedule.
    - Pruning unnecessary kernels.
    - Graphing (using `GraphRunner`) multiple kernels into a single submission (if supported by the device).
    - Managing memory (reusing buffers).
- **`GraphRunner`**: Executes a batch of kernels as a single graph.

## `memory.py`

This file handles memory planning.
- **`memory_planner`**: Optimizes memory usage by reusing buffers that are no longer needed. It uses a greedy approach or a TLSF allocator simulation (`_internal_memory_planner`).
