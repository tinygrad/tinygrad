# Realize Implementation Details

`tinygrad/engine/realize.py` handles the lowering and execution of scheduled items.

## 1. `ExecItem`

Represents a unit of work.
*   **`ast` (`UOp`)**: The operation to perform (if it's a kernel).
*   **`bufs` (`list[Buffer]`)**: The input and output buffers.
*   **`prg` (`Runner`)**: The lowered program (compiled kernel, copy op, etc.).

### 1.1 `lower()`
Converts the `ast` into a `Runner`.
*   Uses `si_lowerer` (PatternMatcher).
*   `Ops.SINK`: Calls `get_runner` (compilation).
*   `Ops.COPY`: Creates `BufferCopy` or `BufferXfer`.
*   `Ops.BUFFER_VIEW`: Creates `ViewOp`.

## 2. Runners

### 2.1 `CompiledRunner`
Manages compiled kernels (GPU/CPU).
*   **`p` (`ProgramSpec`)**: Metadata (globals, locals, name).
*   **`lib`**: The binary code.
*   **`clprg`**: The runtime handle (created by `Device[d].runtime(..., lib)`).
*   **`__call__`**:
    *   Calculates global/local launch dimensions.
    *   Optimizes local size if needed (`optimize_local_size`).
    *   Calls the runtime program.

### 2.2 `BufferCopy` / `BufferXfer`
Handles memory movement.
*   `BufferCopy`: Standard copy (CPU <-> Device).
*   `BufferXfer`: Peer-to-Peer copy (Device <-> Device) if supported.

### 2.3 `ViewOp`
A virtual operation. It asserts that the destination buffer is a view of the source buffer.

## 3. Execution (`run_schedule`)

Loops through the list of `ExecItem`s.
1.  **Lower**: Ensures the item has a `prg`.
2.  **JIT Capture**: If `capturing`, adds the item to the JIT cache and skips execution.
3.  **Run**: Calls `ei.run()`.
4.  **Stats**: Updates `GlobalCounters` (ops, memory, time).

## 4. Compilation Cache (`method_cache`)

Memoizes `CompiledRunner` creation based on:
*   Device
*   AST structure
*   Optimization flags (BEAM, NOOPT).

This prevents recompiling the same kernel repeatedly.
