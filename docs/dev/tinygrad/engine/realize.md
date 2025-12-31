# Realize Implementation Details

`tinygrad/engine/realize.py` manages the execution of the schedule.

## 1. `run_schedule`

The main execution loop.

```python
def run_schedule(schedule:list[ExecItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  while len(schedule):
    ei = schedule.pop(0).lower()
    # ... handle JIT capturing ...
    ei.run(var_vals, do_update_stats=do_update_stats)
```

1.  **Lowering**: Calls `ei.lower()`. This ensures the abstract `UOp` is converted to a concrete `Runner` (compiled binary).
2.  **Capturing**: If `capturing` list is non-empty (from `TinyJit`), adds `ei` to it and **stops**.
3.  **Validation**: If `VALIDATE_WITH_CPU`, runs the kernel on GPU, then copies inputs to CPU, runs on CPU, and compares.
4.  **Run**: Calls `ei.run()`.

## 2. `ExecItem`

A unit of execution.

*   **`lower()`**:
    *   Uses `si_lowerer` (`PatternMatcher`).
    *   `Ops.SINK` -> `get_runner` (compilation).
    *   `Ops.COPY` -> `BufferCopy` / `BufferXfer`.
*   **`run()`**:
    *   Updates `GlobalCounters` (ops, mem).
    *   Calls `self.prg(bufs, var_vals)`.

## 3. Runners

### 3.1 `CompiledRunner` (`__call__`)
1.  **Launch Dims**:
    *   Calculates `global_size` and `local_size` using `self.p.launch_dims(var_vals)`.
    *   If `local_size` is missing (and device supports it), calls `optimize_local_size`.
2.  **Optimize Local Size**:
    *   Tries various power-of-2 local sizes (e.g., [32, 1, 1], [4, 8, 1]).
    *   Runs the kernel (measure execution time).
    *   Picks the fastest.
    *   *Note*: This happens once per kernel (unless cached).
3.  **Execution**:
    *   Calls `self._prg` (the runtime handle).
    *   Passes raw buffer handles (`x._buf`).
    *   Passes symbolic variable values (`vals`).

### 3.2 `BufferCopy`
1.  **Selection**: Checks if devices support P2P (`BufferXfer`) or need host copy (`BufferCopy`).
2.  **Execution**:
    *   `dest.copyin(src.as_buffer())`.
    *   Typically involves `map` -> `memcpy` -> `unmap`.

## 4. `get_runner` (Compilation)

1.  **Cache Key**: `(device, compiler_type, ast_key, optimization_flags)`.
2.  **Check Cache**: `method_cache`.
3.  **Compile**:
    *   Calls `get_program(ast, renderer)`.
    *   Returns `CompiledRunner`.

## 5. `Estimates`

Static analysis logic (`from_uops`).
*   Iterates UOps.
*   `Ops.LOAD`: Adds `dtype.itemsize` to `lds`.
*   `Ops.MUL/ADD`: Adds FLOPs to `ops`.
*   `Ops.RANGE`: Multiplies subsequent costs by loop range.
*   Used for deciding `BEAM` search winners.
