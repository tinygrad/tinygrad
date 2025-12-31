# Helpers Implementation Details

`tinygrad/helpers.py` is a grab-bag of utility functions, but several are critical to the architecture.

## 1. Context Management (`ContextVar`)

Tinygrad relies heavily on environment variables for configuration (`DEBUG`, `JIT`, `WINO`).

### 1.1 `ContextVar`
*   **Initialization**: Reads from `os.environ` on startup.
*   **Caching**: Caches the converted value (usually int).
*   **Context Manager**:
    *   `Context` decorator allowing temporary overrides.
    *   `with Context(DEBUG=4): ...`
    *   Useful for debugging specific sections or disabling optimizations (`NOOPT`) for a block.

### 1.2 Global Flags
*   **`DEBUG`**: Logging level (1-5).
*   **`IMAGE`**: Enable ImageDType support (texture memory).
*   **`JIT`**: Enable JIT compilation.
*   **`BEAM`**: Beam search depth for kernel optimization.
*   **`WINO`**: Enable Winograd convolution.

## 2. Caching (`diskcache`)

A persistent cache using SQLite.
*   **Path**: `~/.cache/tinygrad/cache.db`.
*   **`diskcache` Decorator**: Memoizes function calls to disk based on arguments.
    *   Used for caching compiled kernels.
*   **`diskcache_get` / `diskcache_put`**: Low-level API.
*   **Why SQLite?**: Concurrent access safety (file locking), single file management.

## 3. Profiling

### 3.1 `GlobalCounters`
Static class tracking stats.
*   **`global_ops`**: Total FLOPS.
*   **`global_mem`**: Total memory bandwidth.
*   **`time_sum_s`**: Total kernel execution time.
*   **`mem_used`**: Current memory usage.

### 3.2 `Profiling` & `Timing`
Context managers for measuring python-side overhead.

### 3.3 `cpu_profile`
Captures events for the `viz` tool.
*   `ProfileRangeEvent`: Start/End times.
*   `ProfilePointEvent`: Instant events.

## 4. Utilities

### 4.1 Math
*   **`prod(x)`**: `reduce(mul, x)`. Returns 1 for empty list.
*   **`ceildiv(n, d)`**: `(n + d - 1) // d` (integer ceiling division).
*   **`round_up(n, d)`**: Rounds `n` up to nearest multiple of `d`.

### 4.2 Formatting
*   **`colored`**: ANSI color codes.
*   **`time_to_str`**: Formats nanoseconds to `ms`, `us`.

### 4.3 Iterables
*   **`flatten`**: Flattens one level of nesting.
*   **`dedup`**: Removes duplicates while preserving order.
*   **`argsort`**: Returns indices that sort a list.

### 4.4 Data Loading
*   **`fetch(url)`**: Downloads file, caches it, handles gzip.

## 5. Other

*   **`Metadata`**: Dataclass for source location tracking (file, line, function).
*   **`tqdm`**: Minimal progress bar implementation.
