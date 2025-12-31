# Device Implementation Details

`tinygrad/device.py` acts as the hardware abstraction layer. It manages devices, memory allocation (`Buffer`), and the compilation pipeline (`Compiled`).

## 1. `Device` Registry

*   **`_Device` Class**: A singleton that manages available devices.
    *   **Registry**: Scans `tinygrad/runtime/ops_*.py` for classes named `*Device` (e.g., `NVDevice`, `MetalDevice`).
    *   **Canonicalization**: Maps variants like `GPU:0` to `GPU`.
    *   **`DEFAULT`**: Determines the default device (e.g., checks env vars `GPU`, `METAL`, `CUDA` or falls back to `CPU`).
    *   **`__getitem__`**: Instantiates and returns the device class on first access.

## 2. `Buffer` Class

Represents a contiguous block of memory on a specific device.

### 2.1 Attributes
*   **`device`**: String identifier (e.g., "GPU").
*   **`size`**: Number of elements.
*   **`dtype`**: Data type (`DType`).
*   **`options`**: `BufferSpec` (image, external_ptr, etc.).
*   **`_buf`**: The opaque handle to the underlying memory (e.g., `CUdeviceptr`, `id<MTLBuffer>`).

### 2.2 Memory Management
*   **`allocate()`**: Calls `device.allocator.alloc()`.
*   **`ensure_allocated()`**: Allocates if not already allocated.
*   **`copyin(mv)`**: Copies data from a Python `memoryview` to the device.
*   **`copyout(mv)`**: Copies data from the device to a Python `memoryview`.
*   **`view()`**: Creates a new `Buffer` object sharing the same underlying memory (`_base`) but with an offset. Used for `reshaping`/`slicing` without copy.

### 2.3 Life Cycle
*   **`__init__`**: sets up metadata.
*   **`__del__`**: calls `deallocate()`.
*   **`deallocate()`**: calls `allocator.free()`.

## 3. `Allocator`

Abstract base class for memory allocation.

### 3.1 `LRUAllocator`
A generic caching allocator.
*   **Cache**: `dict[tuple[size, options], list[opaque_ptr]]`.
*   **Logic**: When `free` is called, instead of freeing the pointer on the device, it adds it to the cache. When `alloc` is called, it checks the cache first.
*   **Why?**: Reducing synchronization overhead with the driver (alloc/free is slow).

## 4. `Compiled` Device

This is the base class for most accelerators (GPU, etc.). It standardizes the pipeline:
**AST -> Render -> Compile -> Run**.

### 4.1 Components
*   **`allocator`**: Handles memory.
*   **`renderer`**: Converts `UOp` graph to source code (C, PTX, LLVM IR, etc.).
*   **`compiler`**: Compiles source code to binary/executable (e.g., `clang`, `nvcc`, `metal`).
*   **`runtime`**: Loads and executes the binary.

### 4.2 Compilation (`_select_compiler_pair`)
Allows run-time selection/overriding of the compiler/renderer pair (e.g., `CPU_LLVM=1`, `CPU_CC=clang`).

## 5. Profiling

*   **`ProfileEvent`**, **`ProfileDeviceEvent`**: Data structures for capturing trace events.
*   **`finalize_profile`**: Dumps profile data to disk (pickle) for `viz` to consume.

## 6. Supported DTypes (`is_dtype_supported`)

Checks hardwre capabilities.
*   e.g., `bfloat16` support varies by device (Ampere+, M1+, etc.).
*   `float16` support.
*   `float64` support (often missing on consumer GPUs).

## 7. `MultiBuffer`

Experimental support for sharding a buffer across multiple devices.
*   Holds a list of `bufs` (one per device).
*   Used by sharded Tensors.
