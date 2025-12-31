# Runtime Implementation Details

The `tinygrad/runtime/` directory connects tinygrad to hardware drivers.

## 1. Interface

Every backend (e.g., `ops_gpu.py`) defines a class inheriting from `Compiled` (or using `Device` registration).

### 1.1 `Allocator`
Must implement:
*   `_alloc(size, options)`: Return opaque handle.
*   `_free(handle, options)`.
*   `_copyin(dest, src_mv)`.
*   `_copyout(dest_mv, src)`.

### 1.2 `Compiler`
*   `compile(src) -> bytes`: Compiles source code to binary.
    *   e.g., calls `clang`, `nvcc`, or uses `libmetal`.

### 1.3 Runtime (Function Handle)
*   `__call__(*bufs, global_size, local_size, wait=False)`: Launches the kernel.

## 2. Examples

### 2.1 `ops_python.py`
The reference implementation.
*   **Allocator**: Uses Python `bytearray` (or `memoryview`).
*   **Compiler**: None (interprets UOps directly via `exec`).
*   **Renderer**: `uop.py`'s `pyrender`.

### 2.2 `ops_gpu.py` (OpenCL/Metal/CUDA)
*   **Allocator**: Uses driver APIs (`clCreateBuffer`, `newBufferWithLength`, `cuMemAlloc`).
*   **Compiler**: Driver compilers (`clBuildProgram`, `mtlLibrary`, `nvrtc`).
*   **Renderer**: `cstyle.py` variants.

### 2.3 `ops_disk.py`
*   **Allocator**: Uses `mmap` or file handles.
*   **Operations**: Supports `copy` (load/store) but not `exec` (computation happens by loading to another device).

## 3. Support Infrastructure

*   **`support/compiler_*.py`**: Wrappers for external compilers.
*   **`autogen/`**: `ctypes` bindings generated from C headers (e.g., `cuda.h`, `opencl.h`). This allows tinygrad to run without installing the full SDKs, just the shared libraries.
