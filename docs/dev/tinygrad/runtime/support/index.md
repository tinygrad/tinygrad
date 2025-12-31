# Runtime Support Implementation Details

`tinygrad/runtime/support/` contains helper classes shared across multiple backends.

## 1. `memory.py` (`TLSFAllocator`)

Implements the **Two-Level Segregated Fit** memory allocation algorithm.
*   **Purpose**: To manage a large chunk of device memory (e.g., a GPU buffer) and sub-allocate small buffers from it without calling the driver's `malloc` repeatedly.
*   **Mechanism**:
    *   Maintains lists of free blocks, categorized by size (logarithmic buckets).
    *   **Alloc**: Finds the smallest free block that fits. Splits it if necessary.
    *   **Free**: Merges the block with adjacent free blocks (coalescing) to prevent fragmentation.

## 2. `compiler_*.py`

Helpers to invoke external compilers.
*   **`compiler_cuda.py`**: Invokes `nvrtc` (NVIDIA Runtime Compiler).
*   **`compiler_cpu.py`**: Invokes `clang` or `gcc` to compile C code into a shared library (`.so`), which is then loaded via `ctypes`.
*   **`compiler_amd.py`**: Handles HIP/ROCm compilation (using `hipcc` or `comgr`).

## 3. `elf.py`

A pure Python ELF (Executable and Linkable Format) parser/loader.
*   Used to load AMD GPU kernels (which are ELF binaries) directly into memory without external tools.
*   Parses sections, symbols, and program headers.

## 4. `hcq.py`

Base classes for Hardware Command Queue interfaces.
*   Defines the structure of signals, values, and queue pointers.
