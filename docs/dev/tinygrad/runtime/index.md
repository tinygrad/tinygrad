# Runtime

The `tinygrad/runtime/` directory implements the runtime support for various devices. Each file typically corresponds to a backend.

## Structure

- `ops_*.py`: Implementation of `Compiled` (or `Interpreter`) device interfaces.
- `graph/`: Graph execution implementations (e.g., CUDA Graphs, Metal Performance Shaders).
- `support/`: Helper functions and classes for runtime implementations (e.g., memory management, compiler interfaces).
- `autogen/`: Auto-generated bindings (ctypes/structs) for low-level APIs.

## Device Implementations

Each `ops_*.py` file typically defines:
1. **`Allocator`**: How to allocate memory on the device.
2. **`Compiler`** (if applicable): How to compile source code to machine code/bytecode.
3. **`Renderer`** (if applicable): How to generate source code from UOps.
4. **`Device`** class: Inherits from `Compiled`. Registers the allocator, compiler, and renderer.

### Examples
- **`ops_python.py`**: Pure Python interpreter backend.
- **`ops_cpu.py`**: CPU backend using Clang/LLVM.
- **`ops_cuda.py`**: NVIDIA CUDA backend.
- **`ops_metal.py`**: Apple Metal backend.
- **`ops_webgpu.py`**: WebGPU backend.
- **`ops_disk.py`**: Disk-mapped buffers.

## Common Components (`runtime/support/`)

- **`memory.py`**: Generic memory allocators (e.g., `TLSFAllocator`).
- **`compiler_*.py`**: Helpers for invoking external compilers (clang, nvcc, etc.).
- **`elf.py`**: ELF binary parsing/loading.
