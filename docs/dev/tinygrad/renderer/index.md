# Renderer

The `tinygrad/renderer/` directory contains code for converting UOps into target-specific code (e.g., C, CUDA, Metal, LLVM IR).

## `__init__.py`

Defines the `Renderer` base class and `ProgramSpec`.

### `ProgramSpec`
Holds the specification of a compiled program:
- `name`: Name of the kernel.
- `src`: Source code.
- `device`: Device name.
- `uops`: List of UOps.
- `global_size`, `local_size`: Launch dimensions.
- `vars`, `globals`: Variables and global buffers.

### `Estimates`
Helper class to estimate FLOPS and memory usage from UOps.

### `Renderer` Class
Abstract base class for all renderers.
- `render(uops: list[UOp]) -> str`: Main method to implement. Converts UOps to source code.
- `code_for_op`: Dictionary mapping `Ops` to functions that generate code for them.
- Configuration flags: `has_local`, `has_shared`, `supports_float4`, etc.

## Specific Renderers

- **`cstyle.py`**: Base class for C-style languages (C, CUDA, Metal, OpenCL, WGSL). Implements common logic for control flow, types, and function definitions.
- **`llvmir.py`**: Renders UOps to LLVM IR.
- **`ptx.py`**: Renders UOps to NVIDIA PTX assembly.
- **`nir.py`**: Renders UOps to NIR (for Mesa/AMD).
- **`wgsl.py`**: Renders UOps to WebGPU Shading Language.
