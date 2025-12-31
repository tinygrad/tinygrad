# Renderer Implementation Details

`tinygrad/renderer/__init__.py` and subfiles handle code generation.

## 1. `Renderer` Base Class

Defines the interface for a backend code generator.

*   **`render(uops) -> str`**: The core method.
*   **Flags**: `has_local`, `has_shared`, `supports_float4`, `tensor_cores`.
*   **`code_for_op`**: Map `Ops` to string templates (e.g., `Ops.ADD: lambda a,b: f"({a}+{b})"`).

## 2. `ProgramSpec`

Contains everything needed to compile a kernel.
*   **`name`**: Kernel name.
*   **`src`**: Generated source code.
*   **`uops`**: The linear list of UOps that generated this code.
*   **`global_size` / `local_size`**: Launch bounds.
*   **`vars`**: Symbolic variables (e.g., batch size).

## 3. C-Style Renderers (`cstyle.py`)

Used for C, CUDA, Metal, OpenCL, WGSL.
*   Inherits from `Renderer`.
*   Implements `render` by iterating UOps and building a C-like AST string.
*   Handles:
    *   Type declarations (`float4`).
    *   Loops (`for (int i = ...)`).
    *   Barriers.
    *   Load/Store syntax.

## 4. LLVM / PTX Renderers

*   **`llvmir.py`**: Generates LLVM IR directly (text format). Used for CPU (via Clang/LLVM) or AMD GPU (via COMGR).
*   **`ptx.py`**: Generates NVIDIA PTX assembly. Used if `nvcc` is not available or for direct control.

## 5. `Estimates`

Static analysis of UOps to estimate:
*   **`ops`**: FLOPS.
*   **`lds`**: Bytes loaded from global memory.
*   **`mem`**: Total memory footprint.
Used for deciding optimization strategies (Beam Search).
