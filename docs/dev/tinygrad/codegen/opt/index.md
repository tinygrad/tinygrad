# Codegen Optimization Implementation Details

`tinygrad/codegen/opt/` implements the optimizer (Beam Search) that selects the best kernel configuration.

## 1. `search.py` (`kernel_optimize_search`)

The main entry point for automatic tuning.

### 1.1 Beam Search
*   **Goal**: Find the sequence of `Opt` (optimizations) that minimizes execution time.
*   **State**: `Kernel` object.
*   **Actions**: `UPCAST`, `UNROLL`, `LOCAL`, `GROUP`, `FLOAT4`.
*   **Process**:
    *   Starts with a base kernel.
    *   Generates all valid "next steps" (valid actions).
    *   Scores them (using `estimates` or real measurement).
    *   Keeps top K (`BEAM`) kernels.
    *   Repeats until no improvements found.

## 2. `tc.py` (Tensor Cores)

Logic to pattern-match and enable Tensor Cores (WMMA).

*   **`TensorCore` dataclass**: Defines the capabilities (e.g., `16x16x16` float16 mult).
*   **`apply_tensor_cores`**:
    *   Looks for `REDUCE` patterns (matrix multiply).
    *   Checks if dimensions can be tiled to match the TC size.
    *   Applies `Opt(TC, ...)` which forces specific unrolling and local sizes.

## 3. `heuristic.py`

Hand-crafted rules for optimization when Beam Search is disabled (`NOOPT`).
*   Example: "If reduction, use group size 32".
*   Example: "If elementwise, try to vectorize float4".
