# Codegen Implementation Details

`tinygrad/codegen/` handles the transformation of `UOp` graphs into kernel-ready structures.

## 1. `get_program`

The main pipeline in `__init__.py`.
1.  **Linearize**: Calls `linearizer` (if not already linear).
2.  **Optimize**: Simplifies UOps.
3.  **Render**: Calls the device renderer.

## 2. Kernel Optimization

### 2.1 `gpudims.py`
Calculates optimal global/local workgroup sizes.
*   Heuristics based on hardware limits (max threads per block, register usage estimates).
*   Handles image constraints.

### 2.2 `simplify.py`
A collection of `PatternMatcher` rules specifically for optimization *after* scheduling but *before* rendering.
*   e.g., `simplify_alu`: Constant folding for integer arithmetic used in indexing.

### 2.3 `opt/` (Optimization Passes)
*   **`search.py`**: **Beam Search**.
    *   Tries different "applied opts" (loop unrolling, tiling, upcasting, vectorization) on the `Kernel` UOp.
    *   Estimates performance (using `Estimates`) or measures it (if running tuning).
    *   Selects the best configuration.
*   **`tc.py`**: **Tensor Cores**.
    *   Detects matrix multiplication patterns (`WMMA`).
    *   Replaces generic `MUL`/`SUM` with hardware-specific `WMMA` ops if shapes align.

## 3. Late Transformations

*   **`late/linearizer.py`**: This is where the old "Linearizer" logic lives (mostly migrated to `UOp` schedulers now, but some parts remain for specific graph transforms).
