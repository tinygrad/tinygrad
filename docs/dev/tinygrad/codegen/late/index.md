# Codegen Late Implementation Details

`tinygrad/codegen/late/` contains transformations applied at the very end of code generation.

## 1. `linearizer.py`

This module is legacy but still used for some logic. It historically converted the AST into a linear list of operations.
*   Most of its responsibilities have moved to `uop/ops.py` and `scheduler.py`.

## 2. `devectorizer.py`

Handles the cleanup of vector types for backends that don't support them well or have specific constraints.
*   **`no_vectorized_alu`**: A pattern matcher rule.
    *   Splits `float4 + float4` into 4 scalar additions if the backend doesn't support vector ALU.
