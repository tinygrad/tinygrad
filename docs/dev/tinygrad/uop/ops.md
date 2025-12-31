# Ops Implementation Details

`tinygrad/uop/ops.py` defines the Micro-Operations (UOps) that form the Intermediate Representation (IR).

## 1. `Ops` Enum

Defines the instruction set.

*   **ALU**: `ADD`, `MUL`, `MAX`, `CMPLT`, `WHERE`, `SIN`, `EXP2`, `LOG2`, `RECIPROCAL`.
*   **Load/Store**: `LOAD`, `STORE`, `INDEX`.
*   **Movement**: `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`.
*   **Buffer**: `BUFFER`, `BUFFER_VIEW`, `CONST`, `DEFINE_VAR`.
*   **Meta**: `SINK` (output), `DETACH`, `BARRIER`.
*   **Loops**: `RANGE`, `END`.

## 2. `UOp` Class

The node class.

### 2.1 Attributes
*   **`op` (`Ops`)**: The operation type.
*   **`dtype` (`DType`)**: The data type of the result.
*   **`src` (`tuple[UOp]`)**: Inputs.
*   **`arg` (`Any`)**: Constant arguments (values, shapes, axes).

### 2.2 Symbolic Math
`UOp`s are used for both tensor data and symbolic shapes.
*   `sint` = `int | UOp`.
*   Methods like `__add__`, `__mul__` on `UOp` create new UOps, building the symbolic expression tree.
*   **`simplify()`**: Applies rewrite rules (from `symbolic.py` logic merged here) to canonicalize expressions (e.g., `x+0 -> x`, constant folding).

### 2.3 Graph Traversal
*   **`toposort`**: Iterative DFS to get topologically sorted list.
*   **`replace`**: Creates a copy with modified fields (immutable style).
*   **`substitute`**: Deep replacement of nodes.

## 3. Pattern Matching (`PatternMatcher`, `UPat`)

A DSL for writing graph rewrite rules.

### 3.1 `UPat`
Describes a pattern.
*   `UPat(Ops.ADD, src=(UPat.var("x"), UPat.var("y")))` matches any ADD.
*   `match(uop, store)`: Recursive check. Populates `store` with captured variables.

### 3.2 `PatternMatcher`
A collection of `(UPat, transformer_function)` pairs.
*   **`rewrite(uop)`**: Checks `uop` against patterns. If match, calls transformer.
*   **`graph_rewrite`**: Systematically applies rewrites to a whole graph until convergence (or single pass).

### 3.3 Example Rule
```python
(UPat(Ops.ADD, src=(UPat.var("x"), UPat.const(None, 0))), lambda x: x)
```
Matches `x + 0` and replaces it with `x`.

## 4. Visualization (`render`, `pyrender`)

*   **`render`**: Generic method to convert UOp to string using a provided matcher.
*   **`pyrender`**: Generates Python code that would reconstruct the UOp graph. Useful for debugging and serialization.
