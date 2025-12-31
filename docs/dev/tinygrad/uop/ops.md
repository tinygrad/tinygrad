# Ops Implementation Details

`tinygrad/uop/ops.py` is the foundation of the IR.

## 1. The `UOp` Class

### 1.1 `__init__`
*   **Memoization**: Uses `UOpMetaClass.ucache` to ensure UOps are unique and reused.
    *   Key: `(op, dtype, src, arg, tag)`.
    *   Uses `weakref` to avoid leaks.
*   **Validation**: If `SPEC > 1`, runs `type_verify`.

### 1.2 `replace`
*   Creates a new UOp with modified fields.
*   Optimization: If args match existing, returns `self`.

### 1.3 `toposort`
*   Iterative DFS using a stack.
*   Returns `dict` (ordered keys = sort order).
*   Handling `visited`: `cache` dict stores results.

### 1.4 `substitute`
*   Inputs: `dvars` (replacement map).
*   Process:
    *   Calls `graph_rewrite` with a special pattern `_substitute`.
    *   Pattern matches `Ops` (all) and replaces if in `dvars`.

## 2. Shape Tracking (`_shape`)

This replaced `shapetracker.py`. The UOp *is* the shape tracker.

### 2.1 Recursive Property
*   `_shape` is a `recursive_property`.
*   It avoids recursion limit by using an iterative stack approach to compute the property.

### 2.2 Logic
*   **`RESHAPE`**: Returns `marg` (argument). Validates size.
*   **`PERMUTE`**: Permutes `src[0].shape` based on `arg`.
*   **`EXPAND`**: Checks broadcast compatibility.
*   **`PAD`**: Adds padding to shape.
*   **`ALU`**: Returns `src[0].shape` (asserts all srcs match).
*   **`REDUCE_AXIS`**: Sets dimensions in `axis` to 1.

## 3. Simplification (`simplify`)

The symbolic algebra engine.

### 3.1 `graph_rewrite`
The driver for simplification.
1.  **Context**: `RewriteContext`.
2.  **Stack**: Iterative traversal.
3.  **Stages**:
    *   **0**: Push srcs.
    *   **1**: All srcs processed. Check pattern match (`pm.rewrite`).
        *   If match: create `new_uop`.
        *   If `new_uop` has new sources, push them.
    *   **2**: Link result.

### 3.2 `PatternMatcher` (`upat.py`)
*   **`rewrite(uop)`**:
    *   Lookups up patterns in `pdict[uop.op]`.
    *   Calls `match(uop)`.
    *   If valid, calls the transform function.
*   **`UPat.match`**:
    *   Checks `op`, `dtype`, `arg`.
    *   Recursively checks `src`.
    *   Populates `store` (wildcard captures).

## 4. Visualization

### 4.1 `render`
*   Used for C-style code generation.
*   Uses a `renderer` PatternMatcher.
*   e.g., `(UPat(Ops.ADD, name="x"), lambda ctx,x: f"({ctx[x.src[0]]}+{ctx[x.src[1]]})")`.

### 4.2 `pyrender`
*   Generates Python code to recreate the graph.
*   Used for the `viz` UI.
*   Handles deduping (assigning variables `c0`, `c1`...).
