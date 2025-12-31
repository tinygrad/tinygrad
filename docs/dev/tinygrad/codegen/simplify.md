# Simplify Implementation Details

`tinygrad/codegen/simplify.py` implements graph simplification passes that run *after* scheduling but *before* rendering. This is distinct from the symbolic simplification in `uop/ops.py` (which runs during graph construction).

## 1. Goal
To reduce the complexity of the UOp graph, specifically focusing on:
*   Redundant ranges/loops.
*   Unnecessary arithmetic in indexing.
*   Simplifying control flow for the renderer.

## 2. Range Simplification

### 2.1 `flatten_range`
*   **Problem**: Nested loops where the inner loop is just a continuation of the outer loop, or effectively a single linear loop split into chunks.
*   **Action**: Flattens nested ranges if possible.
*   **Pattern**: `(UPat((Ops.REDUCE, Ops.STORE, Ops.END), name="r"), flatten_range)`

### 2.2 `simplify_merge_adjacent`
*   **Problem**: Two adjacent loops (e.g., `0..10` and `10..20`) that perform the same operation.
*   **Action**: Merges them into `0..20`.
*   **Logic**:
    *   Checks if ranges are adjacent (`r0.end == r1.start`).
    *   Checks if they are used in the same reductions.
    *   Replaces `r0` and `r1` with `new_range`.
    *   Substitutes `r0` -> `new_range // s1`, `r1` -> `new_range % s1` (if dimensions merged) or simple addition.
    *   *Verification*: Checks `count_divmod` to ensure the merge didn't make indexing *more* complex (e.g., introducing expensive modulos).

## 3. Reduce Simplification

### 3.1 `reduce_unparented`
*   **Problem**: A `REDUCE` op depends on a `RANGE` that isn't actually used in its input expression.
*   **Action**: Removes the range from the reduction (it's effectively a multiply/power by the range size).
*   **Math**: `sum(x for i in 0..10) -> x * 10` (if x doesn't depend on i).

### 3.2 `reduce_collapse`
*   **Goal**: Remove reductions entirely if they can be analytically solved or folded.
*   **Patterns**:
    *   `sum(x+y)` -> `sum(x) + sum(y)`.
    *   `sum(x < c)` where `x` is a range -> analytic count.
    *   `sum(x * gate)` -> `sum(x where gate)`.

## 4. Indexing Simplification

### 4.1 `pm_lower_index_dtype`
*   **Goal**: Ensure all indexing math uses appropriate integer types (`int32` vs `int64`).
*   **Action**: Matches `Ops.INDEX` and its inputs. Casts variables to match pointer width or register size.

### 4.2 `mark_range_mod` / `do_substitute`
*   **Goal**: Optimize `r % c` where `r` is a range.
*   **Action**: Splits the range into `outer * c + inner`.
*   This avoids expensive hardware modulo operations in the inner loop.

## 5. Store Simplification

### 5.1 `cut_store_range` (CPU specific)
*   **Goal**: Split a store loop if it contains a conditional that splits the range cleanly.
*   **Action**: `for i in 0..10: if i < 5: A[i]=x else: A[i]=y` becomes `for i in 0..5: A[i]=x; for i in 5..10: A[i]=y`.
*   **Benefit**: Removes branching inside the loop, enabling vectorization.
