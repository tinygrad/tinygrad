# Plan for Removing View and ShapeTracker from the Torch Backend

The objective is to completely remove the `View` and `ShapeTracker` classes from `extra/torch_backend/backend.py`. This is a significant refactoring task that aims to simplify the tinygrad/torch compatibility layer. This document outlines a plan of attack.

## 1. Understand the Current Implementation

The first step is to deeply understand the roles of `View` and `ShapeTracker`.

-   **`View` class:** Represents a single view on a tensor's data. It holds `shape`, `strides`, `offset`, `mask`, and a `contiguous` flag. It has methods to perform view operations like `pad`, `shrink`, `permute`, `reshape`, etc. The `to_valid_uop` method is crucial as it generates a `UOp` graph for indexing.
-   **`ShapeTracker` class:** A collection of `View` objects (`views: tuple[View, ...]`). It represents a chain of view operations. It provides methods for movement ops (`pad`, `shrink`, `reshape`, etc.) that operate on the last view in the chain. It also has a `simplify` method that merges views.
-   **Usage:**
    -   `_get_view_st` creates a `ShapeTracker` for a tensor, which is then used by view-creating `aten` ops.
    -   `_as_strided_impl` and `_strided_view_ops` rely heavily on `ShapeTracker` to handle `as_strided` calls, which can create very complex views. The `_to_movement_ops` function seems to be a complex piece of logic to convert a `ShapeTracker` into a sequence of simpler movement ops.
    -   Many `aten` implementations for view operations are wrappers that record the operation and update the `_view_ops` on the tensor, which are then used to reconstruct the `ShapeTracker`.

## 2. Research `uops` and Migration History

To find a suitable replacement, we need to investigate alternatives within the tinygrad ecosystem.

-   **`uops` for Movement:** Analyze how tinygrad's `uops` (micro-operations) framework can represent tensor views and movement operations. The symbolic engine (`tinygrad.uop.symbolic`) is likely key here. We need to understand how to express strides, offsets, and masks using `uops`.
-   **Git History:** Review the git log for `extra/torch_backend/backend.py` and other related files like `tinygrad/shape/shapetracker.py` (if it exists or existed). This will provide historical context on why `ShapeTracker` was introduced and how it has evolved. It might also show how it was replaced in other parts of the codebase.

## 3. Formulate a Replacement Strategy

The core idea is to replace the `ShapeTracker` object with a more direct representation of the tensor's view, likely using tinygrad's existing features.

-   **Track Strides and Offset:** The main requirement is to track strides and offset. Instead of a stack of `View` objects, we could perhaps store the final `strides` and `offset` directly on the `Tensor` object (or a wrapper).
-   **Symbolic Representation:** Leverage `tinygrad.uop.symbolic` to handle symbolic shapes and strides, which is something `ShapeTracker` already does.
-   **Direct `UOp` Generation:** Instead of `ShapeTracker.to_valid_uop`, we could build the indexing `UOp` graph directly from the simplified view information (shape, strides, offset).
-   **`as_strided` Handling:** The `as_strided` operation is the most complex case. We need a robust way to translate `(shape, stride, offset)` into a representation that tinygrad's backend can handle. This might involve creating a `UOp` graph that computes the correct indices. The current `_to_movement_ops` is a monster and should be replaced with a more direct, `uop`-based approach if possible.

## 4. Step-by-Step Implementation Plan

This will be an iterative process.

1.  **Introduce new view tracking:** Add new attributes to `Tensor` (or a wrapper object if that's cleaner) to store view information, e.g., `view_strides`, `view_offset`, `view_mask`.
2.  **Start with simple view ops:**
    -   Modify a simple view op like `permute`. Instead of creating a `ShapeTracker` and calling `.permute()`, directly calculate the new strides and update the `view_strides` attribute.
    -   Do the same for `expand`, `shrink`, `pad`.
3.  **Tackle `reshape`:** `reshape` is more complex as it can change strides in non-trivial ways. The logic inside `View.reshape` will need to be ported to work with the new view representation.
4.  **Address `as_strided`:** This is the most challenging part.
    -   Implement a function that takes `(shape, stride, offset)` and creates a tensor view.
    -   This will likely involve creating a `LazyOp` with a `LoadOp` that has access to the underlying buffer and the view parameters. The `to_uop` for this `LoadOp` would then generate the correct indexing logic.
5.  **Remove `ShapeTracker` and `View`:** Once all view operations are migrated to the new system, the `ShapeTracker` and `View` classes can be deleted. This will involve cleaning up all the places they were used.
6.  **Testing:** Throughout the process, run the `extra/torch_backend/test.py` test suite to ensure that the functionality remains correct. Pay close attention to tests for view operations and `as_strided`.

This plan provides a high-level roadmap. The exact implementation details will become clearer as we progress through the steps.
