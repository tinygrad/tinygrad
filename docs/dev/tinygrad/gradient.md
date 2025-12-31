# Gradient

`tinygrad/gradient.py` implements automatic differentiation (autograd) for tinygrad.

## `compute_gradient`

The main entry point is `compute_gradient(root: UOp, root_grad: UOp, targets: set[UOp]) -> dict[UOp, UOp]`.
It computes the gradients of `root` with respect to `targets`, given the gradient of `root` (`root_grad`).

### Algorithm
1. **Topological Sort**: It performs a backward pass traversal (reverse topological sort) starting from `root`.
2. **Deep Walk**: It uses `_deepwalk` to find the path from `root` to `targets`, ensuring we only compute relevant gradients.
3. **Pattern Matching**: It uses `PatternMatcher` (`pm_gradient`) to define gradient rules for each operation.

## Gradient Rules (`pm_gradient`)

The `pm_gradient` dictionary maps UOp operations to functions that compute their gradients.
Examples:
- `ADD`: Gradient flows equally to both inputs.
- `MUL`: Gradient for input `x` is `grad * y`, for `y` is `grad * x`.
- `MAX`: Gradient flows only to the input that was the maximum.
- `REDUCE_AXIS`: Uses `reduce_gradient` helper.

## Metadata
The module also handles backpropagation of metadata for debugging and visualization purposes.
