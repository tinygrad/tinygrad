# Gradient Implementation Details

`tinygrad/gradient.py` is the heart of the automatic differentiation system. It translates a forward `UOp` graph into a backward `UOp` graph representing the gradients.

## 1. `compute_gradient`

This is the main entry point called by `Tensor.backward`.

```python
def compute_gradient(root: UOp, root_grad: UOp, targets: set[UOp]) -> dict[UOp, UOp]:
    # ...
```

### 1.1 Inputs
*   **`root`**: The `UOp` representing the loss (scalar).
*   **`root_grad`**: The gradient of the loss (usually 1.0).
*   **`targets`**: The set of `UOp`s (weights, inputs) we want gradients for.

### 1.2 Algorithm

1.  **Initialize Grads**: Starts with `grads = {root: root_grad}`.
2.  **Reverse Topological Walk (`_deepwalk`)**:
    *   It performs a topological sort of the graph starting from `root`.
    *   It filters nodes to only include those on a path to `targets`. This optimization prevents computing gradients for irrelevant branches.
    *   It iterates in *reverse* order (from output to input).
3.  **Local Gradient Computation**:
    *   For each node `t0`, it looks up its gradient `ctx` (from `grads`).
    *   It calls `pm_gradient.rewrite(t0, ctx=ctx)` to get the gradients for `t0`'s inputs (`t0.src`).
    *   The result is a tuple of gradients corresponding to `t0.src`.
4.  **Accumulation**:
    *   The computed gradients for inputs are added to `grads`.
    *   If an input already has a gradient (from another path), the new gradient is added to it (`grads[k] = grads[k] + v`). This implements the multivariate chain rule sum.
5.  **Metadata Propagation**:
    *   It copies metadata (function names, line numbers) from the forward nodes to the backward nodes, marking them with `backward=True`. This is crucial for debugging and visualization.

## 2. Gradient Patterns (`pm_gradient`)

`pm_gradient` is a `PatternMatcher` that defines the derivative rules for each `Ops`.

### 2.1 Arithmetic Ops
*   **`ADD`**: `z = x + y` -> `dx = dz`, `dy = dz`.
*   **`MUL`**: `z = x * y` -> `dx = dz * y`, `dy = dz * x`.
*   **`SUB`**: `z = x - y` -> `dx = dz`, `dy = -dz`.
*   **`MAX`**: `z = max(x, y)` -> Gradient flows to the larger input. Uses `WHERE`.
    *   `dx = (x > y) ? dz : ((x == y) ? 0.5 * dz : 0)` (splitting gradient for equality).

### 2.2 Unary Ops
*   **`RECIPROCAL`**: `y = 1/x` -> `dx = -dz / (x^2)`.
*   **`SIN`**: `y = sin(x)` -> `dx = dz * cos(x)`.
*   **`LOG2`**: `y = log2(x)` -> `dx = dz / (x * ln(2))`.
*   **`EXP2`**: `y = 2^x` -> `dx = dz * 2^x * ln(2)`.
*   **`SQRT`**: `y = sqrt(x)` -> `dx = dz / (2 * sqrt(x))`.

### 2.3 Movement Ops
*   **`RESHAPE`**: `dx = dz.reshape(x.shape)`.
*   **`PERMUTE`**: `dx = dz.permute(argsort(order))`.
*   **`EXPAND`**: `dx = dz.sum(axis=expanded_axes)`. Gradient of expand is sum/reduce.
*   **`PAD`**: `dx = dz.shrink(...)`. Gradient of pad is shrink.
*   **`SHRINK`**: `dx = dz.pad(...)`. Gradient of shrink is pad.

### 2.4 Reduction Ops (`REDUCE_AXIS`, `REDUCE`)
*   **`SUM` (`ADD`)**: `z = x.sum()` -> `dx = dz.expand(x.shape)`. Gradient of sum is expand (broadcast).
*   **`MAX`**:
    *   Requires re-computing which element was max.
    *   `mask = (x == z.expand())`
    *   `dx = dz.expand() * mask / count`. (Handling ties by dividing gradient).

### 2.5 Special Ops
*   **`WHERE`**: `z = cond ? x : y`.
    *   `dx = cond ? dz : 0`
    *   `dy = !cond ? dz : 0`
    *   `dcond = None` (cond usually not differentiable, or boolean).
*   **`CAST`**: `dx = dz.cast(x.dtype)`.
*   **`CONTIGUOUS`**: `dx = dz`. (No-op for math, just layout).

## 3. Helper Functions

### 3.1 `_deepwalk`
Traverses the graph to find all nodes between `root` and `targets`.
*   Uses `in_target_path` map to cache reachability.
*   Stops at `DETACH` or `ASSIGN` ops (gradients don't flow through them).

### 3.2 `reduce_gradient`
Helper for `REDUCE` and `REDUCE_AXIS` backward.
*   Handles the shape matching (broadcast back to input).
*   Implements the logic for `ADD` (broadcast), `MUL` (broadcast * other), and `MAX` (masking).

## Why this design?
*   **Symbolic Differentiation**: By operating on `UOp`s, the gradient is just another computation graph. This allows the JIT and optimizer to fuse and optimize the backward pass just like the forward pass.
*   **Memory Efficiency**: We don't store "saved tensors" aggressively like PyTorch eagerly does. We rebuild parts of the graph if needed, or rely on the `UOp` graph structure to keep references.
*   **Flexibility**: New operations can support autograd simply by adding a rule to `pm_gradient`.
