# Optim Implementation Details

`tinygrad/nn/optim.py` implements the optimization algorithms. They are designed to be "functional" - they take tensors and gradients, and return new tensor values (assignment is handled by the `Optimizer` base class).

## 1. `Optimizer` Base Class

### 1.1 `__init__`
*   **Params**: Stores list of parameters (`requires_grad=True`) and buffers (stats like momentum).
*   **LR**: Converts learning rate to a `Tensor`. If `CONST_LR` env var is set, it's a constant (faster JIT); otherwise, it's a tensor (dynamic LR).

### 1.2 `step()`
The core logic.
1.  **Check Training**: Ensures `Tensor.training == True`.
2.  **Schedule**: Calls `schedule_step`.
3.  **Realize**: Forces execution of the update kernels.

### 1.3 `schedule_step`
1.  **Fusion (`fused=True`)**:
    *   Flattens all parameters into one giant vector.
    *   Flattens all gradients into one giant vector.
    *   Calls `_step` once on this giant vector.
    *   *Why?* Reduces kernel launch overhead (1 kernel vs N kernels).
    *   Splits the result back into individual parameters.
2.  **No Fusion**:
    *   Calls `_step` with list of params and list of grads.
3.  **Assign**: `param.assign(new_value)`. This creates the assignment UOp.

### 1.4 `zero_grad`
Sets `param.grad = None`. Does not run kernels (lazy).

## 2. Implementations

### 2.1 `SGD` (Stochastic Gradient Descent)
*   **Momentum**: `b = momentum * b + g`.
*   **Nesterov**: `g = g + momentum * b`.
*   **Weight Decay**: `g = g + wd * param` (pre-update) or `param = param * (1 - wd*lr)` (post-update, "decoupled").
*   **Update**: `param = param - lr * g`.

### 2.2 `Adam` / `AdamW` (Adaptive Moment Estimation)
Implemented via `LAMB` class with `adam=True`.
*   **m**: First moment (exponential moving average of grad).
*   **v**: Second moment (EMA of grad squared).
*   **Bias Correction**: `m_hat = m / (1 - b1^t)`, `v_hat = v / (1 - b2^t)`.
*   **Update**: `param = param - lr * m_hat / (sqrt(v_hat) + eps)`.
*   **AdamW**: Decoupled weight decay.

### 2.3 `LAMB` (Layer-wise Adaptive Moments based for Batch training)
*   Optimized version of Adam for large batches.
*   Calculates a trust ratio `r` per layer based on norm of weights vs norm of update.
*   Scales update by `r`.

### 2.4 `LARS` (Layer-wise Adaptive Rate Scaling)
*   SGD with trust ratio.
*   Used for large batch training (e.g., ResNet-50 on ImageNet).

### 2.5 `Muon` (Momentum Orthogonalized optimizer)
*   An optimizer designed for training neural networks, specifically transformers.
*   Uses **Newton-Schulz iterations** to orthogonalize the gradient matrix (effectively preconditioning).
*   `g = newton_schulz(g, steps, coeffs)`.
*   This replaces the adaptive second moment of Adam. It is memory efficient (no second moment buffer).
