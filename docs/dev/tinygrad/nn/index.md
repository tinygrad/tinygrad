# NN Implementation Details

`tinygrad/nn/` provides the neural network building blocks.

## 1. Layers

Implemented as classes inheriting from nothing (just Python classes) or using `tinygrad.tensor.Tensor`.

*   **`Linear`**: `x @ w + b`.
*   **`Conv2d`**: `x.conv2d(w, b, ...)`.
*   **`BatchNorm`**: `x.batchnorm(...)`. Track running stats if `Tensor.training`.

## 2. Optimizers (`optim.py`)

*   **`Optimizer`**: Base class.
    *   `params`: List of tensors to update.
    *   `zero_grad()`: Sets `.grad = None` for all params.
    *   `step()`: Updates params using `.grad`.
    *   `realize()`: Forces realization of new param values (needed because update is lazy).

*   **Implementations**: `SGD`, `Adam`, `AdamW`, `RMSprop`.
    *   They implement the update math using `Tensor` ops.

## 3. State Management (`state.py`)

*   **`get_parameters(model)`**: Reflection. Walks the object tree to find `Tensor` attributes.
*   **`get_state_dict(model)`**: Returns `dict[name, Tensor]`.
*   **`safe_save` / `safe_load`**:
    *   Uses the **SafeTensors** format.
    *   Zero-copy loading using `mmap` (Tensor from `pathlib.Path`).
    *   Handling of device mapping (load to CPU vs disk vs GPU).

## 4. `torch` Compatibility

`tinygrad/nn/torch.py` provides helpers to load PyTorch weights (pickled or safetensors) and convert logic.
