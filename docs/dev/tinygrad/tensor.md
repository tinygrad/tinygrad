# Tensor

`tinygrad/tensor.py` defines the `Tensor` class, which is the core data structure in tinygrad. It is similar to PyTorch's Tensor but built on top of tinygrad's `UOp` (micro-op) abstraction.

## The Tensor Class

The `Tensor` class represents a multi-dimensional matrix containing elements of a single data type. It handles high-level operations and autograd.

### Attributes
- `uop`: The underlying `UOp` representing the computation graph for this tensor.
- `requires_grad`: A boolean indicating if gradients need to be computed for this tensor.
- `grad`: Stores the gradient tensor if `backward()` is called.

### Creation
Tensors can be created from:
- Python lists/tuples: `Tensor([1, 2, 3])`
- NumPy arrays: `Tensor(numpy_array)`
- Constants: `Tensor.zeros(shape)`, `Tensor.ones(shape)`, `Tensor.rand(shape)`
- Binary data: `Tensor.from_blob(...)`
- URLs: `Tensor.from_url(...)`

### Operations
The `Tensor` class implements a wide range of operations:
- **Element-wise ops**: `add`, `sub`, `mul`, `div`, `pow`, etc.
- **Matrix multiplication**: `matmul`, `dot`
- **Reduction ops**: `sum`, `max`, `mean`, `var`, `std`
- **Movement ops**: `reshape`, `permute`, `expand`, `shrink`, `pad`
- **Activation functions**: `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, etc.
- **Processing ops**: `conv2d`, `avg_pool2d`, `max_pool2d`

### Autograd
The `backward()` method computes gradients. It triggers the construction of a backward graph using `tinygrad.gradient.compute_gradient`.

### Execution
Operations on Tensors are lazy. They build a UOp graph. Computation is triggered when:
- `realize()` is called.
- Data is accessed (e.g., `.numpy()`, `.item()`, `.data()`).

## Implementation Details

- **Lazy Evaluation**: Most operations just create new UOps.
- **Device Management**: Tensors keep track of their device. Operations automatically handle device checking.
- **Broadcasting**: Implicit broadcasting is supported for element-wise operations using `_broadcasted`.
- **UOp Integration**: The `Tensor` class wraps a `UOp`. All operations eventually translate to UOps.
