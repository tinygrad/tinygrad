# FILE: /tinygrad-docs/tinygrad-docs/docs/usage.md

# Usage of tinygrad

This document outlines how to use the `tinygrad` package effectively. Below are some basic usage examples and explanations of key features.

## Getting Started

To begin using `tinygrad`, ensure you have installed the package as described in the [installation guide](installation.md).

## Basic Usage

Here are some basic examples to get you started:

### Example 1: Creating a Simple Tensor

```python
from tinygrad.tensor import Tensor

# Create a tensor
x = Tensor([1, 2, 3])
print(x)
```

### Example 2: Performing Operations

```python
# Perform basic operations
y = x + 2
print(y)  # Output: [3, 4, 5]
```

### Example 3: Backpropagation

```python
# Example of backpropagation
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
y.backward()  # Compute gradients
print(x.grad)  # Output: [2.0, 2.0, 2.0]
```

## Key Features

- **Tensors**: The core data structure for numerical computations.
- **Automatic Differentiation**: Easily compute gradients for optimization tasks.
- **GPU Support**: Leverage GPU acceleration for faster computations.

## Conclusion

This is just a brief overview of how to use `tinygrad`. For more detailed information, refer to the [API reference](api_reference.md) and the [examples](examples/example1.md) section.