# FILE: /tinygrad-docs/tinygrad-docs/docs/examples/example1.md

# Example 1: Basic Usage of tinygrad

In this example, we will demonstrate how to use the `tinygrad` package to perform a simple computation. This example will cover the creation of a basic neural network and how to run a forward pass through it.

## Prerequisites

Make sure you have `tinygrad` installed. You can follow the installation instructions in the [installation guide](../installation.md).

## Example Code

```python
import numpy as np
from tinygrad.tensor import Tensor

# Create input data
x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Define a simple linear layer
class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x.dot(self.weights) + self.bias

# Create a linear layer
linear_layer = Linear(2, 2)

# Perform a forward pass
output = linear_layer.forward(x)

print("Output of the linear layer:")
print(output.data)
```

## Explanation

1. **Input Data**: We create a 2x2 input tensor using NumPy.
2. **Linear Layer**: We define a simple linear layer class that initializes weights and bias.
3. **Forward Pass**: We perform a forward pass through the linear layer and print the output.

This example illustrates the basic usage of `tinygrad` for creating and using a simple neural network layer. For more advanced usage and features, refer to the [usage guide](../usage.md) and the [API reference](../api_reference.md).