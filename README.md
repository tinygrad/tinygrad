# tinygrad

For something in between a grad and a karpathy/micrograd

The Tensor class is a wrapper around a numpy array

### Example

```python
import numpy as np
from tinygrad.tensor import Tensor

x = Tensor(np.eye(3))
y = Tensor(np.array([[2.0,0,-2.0]]))
z = y.dot(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

### Same example in torch

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

