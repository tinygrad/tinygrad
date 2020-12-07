# Installation

```bash
pip3 install git+https://github.com/geohot/tinygrad.git --upgrade
```

## Example

```python3
from tinygrad.tensor import Tensor

x = Tensor.eye(3)
y = Tensor([[2.0,0,-2.0]])
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```
