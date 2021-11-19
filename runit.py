from tinygrad.tensor import Tensor

x = Tensor.randn(3, 3)
y = Tensor.randn(3, 3)

z = x.cat(y, dim=2)
