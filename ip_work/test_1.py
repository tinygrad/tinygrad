from tinygrad.tensor import Tensor
from tinygrad.device import Device


a = Tensor([1.0,2.0])
b = Tensor([3.0,4.0])
c = a.dot(b)

d = c.numpy()
print(d)

