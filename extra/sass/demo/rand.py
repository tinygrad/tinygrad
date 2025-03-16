# NV=1 SASS=1 SM=80 python rand.py
from tinygrad import Tensor

a = Tensor.rand(3, 4).realize()
print(a.tolist())

