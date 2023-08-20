import numpy as np
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
dtype = dtypes.half if getenv("HALF") else dtypes.float
N = 4096
a, b = Tensor.rand(N, N, dtype=dtype).realize(), Tensor.rand(N, N, dtype=dtype).realize()
for i in range(10):
  c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).float().sum(axis=2).realize()
print((c.numpy() - (a.numpy().astype(np.float32) @ b.numpy().astype(np.float32))).mean())
