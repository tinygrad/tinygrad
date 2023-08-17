import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
N = 4096
a, b = Tensor.rand(N, N, dtype=dtypes.half).realize(), Tensor.rand(N, N, dtype=dtypes.half).realize()
c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).float().sum(axis=2)
print((c.numpy() - (a.numpy().astype(np.float32) @ b.numpy().astype(np.float32))).mean())
