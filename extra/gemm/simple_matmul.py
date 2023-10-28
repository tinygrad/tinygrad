import numpy as np
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
dtype_in = dtypes.half if getenv("HALF") else dtypes.float
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
for i in range(CNT):
  if i > 0 and getenv("RAND", 0) != 0:
    a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
  c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).float().sum(axis=2).realize() if getenv("ACCUM_FP32") else (a @ b).realize()
comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
nc = c.numpy()
np.testing.assert_allclose(nc, comp, atol=1e-4, rtol=1e-2)
