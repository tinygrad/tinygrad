import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor
import time
dtype_in = dtypes.half if getenv("HALF") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else None
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
FLOP = N*N*N*2
for i in range(CNT):
  if i > 0 and getenv("RAND", 0) != 0:
    a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
  else:
    a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
  start = time.monotonic()
  c = a.matmul(b, acc_dtype=acc_dtype).realize()
  end = time.monotonic()
  s = end - start
  print(f"{(FLOP*1e-9)/s:.2f} GFLOP/S, {s*1e3:.2f} ms")
comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
nc = c.numpy()
np.testing.assert_allclose(nc, comp, atol=1e-4, rtol=3e-2)
