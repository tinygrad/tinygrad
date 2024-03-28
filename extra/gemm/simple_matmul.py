import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor
dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
    c = a.matmul(b, acc_dtype=acc_dtype).realize()
  comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  nc = c.numpy()
  np.testing.assert_allclose(nc, comp, atol=ATOL, rtol=RTOL)
