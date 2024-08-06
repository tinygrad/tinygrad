from tinygrad import Tensor, dtypes
import numpy as np


"""dt = dtypes.float8
t = Tensor(np.ones(3)).cast(dt).realize().detach()
t2 = Tensor(np.ones(3)).cast(dt).realize().detach()


t3 = t @ t2

print(t3.cast(dtypes.float32).numpy())
"""

from tinygrad.helpers import getenv
dtype_in = dtypes.f8e5m2
acc_dtype = dtypes.float32
N = getenv("N", 1000)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  import time
  start_time = time.time()
  a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = Tensor.rand(M, K, dtype=dtype_in).realize(), Tensor.rand(K, N, dtype=dtype_in).realize()
    c = a.matmul(b, acc_dtype=acc_dtype).realize()
  comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  nc = c.numpy()
  try:
    np.testing.assert_allclose(nc, comp, atol=ATOL, rtol=RTOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES") > 0:
      indices = np.where(~np.isclose(nc, comp, rtol=RTOL, atol=ATOL))
      non_matching_elements_nc = nc[indices]
      non_matching_elements_comp = comp[indices]
      print(indices)
      print("result      :", non_matching_elements_nc)
      print("ground truth:", non_matching_elements_comp)
    raise e
