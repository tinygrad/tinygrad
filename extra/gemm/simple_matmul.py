import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor

dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
if getenv("INT"):  dtype_in, acc_dtype = dtypes.int8, dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

dtype_in = (
    dtypes.half if getenv("HALF") else
    dtypes.bfloat16 if getenv("BFLOAT16") else
    dtypes.fp8e4m3 if getenv("FP8E4M3") else
    dtypes.fp8e5m2 if getenv("FP8E5M2") else
    dtypes.float
)

acc_dtype = (
    dtypes.half if getenv("ACC_HALF") else
    dtypes.bfloat16 if getenv("ACC_BFLOAT16") else
    dtypes.fp8e4m3 if getenv("ACC_FP8E4M3") else
    dtypes.fp8e5m2 if getenv("ACC_FP8E5M2") else
    None
)

if getenv("INT"): dtype_in = dtypes.int8acc_dtype = dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
ATOL = getenv("ATOL", 1e-4)
RTOL = getenv("RTOL", 3e-2)

if __name__ == "__main__":
  def init_matrix(rows, cols):
    if dtype_in in dtypes.ints:
      return Tensor.randint((rows, cols), dtype=dtype_in).realize()
    return Tensor.rand(rows, cols, dtype=dtype_in).realize()

  a, b = init_matrix(M, K), init_matrix(K, N)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = init_matrix(M, K), init_matrix(K, N)
    c = a.matmul(b, acc_dtype=acc_dtype).realize()

  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  res = c.numpy()
  if getenv("DEBUG_VALUES", 0) > 1:
    print("\nA:")
    for row in a.numpy():
      print(" ".join(f"{value:.5f}" for value in row))
    print("\nB:")
    for row in b.numpy():
      print(" ".join(f"{value:.5f}" for value in row))
    print("-" * 80)
    print("\nResult:")
    for row in res:
      print(" ".join(f"{value:.5f}" for value in row))
    print("\nGround truth:")
    for row in ref:
      print(" ".join(f"{value:.5f}" for value in row))
  try:
    np.testing.assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES", 0) > 0:
      mismatch = np.where(~np.isclose(res, ref, rtol=RTOL, atol=ATOL))
      print("Mismatch indices:", mismatch)
      print("Result          :", res[mismatch])
      print("Ground truth    :", ref[mismatch])
    raise e
