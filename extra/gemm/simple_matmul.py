import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, get_single_element
from tinygrad.dtype import _to_np_dtype
from tinygrad.codegen.opt import OptOps

dtype_in = (dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else
            dtypes.fp8e4m3 if getenv("FP8E4M3") else dtypes.fp8e5m2 if getenv("FP8E5M2") else dtypes.float)
acc_dtype = (dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else
            dtypes.fp8e4m3 if getenv("ACC_FP8E4M3") else dtypes.fp8e5m2 if getenv("ACC_FP8E5M2") else None)
if getenv("INT"):  dtype_in, acc_dtype = dtypes.int8, dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

dtype_in_a = dtype_in
dtype_in_b = dtype_in
fp8_hybrid = getenv("FP8_HYBRID", 0)

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)

atol, rtol = {dtypes.half:{1e-3, 1e-2}, dtypes.bfloat16:(1e-3, 1e-2), dtypes.fp8e4m3:(1e-1, 1e-1),
              dtypes.fp8e5m2:(1.0, 5e-1)}.get(dtypes.fp8e5m2 if fp8_hybrid else dtype_in, (1e-4, 3e-2))
if fp8_hybrid == 1: dtype_in_a, dtype_in_b = dtypes.fp8e4m3, dtypes.fp8e5m2
elif fp8_hybrid == 2: dtype_in_a, dtype_in_b = dtypes.fp8e5m2, dtypes.fp8e4m3

ATOL, RTOL = getenv("ATOL", atol), getenv("RTOL", rtol)

INT_LOW = getenv("INT_LOW", 0)
INT_HIGH = getenv("INT_HIGH", 10)

if __name__ == "__main__":
  def init_matrix(rows, cols, dtype=None):
    if dtype is None: dtype = dtype_in
    rng = np.random.default_rng()
    # NOTE: numpy does not support bfloat16
    if (np_dtype := _to_np_dtype(dtype)) is None: np_dtype = np.float32
    if dtype in dtypes.ints:
      return Tensor(rng.integers(INT_LOW, INT_HIGH, (rows, cols), dtype=np_dtype)).realize()
    return Tensor(rng.random((rows, cols), dtype=np.float32).astype(np_dtype)-0.5).cast(dtype).realize()

  if fp8_hybrid:
    from extra.fp8.fp8_linear import _scaled_mm
    def do_matmul(a, b): return _scaled_mm(a, b.T)
  else:
    def do_matmul(a, b): return a.matmul(b, dtype=acc_dtype)

  a, b = init_matrix(M, K, dtype_in_a), init_matrix(K, N, dtype_in_b)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = init_matrix(M, K, dtype_in_a), init_matrix(K, N, dtype_in_b)
    c = do_matmul(a, b).realize()

  if getenv("SHOULD_USE_TC"):
    sched = do_matmul(a, b).schedule()
    ei = get_single_element(sched)
    ei.lower()
    assert any(opt.op is OptOps.TC for opt in ei.prg.p.applied_opts), f"TC not triggered, {ei.prg.p.applied_opts}"

  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  res = c.numpy()
  try:
    np.testing.assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES", 0) > 0:
      mismatch = np.where(~np.isclose(res, ref, rtol=RTOL, atol=ATOL))
      print("Mismatch indices:", mismatch)
      print("Result          :", res[mismatch])
      print("Ground truth    :", ref[mismatch])
    raise e
