import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, get_single_element
from tinygrad.dtype import _to_np_dtype
from tinygrad.engine.realize import compile_linear
from tinygrad.codegen.opt import OptOps

dtype_in = (dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else
            dtypes.fp8e4m3 if getenv("FP8E4M3") else dtypes.fp8e5m2 if getenv("FP8E5M2") else dtypes.float)
acc_dtype = (dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else
            dtypes.fp8e4m3 if getenv("ACC_FP8E4M3") else dtypes.fp8e5m2 if getenv("ACC_FP8E5M2") else None)
if getenv("INT"):  dtype_in, acc_dtype = dtypes.int8, dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)

atol, rtol = {dtypes.half:{1e-3, 1e-2}, dtypes.bfloat16:(1e-3, 1e-2), dtypes.fp8e4m3:(1e-1, 1e-1), dtypes.fp8e5m2:(1.0, 5e-1)}.get(dtype_in, (1e-4, 3e-2))
ATOL, RTOL = getenv("ATOL", atol), getenv("RTOL", rtol)

INT_LOW = getenv("INT_LOW", 0)
INT_HIGH = getenv("INT_HIGH", 10)

if __name__ == "__main__":
  exact = getenv("EXACT", 0) != 0
  seed = getenv("SEED", 0)
  def init_matrix(rows, cols):
    rng = np.random.default_rng(seed + rows*1000003 + cols)
    # NOTE: numpy does not support bfloat16
    if (np_dtype := _to_np_dtype(dtype_in)) is None: np_dtype = np.float32
    if dtype_in in dtypes.ints:
      return Tensor(rng.integers(INT_LOW, INT_HIGH, (rows, cols), dtype=np_dtype)).realize()
    if exact:
      # With K<=1024 these products and their sums are exactly representable in
      # FP16, making this a strict semantic check rather than a loose error test.
      values = rng.integers(-1, 2, (rows, cols), dtype=np.int8).astype(np.float16) * np.float16(1/32)
      return Tensor(values).cast(dtype_in).realize()
    return Tensor(rng.random((rows, cols), dtype=np.float32).astype(np_dtype)-0.5).cast(dtype_in).realize()

  a, b = init_matrix(M, K), init_matrix(K, N)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = init_matrix(M, K), init_matrix(K, N)
    c = a.matmul(b, dtype=acc_dtype).realize()

  if getenv("SHOULD_USE_TC"):
    linear = compile_linear(a.matmul(b, dtype=acc_dtype).schedule_linear())
    call = get_single_element(list(linear.src))
    applied_opts = call.src[0].src[0].arg.applied_opts
    assert any(opt.op is OptOps.TC for opt in applied_opts), f"TC not triggered, {applied_opts}"

  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  res = c.numpy()
  if exact and dtype_in is dtypes.half and acc_dtype is dtypes.half:
    wrong = np.flatnonzero(res.reshape(-1).view(np.uint16) != ref.astype(np.float16).reshape(-1).view(np.uint16))
    err = np.abs(res.astype(np.float32)-ref)
    print(f"EXACT CHECK outputs={res.size} bad_count={wrong.size} max_abs={float(err.max()):.9g} mean_abs={float(err.mean()):.9g}")
    if wrong.size: raise AssertionError(f"exact FP16 GEMM mismatch at flat index {int(wrong[0])}")
    raise SystemExit(0)
  try:
    np.testing.assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
  except AssertionError as e:
    if getenv("DEBUG_VALUES", 0) > 0:
      mismatch = np.where(~np.isclose(res, ref, rtol=RTOL, atol=ATOL))
      print("Mismatch indices:", mismatch)
      print("Result          :", res[mismatch])
      print("Ground truth    :", ref[mismatch])
    raise e
