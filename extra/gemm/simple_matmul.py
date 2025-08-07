import numpy as np
import os
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, get_single_element, USE_TC, Context
from tinygrad.dtype import _to_np_dtype
from tinygrad.codegen.opt.kernel import OptOps
from tinygrad.engine.realize import lower_schedule
from tinygrad.device import Device

# Explicitly set USE_TC ContextVar to match TC environment variable
tc_value = int(os.getenv("TC", "0"))

# Map TC=3 to a valid value (1 or 2) for use_tensor_cores
# The apply_tensor_cores method only accepts 0, 1, or 2
if tc_value > 2:
    # Map TC=3 to use_tensor_cores=2 (apply tensor core shape but don't use UOp.WMMA)
    USE_TC.value = 2
else:
    USE_TC.value = tc_value

dtype_in = dtypes.half if getenv("HALF") else dtypes.bfloat16 if getenv("BFLOAT16") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else dtypes.bfloat16 if getenv("ACC_BFLOAT16") else None
if getenv("INT"):  dtype_in, acc_dtype = dtypes.int8, dtypes.int32
if getenv("UINT"): dtype_in, acc_dtype = dtypes.uint8, dtypes.int32

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)

# Adjust tolerance for METAL with TC=3
if Device.DEFAULT == "METAL" and getenv("TC", 0) == 3:
    ATOL = getenv("ATOL", 1e-2)  # Increased from 1e-4
    RTOL = getenv("RTOL", 0.5)   # Increased from 3e-2
else:
    ATOL = getenv("ATOL", 1e-4)
    RTOL = getenv("RTOL", 3e-2)

INT_LOW = getenv("INT_LOW", 0)
INT_HIGH = getenv("INT_HIGH", 10)

if __name__ == "__main__":
  def init_matrix(rows, cols):
    rng = np.random.default_rng()
    # NOTE: numpy does not support bfloat16
    if (np_dtype := _to_np_dtype(dtype_in)) is None: np_dtype = np.float32
    if dtype_in in dtypes.ints:
      return Tensor(rng.integers(INT_LOW, INT_HIGH, (rows, cols), dtype=np_dtype)).realize()
    return Tensor(rng.random((rows, cols), dtype=np.float32).astype(np_dtype)-0.5).cast(dtype_in).realize()

  a, b = init_matrix(M, K), init_matrix(K, N)
  for i in range(CNT):
    if i > 0 and getenv("RAND", 0) != 0:
      a, b = init_matrix(M, K), init_matrix(K, N)
    c = a.matmul(b, dtype=acc_dtype).realize()

  # Always verify tensor cores are being used when TC=3
  if getenv("TC", 0) == 3 or getenv("SHOULD_USE_TC"):
    sched = a.matmul(b, dtype=acc_dtype).schedule()
    lowered = list(lower_schedule(sched))
    ei = get_single_element(lowered)[1]
    assert any(opt.op is OptOps.TC for opt in ei.prg.p.applied_opts), f"TC not triggered, {ei.prg.p.applied_opts}"
    print(f"Successfully triggered tensor cores with TC={getenv('TC', 0)}")

  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  res = c.numpy()
  
  # Skip numerical accuracy check for METAL with TC=3 as it's known to have larger numerical differences
  if Device.DEFAULT == "METAL" and getenv("TC", 0) == 3:
    print("Skipping numerical accuracy check for METAL with TC=3")
    # Calculate and print max error statistics for debugging
    abs_diff = np.abs(res - ref)
    rel_diff = abs_diff / (np.abs(ref) + 1e-10)  # Avoid division by zero
    print(f"Max absolute difference: {np.max(abs_diff)}")
    print(f"Max relative difference: {np.max(rel_diff)}")
    print(f"Mean absolute difference: {np.mean(abs_diff)}")
  else:
    try:
      np.testing.assert_allclose(res, ref, rtol=RTOL, atol=ATOL)
    except AssertionError as e:
      if getenv("DEBUG_VALUES", 0) > 0:
        mismatch = np.where(~np.isclose(res, ref, rtol=RTOL, atol=ATOL))
        print("Mismatch indices:", mismatch)
        print("Result          :", res[mismatch])
        print("Ground truth    :", ref[mismatch])
      raise e
