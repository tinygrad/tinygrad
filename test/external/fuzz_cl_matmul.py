import numpy as np
from hypothesis import given, settings, strategies as st
from tinygrad import Tensor, dtypes

BOUNDARY = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256]

@st.composite
def matmul_cases(draw):
  m = draw(st.sampled_from([1, 2, 7, 8, 16, 32, 64, 128]))
  n = draw(st.sampled_from([1, 2, 7, 8, 16, 32, 64, 128]))
  k = draw(st.sampled_from(BOUNDARY))
  post = draw(st.sampled_from(["none", "relu", "bias", "sum"]))
  return m, n, k, post

def run_case(m, n, k, post):
  rng = np.random.default_rng((m * 1000003 + n * 1009 + k) & 0xFFFFFFFF)
  a_np = rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
  b_np = rng.standard_normal((k, n), dtype=np.float32).astype(np.float16)

  ref = a_np.astype(np.float32) @ b_np.astype(np.float32)

  a = Tensor(a_np, device="CL", dtype=dtypes.float16)
  b = Tensor(b_np, device="CL", dtype=dtypes.float16)
  got_t = a @ b

  if post == "relu":
    ref = np.maximum(ref, 0)
    got_t = got_t.relu()
  elif post == "bias":
    bias = rng.standard_normal((n,), dtype=np.float32).astype(np.float16)
    ref = ref + bias.astype(np.float32)
    got_t = got_t + Tensor(bias, device="CL", dtype=dtypes.float16)
  elif post == "sum":
    ref = ref.sum(axis=1)
    got_t = got_t.sum(axis=1)

  got = got_t.numpy()
  np.testing.assert_allclose(got, ref, rtol=3e-2, atol=3e-2)

@settings(max_examples=50, deadline=None)
@given(matmul_cases())
def test_cl_matmul(case):
  m, n, k, post = case
  try:
    run_case(m, n, k, post)
  except Exception:
    print(f"\nREPRO: m={m} n={n} k={k} post={post}")
    raise
