#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_mla_bmm.py: amd.mla_bmm (mla absorb matmuls, one token) against the tensor matmul
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd import mla_bmm

rng = np.random.default_rng(0)
for H, I, O, T in ((24, 128, 512, 1), (24, 512, 128, 1), (24, 128, 512, 3)):  # q_nope @ k_b^T, (attn @ v) @ v_b^T, a few tokens
  x = Tensor(rng.standard_normal((1, H, T, I)).astype(np.float32))
  w = Tensor(rng.standard_normal((H, O, I)).astype(np.float16), dtype=dtypes.half)
  got, want = mla_bmm(x, w).numpy(), (x @ w.float().transpose(-1, -2)).numpy()
  np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
  print(f"H {H} I {I} O {O} T {T}: ok, max err {np.abs(got - want).max():.2e}")
