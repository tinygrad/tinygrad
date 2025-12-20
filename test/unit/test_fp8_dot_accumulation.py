#!/usr/bin/env python
import unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import float_to_fp8, fp8_to_float


def quantize_fp8_ref(x: np.ndarray, fp8_dtype) -> np.ndarray:
  x = x.astype(np.float32, copy=False)
  out = np.empty_like(x, dtype=np.float32)
  it = np.nditer([x, out], flags=["multi_index"], op_flags=[["readonly"], ["writeonly"]])
  for vin, vout in it:
    vout[...] = fp8_to_float(float_to_fp8(float(vin), fp8_dtype), fp8_dtype)
  return out


class TestFP8DotAccumulation(unittest.TestCase):
  def _check(self, fp8_dtype):
    Device.DEFAULT = "PYTHON"
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 64), dtype=np.float32)
    b = rng.standard_normal((64, 32), dtype=np.float32)

    aq = quantize_fp8_ref(a, fp8_dtype)
    bq = quantize_fp8_ref(b, fp8_dtype)
    ref = aq @ bq

    out = Tensor(a, device=Device.DEFAULT).cast(fp8_dtype).matmul(Tensor(b, device=Device.DEFAULT).cast(fp8_dtype), dtype=dtypes.float).realize().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  def test_fp8e4m3(self):
    self._check(dtypes.fp8e4m3)

  def test_fp8e5m2(self):
    self._check(dtypes.fp8e5m2)


if __name__ == "__main__":
  unittest.main()
