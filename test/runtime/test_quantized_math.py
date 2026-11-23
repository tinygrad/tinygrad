import unittest, numpy as np
from tinygrad import Tensor, Device, dtypes

int_math = not Device[Device.DEFAULT].renderer.vector_floats

class TestQuantizedMath(unittest.TestCase):
  def test_int_sum(self):
    x, w = np.random.randint(0, 256, (8, 4096)), np.random.randint(0, 256, 4096)
    out = ((Tensor(x.astype(np.uint8)).int()-3).float() * (Tensor(w.astype(np.uint8)).int()-7).float()).sum(1).numpy()
    # int32 accumulation is exact past float32's 24-bit mantissa
    if int_math: np.testing.assert_equal(out, ((x-3)*(w-7)).sum(1).astype(np.float32))
    else: np.testing.assert_allclose(out, ((x-3)*(w-7)).sum(1), rtol=1e-4)

  def test_requantize(self):
    x = np.arange(-30000, 30000, dtype=np.int16)
    out = ((Tensor(x).int().float()*0.00731 + 0.4999 + 128).int().maximum(0)).numpy()
    ref = np.maximum((x.astype(np.float32)*np.float32(0.00731) + np.float32(0.4999) + np.float32(128)).astype(np.int32), 0)
    if int_math:
      self.assertLessEqual(np.abs(out - ref).max(), 1)
      self.assertLess((out != ref).mean(), 2**-10)
    else: np.testing.assert_equal(out, ref)

  def test_requantize_unbounded(self):
    x = np.arange(-30000, 30000, dtype=np.int32)
    out = ((Tensor(x).float()*0.00731 + 0.4999 + 128).int().maximum(0)).numpy()
    np.testing.assert_equal(out, np.maximum((x.astype(np.float32)*np.float32(0.00731) + np.float32(0.4999) + np.float32(128)).astype(np.int32), 0))

  def test_requantize_edge_cases(self):
    x = np.arange(-8, 8, dtype=np.float32)
    out = ((Tensor(x).int().float()*0.00731 + 1.0).int().maximum(0)).numpy()
    np.testing.assert_equal(out, np.maximum((x.astype(np.int32).astype(np.float32)*np.float32(0.00731) + np.float32(1.0)).astype(np.int32), 0))
    _ = ((Tensor(np.arange(8, dtype=np.int8)).float()*0.00731 + float("nan")).int().maximum(0)).realize()
    u_in = np.arange(0, 200, dtype=np.int16)
    u = ((Tensor(u_in).int().float()*0.00731 + 10.0).cast(dtypes.uint8).maximum(0)).numpy()
    np.testing.assert_equal(u, (u_in.astype(np.float32)*np.float32(0.00731) + np.float32(10.0)).astype(np.uint8))
    s2u = np.arange(-128, 128, dtype=np.int8)
    out_s2u = ((Tensor(s2u).clip(-1, 1).cast(dtypes.uint8).float()*0.5 + 0.5).int().maximum(0)).numpy()
    ref_s2u = np.maximum((np.clip(s2u, -1, 1).astype(np.uint8).astype(np.float32)*np.float32(0.5) + np.float32(0.5)).astype(np.int32), 0)
    np.testing.assert_allclose(out_s2u, ref_s2u, atol=1)

if __name__ == '__main__':
  unittest.main()
