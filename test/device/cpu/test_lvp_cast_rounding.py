import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import DEV
from tinygrad.renderer.nir import LVPRenderer, ncast

class TestNIRFloat16Rounding(unittest.TestCase):
  def test_float_narrowing_uses_nearest_even(self):
    for source in (dtypes.float, dtypes.double):
      with self.subTest(source=source), patch("tinygrad.renderer.nir.nalu") as alu:
        ncast(None, None, source, dtypes.half)
        self.assertEqual(alu.call_args.args[1], "f2f16_rtne")

  @unittest.skipUnless(DEV.target("CPU").renderer == "LVP", "requires DEV=CPU:LVP with tinymesa and LLVM 20")
  def test_float64_midpoint_rounding_on_lvp(self):
    values = np.array([sign * (1 + 2**-11 + delta) for sign in (1, -1) for delta in (-2**-40, 0, 2**-40)], dtype=np.float64)
    expected = np.array([0x3c00, 0x3c00, 0x3c01, 0xbc00, 0xbc00, 0xbc01], dtype=np.uint16)
    self.assertIsInstance(Device["CPU"].renderer, LVPRenderer)
    actual = Tensor(values, device="CPU").cast(dtypes.float16).bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
  unittest.main()
