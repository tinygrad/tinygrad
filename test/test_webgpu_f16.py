import unittest
from tinygrad import Tensor, Device
import numpy as np
import sys

class TestWebGPUf16(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT != "WEBGPU", "device isn't WEBGPU")
  @unittest.skipUnless(sys.platform == "win32", "need windows for f16 fix")
  def test_use_dxc_toggle_enables_f16(self):
    ref = np.arange(10).astype(np.float16).sum()
    result = Tensor.arange(10).half().sum().numpy()
    np.testing.assert_equal(ref, result)
