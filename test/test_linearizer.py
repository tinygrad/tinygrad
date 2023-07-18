import numpy as np
import unittest

from tinygrad.lazy import Device
from tinygrad.ops import GlobalCounters, Compiled
from tinygrad.tensor import Tensor

class TestLinearizer(unittest.TestCase):
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT], Compiled), "Only Compiled supports cache")
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)

if __name__ == '__main__':
  unittest.main()
