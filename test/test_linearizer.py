import numpy as np
import unittest

from tinygrad.lazy import Device
from tinygrad.ops import GlobalCounters, Compiled
from tinygrad.tensor import Tensor

class TestLinearizer(unittest.TestCase):
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT], Compiled), "Only Compiled supports cache")
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()

if __name__ == '__main__':
  unittest.main()
