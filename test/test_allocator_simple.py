import unittest
from tinygrad import Tensor
from tinygrad.ops import Device, Compiled

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only test for compiled backends")
class TestAllocatorSimple(unittest.TestCase):
  def test_reuses(self):
    a = Tensor.empty(128, 128).realize()
    del a
    b = Tensor.empty(128, 128).realize()
    del b

if __name__ == '__main__':
  unittest.main()
