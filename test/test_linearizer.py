import unittest

from tinygrad.lazy import Device
from tinygrad.ops import GlobalCounters, Compiled
from tinygrad.tensor import Tensor

class TestLinearizer(unittest.TestCase):
  @unittest.skipUnless(isinstance(Device[Device.DEFAULT], Compiled), "Only Compiled supports cache")
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    GlobalCounters.cache = []
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()
    rawbufs = GlobalCounters.cache[0][1]
    GlobalCounters.cache = None
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.realized, b.lazydata.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    assert (np_c == c.numpy()).all()

if __name__ == '__main__':
  unittest.main()
