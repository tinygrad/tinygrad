import unittest
from tinygrad import Tensor

class TestSchedule(unittest.TestCase):
  def test_simple_copy(self):
    a = Tensor([1])
    a.realize()
    self.assertIsNotNone(a.lazydata.base.realized)
    self.assertListEqual(a.tolist(), [1])

  def test_const_folding(self):
    x = Tensor.ones(4, 4)
    x.realize()
    self.assertIsNone(x.lazydata.base.realized)

  def test_const_folding_contiguous(self):
    x = Tensor.ones(4, 4).contiguous().realize()
    self.assertIsNotNone(x.lazydata.base.realized)

  def test_ops_folding(self):
    x = Tensor.ones(4, 4).contiguous()
    y = x+0
    self.assertIsNot(x.lazydata, y.lazydata)
    y.realize()
    self.assertIs(x.lazydata, y.lazydata)

if __name__ == "__main__":
  unittest.main()
