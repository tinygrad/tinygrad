import unittest
from tinygrad import Tensor

def unwrap_buf(t) -> list:
  assert t.lazydata.base.realized is not None, f"expected {t.lazydata} to be real"
  return t.tolist()

class TestMap(unittest.TestCase):
  def test_simple_copy(self):
    a = Tensor([11])
    a.realize()
    self.assertListEqual(unwrap_buf(a), [11])

  def test_multi_copy(self):
    a = Tensor([11])
    b = Tensor([1111])
    Tensor.realize(a, b)
    self.assertListEqual(unwrap_buf(a), [11])
    self.assertListEqual(unwrap_buf(b), [1111])

  def test_add(self):
    a = Tensor([11])
    b = Tensor([2])
    c = a+b
    c.realize()
    self.assertListEqual(unwrap_buf(c), [13])

  def test_simple_const(self):
    a = Tensor([11])
    b = a+2
    b.realize()
    self.assertListEqual(unwrap_buf(b), [13])

  def test_simple_const_folding(self):
    a = Tensor([11])
    b = a*1
    b.realize()
    self.assertListEqual(unwrap_buf(b), [11])

  def test_const_folding_alt(self):
    a = Tensor([11])
    b = (a*0).contiguous()
    b.realize()
    self.assertListEqual(unwrap_buf(b), [0])

  def test_const_folding_folds(self):
    a = Tensor([11])
    b = (a*0)
    b.realize()
    self.assertIsNone(b.lazydata.realized)

if __name__ == "__main__":
  unittest.main()
