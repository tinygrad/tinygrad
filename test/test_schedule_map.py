import unittest
from tinygrad import Tensor

class TestMap(unittest.TestCase):
  def test_simple_copy(self):
    a = Tensor([11])
    a.realize()
    self.assertEqual(a.item(), 11)

  def test_multi_copy(self):
    a = Tensor([11])
    b = Tensor([1111])
    Tensor.realize(a, b)
    self.assertEqual(a.item(), 11)
    self.assertEqual(b.item(), 1111)

if __name__ == "__main__":
  unittest.main()
