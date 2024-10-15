import unittest
from tinygrad import Tensor

class TestTiny(unittest.TestCase):
  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

if __name__ == '__main__':
  unittest.main()

