import unittest
from tinygrad import Tensor

class TestLinearizer(unittest.TestCase):
  def test_late_bias_load(self):
    img = Tensor.empty(1, 3, 16, 16)
    w = Tensor.empty(16, 3, 3, 3)
    b = Tensor.empty(16)
    img.conv2d(w, b).realize()

if __name__ == '__main__':
  unittest.main()
