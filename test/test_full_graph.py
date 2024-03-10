import unittest
from tinygrad import Tensor

class TestFullGraph(unittest.TestCase):
  def test_double_matmul(self):
    N = 32
    x = Tensor.empty(N, N)
    w1 = Tensor.empty(N, N)
    w2 = Tensor.empty(N, N)
    out = (x@w1)@w2
    out.realize()

if __name__ == '__main__':
  unittest.main(verbosity=2)