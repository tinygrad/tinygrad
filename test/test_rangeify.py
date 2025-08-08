import unittest
from tinygrad import Tensor

class TestRangeify(unittest.TestCase):
  def test_double_gemm(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (A@B@C).realize()

  def test_double_gemm_half_contig(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous(arg=(1,))@C).realize()

  def test_double_gemm_contig(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous()@C).realize()

  def test_many_gemm(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    D = Tensor.empty(N, N)
    E = Tensor.empty(N, N)
    F = Tensor.empty(N, N)
    (A@B@C@D@E@F).realize()

  def test_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    x.conv2d(w1).realize()

  def test_double_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).conv2d(w2).realize()

  def test_double_conv2d_half_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    # NOTE: this contiguous doesn't help
    x.conv2d(w1).contiguous(arg=(1,)).conv2d(w2).permute(0,2,3,1).contiguous().realize()

  def test_double_conv2d_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).contiguous().conv2d(w2).realize()

if __name__ == '__main__':
  unittest.main()
