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

if __name__ == '__main__':
  unittest.main()
