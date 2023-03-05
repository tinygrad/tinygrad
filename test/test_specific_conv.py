import unittest
from tinygrad.tensor import Tensor
# similar to test/external/external_test_gpu_ast.py, but universal

class TestSpecific(unittest.TestCase):
  # 1x1 6 <- 24
  def test_1x1_6_24(self):
    x = Tensor.randn(1,   24*4, 32, 64)
    w = Tensor.randn(6*4, 24*4, 1,  1)
    x.conv2d(w).permute(0,2,3,1).reshape(32, 384, 4).contiguous().realize()

  def test_vec_mul(self):
    x = Tensor.randn(1, 2048)
    w = Tensor.randn(2048, 512)
    (x @ w).reshape(1, 128, 4).contiguous().realize()

if __name__ == '__main__':
  unittest.main()