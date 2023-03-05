import unittest
from tinygrad.tensor import Tensor
# similar to test/external/external_test_gpu_ast.py, but universal

# 1x1 6 <- 24
class TestSpecificConv(unittest.TestCase):
  def test_1x1_6_24(self):
    x = Tensor.randn(1,24*4,32,64)
    w = Tensor.randn(6*4,24*4,1,1)
    x.conv2d(w).permute(0,2,3,1).reshape(32, 384, 4).contiguous().realize()

if __name__ == '__main__':
  unittest.main()