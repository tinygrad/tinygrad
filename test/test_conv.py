import unittest
from tinygrad.tensor import Tensor

class TestConv(unittest.TestCase):
  def test_simple(self):
    x = Tensor.ones(1,12,128,256)
    w = Tensor.ones(32,12,3,3)
    ret = x.conv2d(w, padding=(1,1))
    # it's not 108 around the padding
    assert (ret.numpy()[:, :, 1:-1, 1:-1] == 108).all()

if __name__ == '__main__':
  unittest.main()