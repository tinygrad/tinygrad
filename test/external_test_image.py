import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CLImage

# DEBUG=2 GPU=1 IMAGE=1 GRAPH=1 python3 test/external_test_image.py

class TestImage(unittest.TestCase):
  def test_image_permute(self):
    root = np.random.randn(1024,1024)
    gbuf = Tensor(root)
    ibuf = gbuf.permute(1,0).reshape(1024, 256, 4).contiguous().realize()
    assert isinstance(ibuf.lazydata.realized._buf, CLImage)
    bbuf = ibuf.reshape(1024,1024).permute(1,0).contiguous().realize()
    np.testing.assert_allclose(root, bbuf.numpy())

if __name__ == '__main__':
  unittest.main()
