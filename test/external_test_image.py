import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.llops.ops_gpu import CLImage

# DEBUG=2 GPU=1 IMAGE=1 GRAPH=1 python3 test/external_test_image.py

def get_two_images(N):
  r1 = np.random.randn(N,N).reshape(N, N//4, 4)
  r2 = np.random.randn(N,N).reshape(N, N//4, 4)
  i1 = (Tensor(r1)+0).contiguous().realize()
  i2 = (Tensor(r2)+0).contiguous().realize()
  assert isinstance(i1.lazydata.realized._buf, CLImage)
  assert isinstance(i2.lazydata.realized._buf, CLImage)
  return (r1,r2), (i1,i2)

class TestImage(unittest.TestCase):
  def test_image_nothing(self):
    root = np.random.randn(1024,1024)
    ibuf = (Tensor(root).reshape(1024, 256, 4)+0).contiguous().realize()
    assert isinstance(ibuf.lazydata.realized._buf, CLImage)
    back = ibuf.reshape(1024, 1024).contiguous().numpy()
    np.testing.assert_allclose(root, back)

  def test_image_permute(self):
    root = np.random.randn(1024,1024)
    ibuf = Tensor(root).permute(1,0).reshape(1024, 256, 4).contiguous().realize()
    assert isinstance(ibuf.lazydata.realized._buf, CLImage)
    bbuf = ibuf.reshape(1024,1024).permute(1,0).contiguous().realize()
    np.testing.assert_allclose(root, bbuf.numpy())

  def test_image_mul(self):
    N = 16
    (r1,r2), (i1,i2) = get_two_images(N)
    ret = (i1*i2).numpy()
    np.testing.assert_allclose((r1*r2), ret, atol=1e-6)

  def test_reduce(self):
    N = 16
    (r1,r2), (i1,i2) = get_two_images(N)
    p1 = (i1*i2).sum(2).numpy()
    p2 = (r1*r2).sum(2)
    np.testing.assert_allclose(p1, p2, atol=1e-6)

  def test_full_reduce(self):
    N = 16
    (r1,r2), (i1,i2) = get_two_images(N)
    p1 = (i1*i2).sum().numpy()
    p2 = (r1*r2).sum()
    np.testing.assert_allclose(p1, p2, atol=1e-5)

if __name__ == '__main__':
  unittest.main()
