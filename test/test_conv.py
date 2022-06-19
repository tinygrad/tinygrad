import unittest
import numpy as np
from tinygrad.tensor import Tensor

class TestConv(unittest.TestCase):
  def test_simple(self):
    x = Tensor.ones(1,12,128,256)
    w = Tensor.ones(32,12,3,3)
    ret = x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()
    # it's not 108 around the padding
    assert (ret[:, :, 1:-1, 1:-1] == 108).all()
    assert ret[0,0,0,0] == 48
    assert ret[0,0,0,1] == 72

  def test_many_simple(self):
    x = Tensor(np.arange(8*2*8).reshape(1,8,2,8).astype(np.float32))
    #w = Tensor(np.arange(8*8*1*1).reshape(8,8,1,1).astype(np.float32))
    w = Tensor.eye(8).reshape((8,8,1,1))
    ret = x.conv2d(w, stride=(1,2), padding=(0,0)).numpy()
    print(ret)

  def test_simple_biased(self):
    C = 8
    x = Tensor.zeros(1,C,5,5)
    w = Tensor.eye(C).reshape((C,C,1,1))
    b = Tensor(np.arange(C))
    ret = Tensor.conv2d(x,w,b).relu().conv2d(w,b)

    print(ret.numpy())

  def test_first_three(self):
    x = Tensor.ones(1,12,128,256)

    w = Tensor.ones(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1))

    w = Tensor.ones(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32)

    w = Tensor.ones(16,32,1,1)
    x = x.conv2d(w)

    x = x.numpy()
    print(x.shape)

  def test_elu(self):
    x = Tensor.ones(1,12,128,256)

    w = Tensor.ones(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1))

    x = x.elu()

    w = Tensor.ones(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32)
    out = x.numpy()

  def test_bias(self):
    from tinygrad.nn import Conv2d
    x = Tensor.ones(1,12,128,256)
    c = Conv2d(12, 32, 3)
    x = c(x)
    x = x.relu()
    w = Tensor.uniform(32, 1, 3, 3)
    x = x.conv2d(w, groups=32)
    out = x.numpy()
  
  def test_multiadd(self):
    w = Tensor.ones(32)
    x = Tensor.ones(32).relu()
    (w+x).numpy()

  def test_reorder(self):
    x = Tensor.ones(1,12,128,256)
    w = Tensor.ones(12,12,3,3)
    x = x.conv2d(w, padding=(1,1))
    print(x.shape)
    x = x.reshape((1, 12, 256, 128))
    x += 1
    x += 1
    x = x.reshape((1, 12, 128, 256))
    x.numpy()

if __name__ == '__main__':
  unittest.main()