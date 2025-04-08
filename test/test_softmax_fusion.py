import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.dtype import DTypeLike
from tinygrad.helpers import DEBUG

def single_kernel_softmax(x_in:Tensor, axis=-1, dtype:DTypeLike|None=None) -> Tensor:
  # only support axis =-1
  x = x_in.reshape(-1, x_in.shape[-1])
  nr_dim, r_dim = x.shape

  inp = x.reshape(nr_dim, 1, 1, r_dim).expand(nr_dim, r_dim, 1, r_dim)
  imx = x.reshape(nr_dim, 1, r_dim, 1).expand(nr_dim, r_dim, r_dim, r_dim).max(axis=-2, keepdim=True)
  m = inp - imx.detach()
  if dtype is not None: m = m.cast(dtype)
  e = m.exp()
  ss = e.sum(axis=-1, keepdim=True)

  inp = x.reshape(nr_dim, r_dim, 1, 1)
  imx = x.reshape(nr_dim, 1, r_dim, 1).expand(nr_dim, r_dim, r_dim, 1).max(axis=-2, keepdim=True)
  m = inp - imx.detach()
  if dtype is not None: m = m.cast(dtype)
  e = m.exp()

  out = e.div(ss).reshape(x_in.shape)
  return out

class TestSoftmaxFusion(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with Context(TRACK_MATCH_STATS=0): cls.test = Tensor.rand(32, 10).contiguous().realize()

  def setUp(self):
    GlobalCounters.reset()

  def test_norm(self):
    print("*** norm ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      # NOTE: there's an implied expand on the mean here
      sout = self.test / self.test.mean(-1, keepdim=True)
      sout.realize()

    print("*** single kernel norm ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      inp = self.test.reshape(32, 10, 1)
      div = self.test.reshape(32, 1, 10).expand(32, 10, 10).mean(axis=-1, keepdim=True)
      out = (inp / div).reshape(32, 10)
      out.realize()

    np.testing.assert_allclose(sout.numpy(), out.numpy())

  def test_softmax(self):
    # this is the softmax from scaled_dot_product_attention
    # it becomes 3 kernels
    print("*** softmax ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      sout = self.test.softmax(-1)
      sout.realize()

    print("*** single kernel softmax ***")
    # NOTE: DONT_GROUP_REDUCES is required here
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2), DONT_GROUP_REDUCES=1):
      out = single_kernel_softmax(self.test)
      out.realize()

    np.testing.assert_allclose(sout.numpy(), out.numpy())

if __name__ == '__main__':
  unittest.main()
