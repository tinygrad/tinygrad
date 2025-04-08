import unittest
from tinygrad import Tensor, GlobalCounters, Context, Device
from tinygrad.helpers import DEBUG
from tinygrad.codegen.kernel import Kernel

class TestSoftmaxFusion(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with Context(TRACK_MATCH_STATS=0): cls.test = Tensor.ones(32, 10).contiguous().realize()

  def setUp(self):
    GlobalCounters.reset()

  def test_norm(self):
    print("*** norm ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      # NOTE: there's an implied expand on the mean here
      out = self.test / self.test.mean(-1, keepdim=True)
      out.realize()

  def test_single_kernel_norm(self):
    print("*** single kernel norm ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      inp = self.test.reshape(32, 10, 1)
      div = self.test.reshape(32, 1, 10).expand(32, 10, 10).mean(axis=-1, keepdim=True)
      out = inp / div
      out.realize()

  def test_softmax(self):
    # this is the softmax from scaled_dot_product_attention
    # it becomes 3 kernels
    print("*** softmax ***")
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2)):
      out = self.test.softmax(-1)
      out.realize()

  def test_single_kernel_softmax(self):
    print("*** single kernel softmax ***")
    # NOTE: DONT_GROUP_REDUCES is required here
    with Context(NOOPT=1, DEBUG=max(DEBUG.value, 2), DONT_GROUP_REDUCES=1):
      inp = self.test.reshape(32, 1, 1, 10).expand(32, 10, 1, 10)
      imx = self.test.reshape(32, 1, 10, 1).expand(32, 10, 10, 10).max(axis=-2, keepdim=True)
      m = inp - imx.detach()
      e = m.exp()
      ss = e.sum(axis=-1, keepdim=True)

      inp = self.test.reshape(32, 10, 1, 1)
      imx = self.test.reshape(32, 1, 10, 1).expand(32, 10, 10, 1).max(axis=-2, keepdim=True)
      m = inp - imx.detach()
      e = m.exp()

      out = e.div(ss)
      out.realize()

if __name__ == '__main__':
  unittest.main()
