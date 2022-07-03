from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2D
from tinygrad import optim
from extra.utils import get_parameters  # TODO: move to optim
import unittest

class TestBatchnorm(unittest.TestCase):
  def test_conv_bn(self):
    Tensor.training = True

    x = Tensor.ones(1,12,128,256, requires_grad=False)
    class LilModel:
      def __init__(self):
        self.c = Conv2d(12, 32, 3, padding=1, bias=False)
        self.bn = BatchNorm2D(32, track_running_stats=False)
      def forward(self, x):
        return self.bn(self.c(x)).relu()

    lm = LilModel()
    loss = lm.forward(x).sum()
    optimizer = optim.SGD(get_parameters(lm), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #out = loss.detach().numpy()
    for p in optimizer.params:
      p.realize()
    Tensor.training = False

if __name__ == '__main__':
  unittest.main()