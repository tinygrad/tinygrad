import torch
import numpy as np
import unittest
from tinygrad.tensor import Tensor

def test_op(shps, f1, f2, atol=1e-7, grad_atol=1e-7):
  ts = [torch.rand(x, requires_grad=True) for x in shps]
  tst = [Tensor(x.detach().numpy()) for x in ts]

  out = f1(*ts)
  ret = f2(*tst)
  # TODO: why so inaccurate?
  np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=atol)

  out.mean().backward()
  ret.mean().backward()

  for t, tt in zip(ts, tst):
    np.testing.assert_allclose(t.grad, tt.grad, atol=grad_atol)

class TestOps(unittest.TestCase):
  def test_conv2d(self):
    for cin in [1,2,3]:
      for H in [2,3,5]:
        for W in [2,3,5]:
          test_op([(5,cin,10,7), (4,cin,H,W)], torch.nn.functional.conv2d, Tensor.conv2d, atol=1e-5)

  def test_maxpool2x2(self):
    test_op([(5,2,11,8)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), Tensor.max_pool2d)

  def test_avgpool2x2(self):
    test_op([(5,2,11,8)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), Tensor.avg_pool2d)

if __name__ == '__main__':
  unittest.main()
