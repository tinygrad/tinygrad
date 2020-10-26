import torch
import numpy as np
import unittest
import timeit
import functools
from tinygrad.tensor import Tensor

def helper_test_op(shps, torch_fxn, tinygrad_fxn, atol=1e-7, grad_atol=1e-7):
  ts = [torch.rand(x, requires_grad=True) for x in shps]
  tst = [Tensor(x.detach().numpy()) for x in ts]

  out = torch_fxn(*ts)
  ret = tinygrad_fxn(*tst)

  # TODO: why so inaccurate?
  np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=atol)

  out.mean().backward()
  ret.mean().backward()

  for t, tt in zip(ts, tst):
    np.testing.assert_allclose(t.grad, tt.grad, atol=grad_atol)

  # speed
  torch_fp = timeit.Timer(functools.partial(torch_fxn, *ts)).timeit(5) * 1000/5
  tinygrad_fp = timeit.Timer(functools.partial(tinygrad_fxn, *tst)).timeit(5) * 1000/5

  torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_fxn, ts)).timeit(5) * 1000/5
  tinygrad_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), tinygrad_fxn, tst)).timeit(5) * 1000/5

  print("testing %30r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms" % (shps, torch_fp, tinygrad_fp, torch_fbp-torch_fp, tinygrad_fbp-tinygrad_fp))

class TestOps(unittest.TestCase):
  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for H in [2,5]:
          for W in [2,3,5]:
            helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
              lambda x,w: torch.nn.functional.conv2d(x,w).relu(),
              lambda x,w: Tensor.conv2d(x,w).relu(), atol=2e-5, grad_atol=2e-6)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=2e-5, grad_atol=2e-6)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=(2,1)).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), atol=2e-5, grad_atol=2e-6)

  def test_maxpool2x2(self):
    helper_test_op([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), Tensor.max_pool2d)

  def test_maxpool_sizes(self):
    for sz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      helper_test_op([(32,2,110,28)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=sz),
        lambda x: Tensor.max_pool2d(x, kernel_size=sz))

  def test_avgpool2x2(self):
    helper_test_op([(32,2,111,28)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), Tensor.avg_pool2d)

if __name__ == '__main__':
  unittest.main(verbosity=2)

