import torch
import numpy as np
import unittest
import timeit
import functools
from tinygrad.tensor import Tensor

def helper_test_op(shps, torch_fxn, tinygrad_fxn, atol=1e-7, grad_atol=1e-7, gpu=False):
  ts = [torch.rand(x, requires_grad=True) for x in shps]
  tst = [Tensor(x.detach().numpy()) for x in ts]
  if gpu:
    tst = [x.cuda() for x in tst]

  out = torch_fxn(*ts)
  ret = tst[0].__getattr__(tinygrad_fxn)(*tst[1:])

  # TODO: why so inaccurate?
  np.testing.assert_allclose(ret.cpu().data, out.detach().numpy(), atol=atol)

  print(ret, ret.gpu)

  out.mean().backward()
  ret.mean().backward()

  for t, tt in zip(ts, tst):
    np.testing.assert_allclose(t.grad, tt.grad.cpu().data, atol=grad_atol)

  # speed
  torch_fp = timeit.Timer(functools.partial(torch_fxn, *ts)).timeit(5) * 1000/5
  tinygrad_fp = timeit.Timer(functools.partial(tinygrad_fxn, *tst)).timeit(5) * 1000/5

  torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_fxn, ts)).timeit(5) * 1000/5
  tinygrad_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), tinygrad_fxn, tst)).timeit(5) * 1000/5

  print("testing %30r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms" % (shps, torch_fp, tinygrad_fp, torch_fbp-torch_fp, tinygrad_fbp-tinygrad_fp))

class TestOps(unittest.TestCase):
  def test_add(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x+y, "add")
  def test_add_gpu(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x+y, "add", gpu=True)
  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, "sub")
  def test_mul(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x*y, "mul")
  def test_div(self):
    # TODO: why does this need more tolerance?
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, "div", atol=5e-5, grad_atol=1e-5)
  def test_pow(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y, "pow")
  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), "sqrt")

  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for groups in [1,3] if cin == 3 else [1]:
          for H in [2,5]:
            for W in [2,3,5]:
              helper_test_op([(bs,cin,11,28), (6,cin//groups,H,W)],
                lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
                lambda x,w: x.conv2d(w,groups=groups).relu(), atol=2e-5, grad_atol=2e-6)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: x.conv2d(w,stride=2).relu(), atol=2e-5, grad_atol=2e-6)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=(2,1)).relu(),
      lambda x,w: x.conv2d(w,stride=(2,1)).relu(), atol=2e-5, grad_atol=2e-6)

  def test_maxpool2x2(self):
    helper_test_op([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), "max_pool2d")

  def test_maxpool_sizes(self):
    for sz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      helper_test_op([(32,2,110,28)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=sz),
        lambda x: x.max_pool2d(kernel_size=sz))

  def test_avgpool2x2(self):
    helper_test_op([(32,2,111,28)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), "avg_pool2d")

if __name__ == '__main__':
  unittest.main(verbosity=2)

