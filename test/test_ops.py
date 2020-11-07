import torch
import numpy as np
import unittest
import timeit
import functools
from tinygrad.tensor import Tensor, GPU

def helper_test_op(shps, torch_fxn, tinygrad_fxn, atol=1e-7, grad_atol=1e-7, gpu=False, forward_only=False):
  ts = [torch.rand(x, requires_grad=True) for x in shps]
  tst = [Tensor(x.detach().numpy()) for x in ts]
  if gpu:
    tst = [x.cuda() for x in tst]

  out = torch_fxn(*ts)
  ret = tinygrad_fxn(*tst)

  np.testing.assert_allclose(ret.cpu().data, out.detach().numpy(), atol=atol)

  if not forward_only:
    out.mean().backward()
    ret.mean().backward()

    for t, tt in zip(ts, tst):
      np.testing.assert_allclose(t.grad, tt.grad.cpu().data, atol=grad_atol)

  # speed
  torch_fp = timeit.Timer(functools.partial(torch_fxn, *ts)).timeit(5) * 1000/5
  tinygrad_fp = timeit.Timer(functools.partial(tinygrad_fxn, *tst)).timeit(5) * 1000/5

  if not forward_only:
    torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_fxn, ts)).timeit(5) * 1000/5
    tinygrad_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), tinygrad_fxn, tst)).timeit(5) * 1000/5
  else:
    torch_fbp, tinygrad_fbp = np.nan, np.nan

  print("testing %30r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms" % (shps, torch_fp, tinygrad_fp, torch_fbp-torch_fp, tinygrad_fbp-tinygrad_fp))

class TestOps(unittest.TestCase):
  gpu = False
  def test_add(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x+y, Tensor.add, gpu=self.gpu)
  def test_broadcast_add(self):
    helper_test_op([(1,32,32,32), (1,32,1,1)], lambda x,y: x+y, Tensor.add, gpu=self.gpu, forward_only=True)
  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub, gpu=self.gpu)
  def test_mul(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x*y, Tensor.mul, gpu=self.gpu)
  def test_div(self):
    # TODO: why does this need more tolerance?
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div, atol=1e-3, grad_atol=1e-3, gpu=self.gpu)
  def test_pow(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y, Tensor.pow, gpu=self.gpu)
  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), Tensor.sqrt, gpu=self.gpu)
  def test_relu(self):
    helper_test_op([(45,65)], lambda x: x.relu(), Tensor.relu, gpu=self.gpu)
  def test_sigmoid(self):
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid, gpu=self.gpu)
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, atol=1e-5, gpu=self.gpu)
  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum(), Tensor.sum, atol=1e-4, gpu=self.gpu)
  def test_logsoftmax(self):
    helper_test_op([(45,65)], lambda x: torch.nn.LogSoftmax(dim=1)(x), Tensor.logsoftmax, atol=1e-5, gpu=self.gpu)

  def test_pad2d(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,1,1,1)), lambda x: x.pad2d(padding=(1,1,1,1)), gpu=self.gpu, forward_only=True)

  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for groups in [1,3] if cin == 3 else [1]:
          for H in [1,2,5]:
            for W in [1,2,3,5]:
              helper_test_op([(bs,cin,11,28), (6,cin//groups,H,W)],
                lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
                lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), atol=2e-5, grad_atol=2e-6, gpu=self.gpu, forward_only=self.gpu)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=2).relu(), atol=2e-5, grad_atol=2e-6, gpu=self.gpu, forward_only=self.gpu)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=(2,1)).relu(),
      lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), atol=2e-5, grad_atol=2e-6, gpu=self.gpu, forward_only=self.gpu)

  def test_maxpool2x2(self):
    helper_test_op([(32,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, (2,2)), Tensor.max_pool2d, gpu=self.gpu, forward_only=self.gpu)

  def test_maxpool_sizes(self):
    for sz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      helper_test_op([(32,2,110,28)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=sz),
        lambda x: Tensor.max_pool2d(x, kernel_size=sz), gpu=self.gpu, forward_only=self.gpu)

  def test_avgpool2x2(self):
    # TODO Grad tolerance needs to be slightly relaxed; why?
    helper_test_op([(32,2,111,28)], lambda x: torch.nn.functional.avg_pool2d(x, (2,2)), Tensor.avg_pool2d, gpu=self.gpu, grad_atol=1e-5)

if GPU:
  class TestOpsGPU(TestOps):
    gpu = True

if __name__ == '__main__':
  unittest.main(verbosity=2)

