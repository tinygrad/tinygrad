import os
import torch
import numpy as np
import unittest
import timeit
import functools
from tinygrad.tensor import Tensor, GPU, ANE, Device

def helper_test_op(shps, torch_fxn, tinygrad_fxn, atol=0, rtol=1e-6, grad_atol=0, grad_rtol=1e-6, device=Device.CPU, forward_only=False):
  torch.manual_seed(0)
  ts = [torch.rand(x, requires_grad=True) for x in shps]
  tst = [Tensor(x.detach().numpy()) for x in ts]
  if device==Device.GPU:
    tst = [x.gpu() for x in tst]
  elif device==Device.ANE:
    tst = [x.ane() for x in tst]

  out = torch_fxn(*ts)
  ret = tinygrad_fxn(*tst)

  np.testing.assert_allclose(ret.cpu().data, out.detach().numpy(), atol=atol, rtol=rtol)

  if not forward_only:
    out.mean().backward()
    ret.mean().backward()

    for t, tt in zip(ts, tst):
      np.testing.assert_allclose(t.grad, tt.cpu().grad.data, atol=grad_atol, rtol=grad_rtol)

  # speed
  torch_fp = timeit.Timer(functools.partial(torch_fxn, *ts)).timeit(5) * 1000/5
  tinygrad_fp = timeit.Timer(functools.partial(tinygrad_fxn, *tst)).timeit(5) * 1000/5

  if not forward_only:
    torch_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), torch_fxn, ts)).timeit(5) * 1000/5
    tinygrad_fbp = timeit.Timer(functools.partial(lambda f,x: f(*x).mean().backward(), tinygrad_fxn, tst)).timeit(5) * 1000/5
  else:
    torch_fbp, tinygrad_fbp = np.nan, np.nan

  print("testing %30r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms" % (shps, torch_fp, tinygrad_fp, torch_fbp-torch_fp, tinygrad_fbp-tinygrad_fp))

# TODO: everywhere you see this, make the op work on GPU
def cpu_only(func):
  def wrapper(self):
    if self.device == Device.CPU:
      func(self)
  return wrapper

class TestOps(unittest.TestCase):
  device=Device.CPU

  def test_add(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x+y, Tensor.add, device=self.device)
  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub, device=self.device)
  def test_mul(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x*y, Tensor.mul, device=self.device)
  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div, device=self.device)
  def test_pow(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y, Tensor.pow, device=self.device)
  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), Tensor.sqrt, device=self.device)
  def test_relu(self):
    helper_test_op([(45,65)], lambda x: x.relu(), Tensor.relu, device=self.device)
  def test_leakyrelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.leaky_relu(x,0.01), Tensor.leakyrelu, device=self.device)
  def test_abs(self):
    helper_test_op([(45,65)], lambda x: torch.abs(x), Tensor.abs, device=self.device)
  def test_log(self):
    helper_test_op([(45,65)], lambda x: torch.log(x), Tensor.log, device=self.device)
  def test_exp(self):
    helper_test_op([(45,65)], lambda x: torch.exp(x), Tensor.exp, device=self.device)
  def test_sigmoid(self):
    helper_test_op([(45,65)], lambda x: x.sigmoid(), Tensor.sigmoid, device=self.device)
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.matmul(y), Tensor.dot, device=self.device)
  def test_multidot(self):
    helper_test_op([(10,45,65), (10,65,45)], lambda x,y: x @ y, Tensor.dot, device=self.device)
    helper_test_op([(3,3,45,65), (3,3,65,45)], lambda x,y: x @ y, Tensor.dot, device=self.device)
  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum(), Tensor.sum, device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)), lambda x: Tensor.sum(x, axis=(1,2)), device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1), lambda x: Tensor.sum(x, axis=1), device=self.device)
  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max(), Tensor.max, device=self.device)
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5), device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: Tensor.max(x, axis=1), device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0].mul(0.5), lambda x: Tensor.max(x, axis=1).mul(0.5), device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: Tensor.max(x, axis=1), device=self.device)
  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)), lambda x: Tensor.mean(x, axis=(1,2)), device=self.device)
  def test_logsoftmax(self):
    helper_test_op([(45,65)], lambda x: torch.nn.LogSoftmax(dim=1)(x), Tensor.logsoftmax, atol=1e-7, grad_atol=1e-7, device=self.device)
  def test_tanh(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), Tensor.tanh, atol=1e-6, grad_atol=1e-6, device=self.device)
  def test_topo_sort(self):
    helper_test_op([(45,65)], lambda x: (x+x)*x, lambda x: x.add(x).mul(x), atol=1e-6, grad_atol=1e-6, device=self.device)

  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2, lambda x: x*2, device=self.device)
  def test_scalar_rmul(self):
    helper_test_op([(45,65)], lambda x: 2*x, lambda x: 2*x, device=self.device)

  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2, lambda x: x-2, device=self.device)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x, lambda x: 2-x, device=self.device)

  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((5,13,24,16), (5,1,24,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          helper_test_op(shapes, torch_op, tinygrad_op, device=self.device)


  def test_broadcast_partial(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)),
                     ((4,1), (4,5)), ((1,4), (5,4))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          # NOTE: ANE backwards?
          helper_test_op(shapes, torch_op, tinygrad_op, device=self.device, forward_only=self.device!=Device.CPU)

  def test_pad2d(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad2d(padding=(1,2,3,4)), device=self.device)

  def test_transpose(self):
    helper_test_op([(3,3,3)], lambda x: x.transpose(1,2), lambda x: x.transpose(order=(0,2,1)), device=self.device)
    helper_test_op([(21,22,23,24)], lambda x: x.movedim((3,0,2,1),(0,1,2,3)), lambda x: x.transpose(order=(3,0,2,1)), device=self.device)
    helper_test_op([(3,4,5,6)], lambda x: x.movedim((3,2,1,0),(0,1,2,3)), lambda x: x.transpose(order=(3,2,1,0)), device=self.device)

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,3,6,6)), lambda x: x.reshape(shape=(-1,3,6,6)), device=self.device)
    helper_test_op([(4,3,6,6)], lambda x: torch.reshape(x, (-1,1,6,6)), lambda x: x.reshape(shape=(-1,1,6,6)), device=self.device)

  def test_detach(self):
    helper_test_op([(4,3,6,6)], lambda x: x.detach(), lambda x: x.detach(), device=self.device, forward_only=True)

  def test_conv2d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for groups in [1,3] if cin == 3 else [1]:
          for H in [1,2,5]:
            for W in [1,2,3,5]:
              with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
                helper_test_op([(bs,cin,11,28), (6,cin//groups,H,W)],
                  lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups).relu(),
                  lambda x,w: Tensor.conv2d(x,w,groups=groups).relu(), device=self.device, grad_rtol=1e-5)

  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    with self.subTest(stride := 2):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=2).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=stride).relu(), device=self.device)
    with self.subTest(stride := (2,1)):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=stride).relu(),
        lambda x,w: Tensor.conv2d(x,w,stride=(2,1)).relu(), device=self.device)

  def test_maxpool2d(self):
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz), device=self.device)

  def test_avgpool2d(self):
    shape = (32,2,111,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1), shape[2:]]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), device=self.device, rtol=1e-5)

@unittest.skipUnless(GPU, "Requires GPU")
class TestOpsGPU(TestOps):
  device=Device.GPU

@unittest.skipUnless(ANE, "Requires ANE")
class TestOpsANE(TestOps):
  device=Device.ANE

if __name__ == '__main__':
  unittest.main(verbosity=2)
