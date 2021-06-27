# Just a temp file to make testing pooling easier to debug

import os
import torch
import numpy as np
import unittest
import timeit
import functools
from tinygrad.tensor import Tensor, DEFAULT_DEVICE, Device

def helper_test_op(shps, torch_fxn, tinygrad_fxn, atol=1e-6, rtol=1e-3, grad_atol=1e-6, grad_rtol=1e-3, forward_only=False, vals=None, a=-0.5, b=20):
  torch.manual_seed(0)
  if shps is None:
    ts = [torch.tensor(x, requires_grad=True) for x in vals]
  else:
    ts = [torch.tensor((np.random.random(size=x).astype(np.float32)+a)*b, requires_grad=True) for x in shps]

  tst = [Tensor(x.detach().numpy()) for x in ts]
  out = torch_fxn(*ts)
  ret = tinygrad_fxn(*tst)

  np.testing.assert_allclose(ret.cpu().data, out.detach().numpy(), atol=atol, rtol=rtol)

  if not forward_only:
    out.mean().backward()
    ret.mean().backward()

    for t, tt in zip(ts, tst):
      print("Our grad (Tinygrad)")
      print(tt.cpu().grad.data)
      print("Correct grad (PyTorch)")
      print(t.grad)
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

class TestOps(unittest.TestCase):
  def test_maxpool2d(self):
    for ksz in [(2,2)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(2,2,6,6)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          # TODO: why is this tolerance so high?
          lambda x: x.maxpool2d(kernel_size=ksz), grad_atol=1e-4)

  def test_strided_maxpool2d(self): # forward only for now
    kernel_sizes = [(2,2), (3,3), (3,2), (5,5), (5,1)]
    strides = [2, 1, 3, 4, 5]
    for i, ksz in enumerate(kernel_sizes):
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz, stride=(strides[i], strides[i])),
          # TODO: why is this tolerance so high?
          lambda x: Tensor.maxpool2d(x, kernel_size=ksz, stride=strides[i]), grad_atol=1e-4, forward_only=True)
  
  def test_strided_avgpool2d(self): # forward only for now
    kernel_sizes = [(2,2), (3,3), (3,2), (5,5), (5,1)]
    strides = [2, 1, 3, 4, 5]
    for i, ksz in enumerate(kernel_sizes):
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,110,28)],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, stride=(strides[i], strides[i])),
          # TODO: why is this tolerance so high?
          lambda x: Tensor.avgpool2d(x, kernel_size=ksz, stride=strides[i]), grad_atol=1e-4, forward_only=True)

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
