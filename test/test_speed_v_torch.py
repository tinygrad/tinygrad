import os
import unittest
import torch
torch.set_num_threads(1)
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

IN_CHANS = [int(x) for x in os.getenv("IN_CHANS", "1,16,64").split(",")]

def test_speed(f1, *args):
  ets = []
  ret = None
  for _ in range(CNT):
    del ret
    st = time.monotonic()
    ret = f1(*args)
    et = (time.monotonic() - st) * 1000
    ets.append(et)
  return ret.numpy().sum(), np.median(ets)

def test_generic_square(name, N, f1, f2):
  torch.manual_seed(0)
  torch_a = torch.rand(N, N) - 0.5
  torch_b = torch.rand(N, N) - 0.5
  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy())

  with torch.no_grad():
    val_torch, et_torch = test_speed(f1, torch_a, torch_b)
  val_tinygrad, et_tinygrad = test_speed(lambda *args: f2(*args).realize(), tiny_a, tiny_b)

  print(f"{name} {N}x{N} {et_torch:7.2f} ms in torch, {et_tinygrad:7.2f} ms in tinygrad, {et_tinygrad/et_torch:7.2f}x slower", val_torch, val_tinygrad)
  relative_error = abs((val_tinygrad-val_torch)/val_torch)
  assert relative_error < 0.01, f"relative error too high: {relative_error}"

CNT = 5
class TestSpeed(unittest.TestCase):
  def test_sum(self):
    def f(a, b): return a.sum()
    test_generic_square('sum', 4096, f, f)

  def test_permute(self):
    def f1(a, b): return a.permute(1,0).contiguous()
    def f2(a, b): return a.permute(1,0) + 0
    test_generic_square('permute', 4096, f1, f2)

  def test_neg(self):
    def f(a, b): return -a
    test_generic_square('neg', 4096, f, f)

  def test_exp(self):
    def f(a, b): return a.exp()
    test_generic_square('exp', 2048, f, f)

  def test_relu(self):
    def f(a, b): return a.relu()
    test_generic_square('relu', 4096, f, f)

  def test_max(self):
    def f(a, b): return a.max()
    test_generic_square('max', 4096, f, f)

  def test_mul_sum(self):
    def f(a, b): return (a*b).sum()
    test_generic_square('mul_sum', 4096, f, f)

  def test_add(self):
    for N in [1024, 4096]:
      def f(a, b): return a + b
      test_generic_square('add', N, f, f)

  def test_add_sq(self):
    def f(a, b): return a*a + b*b
    test_generic_square('add_sq', 4096, f, f)

  def test_gemm(self):
    def f(a, b): return a @ b
    test_generic_square('gemm', 512, f, f)

  def test_conv2d(self):
    torch.manual_seed(0)
    for bs in [32]:
      for in_chans in IN_CHANS:
        for out_chans in [16]:
          img_size = 32
          torch_dat = torch.rand(bs, in_chans, img_size, img_size)
          torch_conv = torch.nn.Conv2d(in_chans, out_chans, 3, bias=None)

          tiny_dat = Tensor(torch_dat.cpu().numpy())
          tiny_conv = Conv2d(in_chans, out_chans, 3, bias=None)
          tiny_conv.weight = Tensor(torch_conv.weight.detach().cpu().numpy())

          def f1(): return torch_conv(torch_dat)
          def f2(): return tiny_conv(tiny_dat).realize()

          with torch.no_grad():
            val_torch, et_torch = test_speed(f1)
          val_tinygrad, et_tinygrad = test_speed(f2)

          print(f"bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} {et_torch:7.2f} ms in torch, {et_tinygrad:7.2f} ms in tinygrad, {et_tinygrad/et_torch:7.2f}x slower", val_torch, val_tinygrad)
          relative_error = abs((val_tinygrad-val_torch)/val_torch)
          assert relative_error < 0.01

if __name__ == '__main__':
  unittest.main()
