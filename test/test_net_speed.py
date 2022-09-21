#!/usr/bin/env python
import time
import cProfile
import pstats
import unittest
import torch
from tinygrad.tensor import Tensor, Device

def start_profile():
  import time
  pr = cProfile.Profile(timer=lambda: int(time.time()*1e9), timeunit=1e-6)
  pr.enable()
  return pr

def stop_profile(pr, sort='cumtime'):
  pr.disable()
  ps = pstats.Stats(pr)
  ps.strip_dirs()
  ps.sort_stats(sort)
  ps.print_stats(0.2)

class TestConvSpeed(unittest.TestCase):

  def test_mnist(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    inter_chan, out_chan = 32, 64

    # ****** torch baseline *******

    torch.backends.mkldnn.enabled = False

    conv = 3
    inter_chan, out_chan = 32, 64
    c1 = torch.randn(inter_chan,1,conv,conv, requires_grad=True)
    c2 = torch.randn(out_chan,inter_chan,conv,conv, requires_grad=True)
    l1 = torch.randn(out_chan*5*5, 10, requires_grad=True)

    c2d = torch.nn.functional.conv2d
    mp = torch.nn.MaxPool2d((2,2))
    lsm = torch.nn.LogSoftmax(dim=1)

    cnt = 5
    fpt, bpt = 0.0, 0.0
    for i in range(cnt):
      et0 = time.time()
      x = torch.randn(128, 1, 28, 28, requires_grad=True)
      x = mp(c2d(x,c1).relu())
      x = mp(c2d(x,c2).relu())
      x = x.reshape(x.shape[0], -1)
      out = lsm(x.matmul(l1))
      out = out.mean()
      et1 = time.time()
      out.backward()
      et2 = time.time()
      fpt += (et1-et0)
      bpt += (et2-et1)

    fpt_baseline = (fpt*1000/cnt)
    bpt_baseline = (bpt*1000/cnt)
    print("torch forward pass:  %.3f ms" % fpt_baseline)
    print("torch backward pass: %.3f ms" % bpt_baseline)

    # ****** tinygrad compare *******

    c1 = Tensor(c1.detach().numpy())
    c2 = Tensor(c2.detach().numpy())
    l1 = Tensor(l1.detach().numpy())

    cnt = 5
    fpt, bpt = 0.0, 0.0
    for i in range(1+cnt):
      et0 = time.time()
      x = Tensor.randn(128, 1, 28, 28)
      x = x.conv2d(c1).relu().avg_pool2d()
      x = x.conv2d(c2).relu().max_pool2d()
      x = x.reshape(shape=(x.shape[0], -1))
      out = x.dot(l1).logsoftmax()
      out = out.mean()
      et1 = time.time()
      out.backward()
      et2 = time.time()
      if i == 0:
        pr = start_profile()
      else:
        fpt += (et1-et0)
        bpt += (et2-et1)

    stop_profile(pr, sort='time')
    fpt = (fpt*1000/cnt)
    bpt = (bpt*1000/cnt)
    print("forward pass:  %.3f ms, %.2fx off baseline %.3f ms" % (fpt, fpt/fpt_baseline, fpt_baseline))
    print("backward pass: %.3f ms, %.2fx off baseline %.3f ms" % (bpt, bpt/bpt_baseline, bpt_baseline))


if __name__ == '__main__':
  unittest.main()
