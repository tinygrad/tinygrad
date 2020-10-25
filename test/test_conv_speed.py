#!/usr/bin/env python

# if you'd like to use the line profiler
try:
  import line_profiler
  prof = line_profiler.LineProfiler()
  import builtins
  builtins.__dict__['profile'] = prof
  # add @profile decorator to probe
except ImportError:
  prof = None

import time
import cProfile
import pstats
import unittest
import numpy as np
from tinygrad.tensor import Tensor

def profile_conv(bs, chans, conv, cnt=10):
  img = Tensor.zeros(bs, 1, 28, 28)
  conv = Tensor.randn(chans, 1, conv, conv)
  fpt, bpt = 0.0, 0.0
  for i in range(cnt):
    et0 = time.time()
    out = img.conv2d(conv)
    et1 = time.time()
    g = out.mean().backward()
    et2 = time.time()
    fpt += (et1-et0)
    bpt += (et2-et1)
  return fpt/cnt, bpt/cnt

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
  ps.print_stats(0.3)

  if prof is not None:
    prof.print_stats()

class TestConvSpeed(unittest.TestCase):
  def test_forward_backward_3x3(self):
    # warmup
    profile_conv(128, 16, 3, cnt=1)

    pr = start_profile()
    fpt, bpt = profile_conv(128, 16, 3)
    stop_profile(pr)

    print("forward pass:  %.3f ms" % (fpt*1000))
    print("backward pass: %.3f ms" % (bpt*1000))

  def test_mnist(self):
    # https://keras.io/examples/vision/mnist_convnet/
    conv = 3
    inter_chan, out_chan = 32, 64
    c1 = Tensor.randn(inter_chan,1,conv,conv)
    c2 = Tensor.randn(out_chan,inter_chan,conv,conv)
    l1 = Tensor.randn(out_chan*5*5, 10)

    for i in range(6):
      x = Tensor.randn(128, 1, 28, 28)
      x = x.conv2d(c1).relu().maxpool2x2()
      x = x.conv2d(c2).relu().maxpool2x2()
      x = x.reshape(Tensor(np.array((x.shape[0], -1))))
      out = x.dot(l1).logsoftmax()
      out.mean().backward()
      if i == 0:
        pr = start_profile()

    stop_profile(pr, sort='time')


if __name__ == '__main__':
  unittest.main()

