#!/usr/bin/env python
import builtins
try:
  import line_profiler
  prof = line_profiler.LineProfiler()
  builtins.__dict__['profile'] = prof
  # add @profile decorator to probe
except ImportError:
  pass

import cProfile
import unittest
from tinygrad.tensor import Tensor

def profile_conv(bs, chans, conv, cnt=10):
  img = Tensor.zeros(bs, 1, 28, 28)
  conv = Tensor.randn(chans, 1, conv, conv)
  for i in range(cnt):
    out = img.conv2d(conv)
    g = out.mean().backward()

class TestConvSpeed(unittest.TestCase):
  def test_forward_backward_3x3(self):
    pr = cProfile.Profile()
    pr.enable()
    profile_conv(128, 16, 3)
    pr.disable()
    pr.print_stats(sort='time')

    if prof is not None:
      prof.print_stats()

if __name__ == '__main__':
  unittest.main()

