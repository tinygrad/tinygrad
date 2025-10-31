#!/usr/bin/env python
import unittest
import torch
from tinygrad import Tensor

def try_events(starter=None, ender=None):
  starter = starter or torch.cuda.Event(enable_timing=True)
  ender = ender or torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize()
  starter.record()
  ender.record()
  torch.cuda.synchronize()
  _ = starter.elapsed_time(ender)
  return starter, ender


class TestCudaPrimaryContext(unittest.TestCase):
  @unittest.skipIf(not __import__("torch").cuda.is_available(), "CUDA not available")
  def test_tinygrad_uses_primary_context_for_events(self):
    pre_s, pre_e = try_events()
    _ = Tensor.randn(64, 64, device="CUDA:0").realize()
    # reuse events created before tinygrad touched CUDA
    try_events(pre_s, pre_e)

if __name__ == "__main__":
  unittest.main()
