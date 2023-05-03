#!/usr/bin/env python3
import numpy as np
from tinygrad.tensor import Tensor
from extra.helpers import Timing

if __name__ == "__main__":
  sz = 1024*1024*256  # 1 GB

  with Timing("CPU creation: ", on_exit=lambda x: f", {(sz*4*2)/x:.2f} GB/sec"):
    c0 = Tensor.ones(sz, device="cpu").realize()
    c1 = (Tensor.ones(sz, device="cpu")/2).realize()

  with Timing("CPU -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a0 = c0.to('gpu:0').realize()
  with Timing("CPU -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b1 = c1.to('gpu:1').realize()

  # cross copy. this is going through the CPU
  with Timing("0 -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a1 = a0.to('gpu:1').realize()
  with Timing("1 -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b0 = b1.to('gpu:0').realize()

  # sum (NOTE: without DEBUG=2, these timings are wrong)
  with Timing("0 -> 0 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab0 = (a0 + b0).realize()
  with Timing("1 -> 1 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab1 = (a1 + b1).realize()

  # cross device sum (does this work?)
  # is this making a copy first? is that copy through the CPU?
  # the slowness comes from the *blocking* clprg call, is this pyopencl?
  with Timing("0+1 -> 0 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx0 = (a0 + b1).realize()

  with Timing("0+1 -> 1 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx1 = (b1 + a0).realize()

  # devices
  print(ab0)
  print(ab1)
  print(abx0)
  print(abx1)

  # same
  print("testing")
  np.testing.assert_allclose(ab0.numpy(), ab1.numpy())
  np.testing.assert_allclose(ab0.numpy(), abx0.numpy())
  np.testing.assert_allclose(ab0.numpy(), abx1.numpy())

