#!/usr/bin/env python3
# cd disassemblers/ && git clone --recursive github.com:geohot/cuda_ioctl_sniffer.git
# LD_PRELOAD=$PWD/disassemblers/cuda_ioctl_sniffer/out/sniff.so GPU=1 python3 test/external/external_multi_gpu.py
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import colored
from tinygrad.helpers import Timing
from tinygrad.runtime.ops_gpu import CL

# TODO: support multidevice in cuda
device = 'gpu'

if __name__ == "__main__":
  sz = 1024*1024*256  # 1 GB
  #sz = 1024*64

  with Timing("CPU creation: ", on_exit=lambda x: f", {(sz*4*2)/x:.2f} GB/sec"):
    c0 = Tensor.ones(sz, device="cpu").realize()
    c1 = (Tensor.ones(sz, device="cpu")/2).realize()

  with Timing("CPU -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a0 = c0.to(f'{device}:0').realize()
    CL.synchronize()
  with Timing("CPU -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b1 = c1.to(f'{device}:1').realize()
    CL.synchronize()

  # cross copy. this is going through the CPU
  with Timing("0 -> CPU -> 1: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    a1 = a0.to(f'{device}:1').realize()
    CL.synchronize()
  with Timing("1 -> CPU -> 0: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    b0 = b1.to(f'{device}:0').realize()
    CL.synchronize()

  # sum
  with Timing("0+0 -> 0 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab0 = (a0 + b0).realize()
    CL.synchronize()
  with Timing("1+1 -> 1 (sum): ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    ab1 = (a1 + b1).realize()
    CL.synchronize()

  # cross device sum (does this work?)
  # is this making a copy first? is that copy through the CPU?
  # the slowness comes from the *blocking* clprg call, is this pyopencl?
  with Timing(colored("0+1 -> 0 (sum): ", "red"), on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx0 = (a0 + b1).realize()
    CL.synchronize()

  with Timing(colored("1+0 -> 1 (sum): ", "red"), on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    abx1 = (b1 + a0).realize()
    CL.synchronize()

  # copy back
  # NOTE: half of this slowness is caused by allocating memory on the CPU
  with Timing("0 -> CPU: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    cc0 = ab0.numpy()
  with Timing("1 -> CPU: ", on_exit=lambda x: f", {(sz*4)/x:.2f} GB/sec"):
    cc1 = ab1.numpy()

  # same
  print("testing")
  np.testing.assert_allclose(cc0, cc1)

  # devices
  print(ab0)
  print(ab1)
  print(abx0)
  print(abx1)
