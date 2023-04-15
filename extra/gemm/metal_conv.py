import os
os.environ["METAL"] = "1"
import numpy as np

BS = 64
CIN = 256
COUT = 256
HW = 32
K = 3
# TODO: this is doing some trick, since with CIN=256 COUT=256 it's over 10.4 TFLOPS.
# are winograd convs less flops?
FLOPS = BS*K*K*CIN*HW*HW*COUT*2

nb = np.random.default_rng().standard_normal(size=(BS,CIN,HW,HW), dtype=np.float32)
nc = np.random.default_rng().standard_normal(size=(COUT,CIN,K,K), dtype=np.float32)

import time, torch, torch.mps
b = torch.from_numpy(nb).to('mps')
c = torch.from_numpy(nc).to('mps')

def torch_prog(b, c):
  st = time.perf_counter()
  a = torch.nn.functional.conv2d(b, c, padding=1)
  torch.mps.synchronize()
  return time.perf_counter() - st
tm = min([torch_prog(b, c) for _ in range(20)])
print(f"{tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS conv in torch")

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.runtime.ops_metal import METAL
b = Tensor(nb)
c = Tensor(nc)
# TODO: slowness without the JIT I suspect comes from a lack of a caching allocator
@TinyJit
def tiny_jit(b, c):
  return b.conv2d(c, padding=1).realize()
def tiny_prog(b, c):
  st = time.perf_counter()
  a = tiny_jit(b, c)
  METAL.synchronize()
  return time.perf_counter() - st
tm = min([tiny_prog(b, c) for _ in range(5)])
print(f"{tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS conv in tinygrad")
