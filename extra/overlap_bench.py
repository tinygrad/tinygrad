#!/usr/bin/env python3
"""
do a device's copies (peer to peer, its copy engine) and its kernels run at the same time in a jit? times of a jit of: a copy alone, a matmul
alone, both (independent), on DEVS (two devices: the copy from the first into the second, the matmul on the second)
env: MB (64), N (4096: the matmul's size, half), REPS (20), KS (matmuls in the jit, 4)
run: DEV=PCI+AMD VFIO=1 PYTHONPATH=. python3 extra/overlap_bench.py
"""
import time
from tinygrad import Tensor, Device, TinyJit, dtypes, function
from tinygrad.helpers import getenv

d0, d1 = Device.DEFAULT, f"{Device.DEFAULT}:1"
MB, N, REPS, KS = getenv("MB", 64), getenv("N", 4096), getenv("REPS", 20), getenv("KS", 4)
src = Tensor.empty(MB * 2**20 // 2, dtype=dtypes.half, device=d0).contiguous().realize()
x = Tensor.rand(N, N, dtype=dtypes.half, device=d1).realize()
w = [Tensor.rand(N, N, dtype=dtypes.half, device=d1).realize() for _ in range(KS)]
def sync():
  for d in (d0, d1): Device[d].synchronize()

@TinyJit
def copy_only(): return src.to(d1).contiguous().realize()
@TinyJit
def mm_only():
  y = x
  for k in range(KS): y = (y @ w[k]).contiguous()
  return y.realize()
@TinyJit
def both():
  c = src.to(d1).contiguous()
  y = x
  for k in range(KS): y = (y @ w[k]).contiguous()
  return Tensor.realize(c, y)

# two chains of matmuls on one device: on one compute queue, and on COMPUTE:0 and COMPUTE:1 (glm_dsa._on_queue: the call's queue)
x2 = Tensor.rand(N, N, dtype=dtypes.half, device=d1).realize()
@function(precompile=True, allow_implicit=True)
def chain(a:Tensor) -> Tensor:
  for k in range(KS): a = (a @ w[k]).contiguous()
  return a
@TinyJit
def two_one_queue(): return Tensor.realize(chain(x), chain(x2))
@TinyJit
def two_queues():
  from tinygrad.llm.glm_dsa import _on_queue
  Device[d1].compute_queue_n(1)
  return Tensor.realize(chain(x), *_on_queue((chain(x2),), "COMPUTE:1"))
for name, f in (("copy", copy_only), ("matmul", mm_only), ("both", both), ("2chain1q", two_one_queue), ("2chain2q", two_queues)):
  for _ in range(3): f()
  sync()
  st = time.perf_counter()
  for _ in range(REPS): f()
  sync()
  dt = (time.perf_counter() - st) / REPS
  print(f"OVERLAP {name:7s} {dt*1e3:8.3f} ms" + (f"  ({MB/1024/dt:.1f} GB/s)" if name == "copy" else ""), flush=True)
