import time
import functools
from tinygrad import Tensor, Device
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import ReduceOps, BinaryOps, GlobalCounters
from tinygrad.features.multi import MultiLazyBuffer, ring_allreduce
from tinygrad.features.jit import TinyJit
from tinygrad.realize import create_schedule, run_schedule
from tinygrad.helpers import getenv
from typing import List, Union

def naive_allreduce(op: ReduceOps, lbs: List[LazyBuffer]):
  bop = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[op]
  return [functools.reduce(lambda x,y: x.e(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

def realize(x: Union[LazyBuffer, List[LazyBuffer]]):
  x = x if isinstance(x, list) else [x]
  run_schedule(create_schedule(x))
  for lb in x: Device[lb.device].synchronize()

def test(impl, devs: List[str], N: int, iters:int = 10):
  def _wrapped(impl, op: ReduceOps, t: Tensor) -> Tensor:
    return Tensor(MultiLazyBuffer(impl(op, t.lazydata.lbs), 0), device=devs)
  _jitted = TinyJit(_wrapped) if getenv("USEJIT", 1) == 1 else _wrapped

  secs, gflops, gbs = 0, 0, 0
  for i in range(-2, iters):
    GlobalCounters.reset()
    lbs = [Tensor.full((N,), float(1+i), device=d).contiguous().lazydata for i,d in enumerate(devs)]
    realize(lbs)
    start = time.time()
    realize(_jitted(impl, ReduceOps.SUM, Tensor(MultiLazyBuffer(lbs, 0), device=devs)).lazydata.lbs)
    end = time.time()
    if i < 0:
      # First time is slow due to kernel compilation
      continue
    i_secs = end-start
    i_gflops = GlobalCounters.global_ops/i_secs/10**9
    i_gbs = (N*4)/i_secs/10**9
    print(f"{impl.__name__} iter {i+1}/{iters}: {i_secs:.6f} sec {i_gflops:.2f} GFLOP/s {i_gbs:.2f} GB/s")
    secs += i_secs
    gflops += i_gflops
    gbs += i_gbs

  return (gflops/iters, gbs/iters, secs/iters)


def main():
  dev, n_gpus = Device.DEFAULT, getenv("GPUS", 6) # number of gpus
  devs = tuple([f"{dev}:{x}" for x in range(n_gpus)])

  sz = getenv("SZ", 1000) * 10**6 # size of data on each gpu
  f32 = 4 # 4 bytes
  N = sz//f32

  print(f"Using {sz/10**9:.2f} GB of numbers on each of {n_gpus} GPUs, {n_gpus*sz/10**9:.2f} GB total.")
  (ring_gflops, ring_gbs, ring_secs) = test(ring_allreduce, devs, N)
  (naive_gflops, naive_gbs, naive_secs) = test(naive_allreduce, devs, N)
  print(f"Ring:\n  {ring_secs:.6f} seconds/iter\n  {ring_gflops:.2f} GFLOP/s\n  {ring_gbs:.2f} GB/s")
  print(f"Naive:\n  {naive_secs:.6f} seconds/iter\n  {naive_gflops:.2f} GFLOP/s\n  {naive_gbs:.2f} GB/s")

if __name__ == "__main__":
  main()