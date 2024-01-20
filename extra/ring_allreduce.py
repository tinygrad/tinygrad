from collections import defaultdict
from typing import List
from tinygrad import Tensor, Device, TinyJit, GlobalCounters
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import ReduceOps, BinaryOps
from tinygrad.features.multi import MultiLazyBuffer, to_sharded
from tinygrad.helpers import Timing
import numpy as np

# N = 4
N = 1024
GPUS = 4
ds = tuple([f"{Device.DEFAULT}:{i+1}" for i in range(GPUS)])
t = Tensor.ones(N, N, N).contiguous().realize().shard(ds, 0)
n = t.numpy()

@TinyJit
def allreduce(t) -> Tensor:
  return t.sum(0)

@TinyJit
def ring_allreduce(t) -> Tensor:
  lbs = t.lazydata.lbs
  rlbs = [lb.r(ReduceOps.SUM, (1, N, N)) for lb in t.lazydata.lbs]
  shlbs = [to_sharded([rlb] * GPUS, 1) for rlb in rlbs]

  reduced = []
  # transpose to make rings
  for i, lbs in enumerate(zip(*shlbs)):
    # rotate each ring to have a different start
    lbs:List[LazyBuffer] = lbs[(i+1):] + lbs[:(i+1)]
    lb:LazyBuffer = lbs[0]
    # ring for scatter reduce
    for dest in lbs[1:]:
      lb = lb.copy_to_device(dest.device).e(BinaryOps.ADD, dest)
    reduced.append(lb)

  gathered = defaultdict(lambda: [None] * GPUS)
  for lb in reduced:
    for dest in ds:
      gathered[dest][ds.index(lb.device)] = lb.copy_to_device(dest)

  catted = {}
  for device, lbs in gathered.items():
    catted[device] = Tensor.stack([Tensor(lb, device=lb.device) for lb in lbs], 1).flatten(end_dim=2).lazydata

  final = MultiLazyBuffer(list(catted.values()), None)
  return Tensor(final)

for i in range(5):
  GlobalCounters.reset()
  with Timing(" ring:"):
    tn = ring_allreduce(t).numpy()
  np.testing.assert_allclose(tn, n.sum(0), atol=1e-4, rtol=1e-4)

for i in range(5):
  GlobalCounters.reset()
  with Timing("naive:"):
    tn = allreduce(t).numpy()
  np.testing.assert_allclose(tn, n.sum(0), atol=1e-4, rtol=1e-4)
