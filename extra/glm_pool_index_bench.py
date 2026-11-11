#!/usr/bin/env python3
"""GLM Flash batched decode pool selection (pool_index, grouped as GLM_SEQ's) on one device, full 262k geometry.

G sequences x R query rows each, their scores over a region's pools (66,048 columns, LP active), the lists of the top TOP pools. Variants
REG=0 (a chunk's keys loaded per pass) and REG=1 (PIDX_REG: the keys in registers, the passes unrolled) must give the same int32 lists.
Example (GPU box only):
  DEV=AMD G=8 PYTHONPATH=. python3 extra/glm_pool_index_bench.py
"""
import functools, os, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernels import amd_glm
from tinygrad.uop.ops import UOp

G, R, KP, TOP = int(os.environ.get("G", 8)), int(os.environ.get("R", 3)), 4, int(os.environ.get("TOP", 512))
LP, STRIDE, CALLS, ITERS = int(os.environ.get("LP", 65536)), 66048, int(os.environ.get("CALLS", 13)), int(os.environ.get("ITERS", 20))
dev = Device.DEFAULT
assert dev.startswith("AMD"), "run this benchmark only on an assigned AMD GPU box"
rng = np.random.default_rng(0)
# (weighted relu sums: mostly positive, some negative, a few ties)
sc = (rng.standard_normal((G*R, STRIDE), dtype=np.float32) * 0.3 + 0.2).astype(np.float32)
sc[:, ::97] = sc[:, :1]
scores = [Tensor(sc, device=dev).realize()] + [Tensor(sc + 0, device=dev).realize() for _ in range(CALLS - 1)]
positions = Tensor(np.full(G, LP*KP - R, dtype=np.int32), device=dev).realize()
zero = UOp.const(0, dtypes.weakint)

def lists(s:Tensor, reg:bool) -> Tensor:
  out = Tensor.empty(G*R, TOP*KP + KP + 1, dtype=dtypes.int32, device=dev)
  return Tensor.custom_kernel(out, s, positions, fxn=functools.partial(amd_glm._pool_index_kernel, q_start=zero, pools=zero, k=TOP, kpool=KP,
                                                                         reg=reg))[0]

res = {}
for reg in [int(x) for x in os.environ.get("CFGS", "0 1").split()]:
  @TinyJit
  def step(): return [lists(s, bool(reg)).contiguous() for s in scores]
  for _ in range(3): outs = step()
  Device[dev].synchronize()
  t0 = time.perf_counter()
  for _ in range(ITERS): outs = step()
  Device[dev].synchronize()
  us = (time.perf_counter() - t0) / ITERS / CALLS * 1e6
  res[reg] = [o.numpy() for o in outs]
  same = all(np.array_equal(a, b) for a, b in zip(res[reg], res[min(res)]))
  print(f"REG={reg} G={G} R={R} LP={LP} TOP={TOP}: {us:.1f} us/call ({us*CALLS/1000:.3f} ms/{CALLS} calls) count0={res[reg][0][0, -1]} "
        f"identical_to_REG={min(res)}: {same}", flush=True)
