#!/usr/bin/env python3
"""GLM Flash batched decode pool selection (pool_index, grouped as GLM_SEQ's) on one device, full 262k geometry.

G sequences x R query rows each, their scores over a region's pools (66,048 columns, LP active), the lists of the top TOP pools. CFGS: the
variants "reg,agg" (PIDX_REG: a thread's keys in registers, the passes unrolled; PIDX_AGG: a wave's equal digits one atomic), every one must
give the first one's lists. DIST: rand (a real context's scores, a few ties), zero (an unfilled cache: all the same), half (half of them 0).
Example (GPU box only):
  DEV=AMD G=8 DIST=zero PYTHONPATH=. python3 extra/glm_pool_index_bench.py
"""
import functools, os, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernels import amd_glm
from tinygrad.uop.ops import UOp

G, R, KP, TOP = int(os.environ.get("G", 8)), int(os.environ.get("R", 3)), 4, int(os.environ.get("TOP", 512))
LP, STRIDE, CALLS, ITERS = int(os.environ.get("LP", 65536)), 66048, int(os.environ.get("CALLS", 13)), int(os.environ.get("ITERS", 20))
DIST = os.environ.get("DIST", "rand")
dev = Device.DEFAULT
assert dev.startswith("AMD"), "run this benchmark only on an assigned AMD GPU box"
rng = np.random.default_rng(0)
sc = (rng.standard_normal((G*R, STRIDE), dtype=np.float32) * 0.3 + 0.2).astype(np.float32)
sc[:, ::97] = sc[:, :1]
if DIST == "zero": sc[:] = 0
if DIST == "half": sc[:, ::2] = 0
scores = [Tensor(sc, device=dev).realize()] + [Tensor(sc + 0, device=dev).realize() for _ in range(CALLS - 1)]
positions = Tensor(np.full(G, LP*KP - R, dtype=np.int32), device=dev).realize()
zero = UOp.const(0, dtypes.weakint)

def lists(s:Tensor, reg:bool, agg:bool) -> Tensor:
  out = Tensor.empty(G*R, TOP*KP + KP + 1, dtype=dtypes.int32, device=dev)
  return Tensor.custom_kernel(out, s, positions, fxn=functools.partial(amd_glm._pool_index_kernel, q_start=zero, pools=zero, k=TOP, kpool=KP,
                                                                         reg=reg, agg=agg))[0]

def valid(o:np.ndarray) -> list:
  # (a row's list: its count (last) of keys, the rest of the row unwritten)
  return [(int(r[-1]), r[:r[-1]].tobytes()) for r in o]

res: dict[str, list] = {}
for cfg in os.environ.get("CFGS", "0,0 0,1 1,0").split():
  reg, agg = (bool(int(x)) for x in cfg.split(","))
  @TinyJit
  def step(): return [lists(s, reg, agg).contiguous() for s in scores]
  for _ in range(3): outs = step()
  Device[dev].synchronize()
  t0 = time.perf_counter()
  for _ in range(ITERS): outs = step()
  Device[dev].synchronize()
  us = (time.perf_counter() - t0) / ITERS / CALLS * 1e6
  res[cfg] = [valid(o.numpy()) for o in outs]
  ref = next(iter(res))
  same = all(a == b for a, b in zip(res[cfg], res[ref]))
  print(f"REG,AGG={cfg} DIST={DIST} G={G} R={R} LP={LP} TOP={TOP}: {us:.1f} us/call ({us*CALLS/1000:.3f} ms/{CALLS} calls) "
        f"count0={res[cfg][0][0][0]} identical_to_{ref}: {same}", flush=True)
