#!/usr/bin/env python3
"""
the sequence parallel prefill's q8 all_gather alone (a (1, T, D) half split on its rows over N devices): ms a gather of every way, and the
bits of each way against the ring of copies (the default). MODES: ring (13 copy engine steps), oneshot (every part to every device at once),
ringk (ring_all_gather_q8: the ring kernel on the compute queue). run: DEV=AMD N=14 PYTHONPATH=. python3 extra/glm_ag_bench.py
"""
import time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_glm_dsa import all_gather_q8, ring_all_gather_q8, sp_reduce_scatter

N, T, D, K = getenv("N", 14), getenv("T", 2688), getenv("D", 6144), getenv("K", 50)
devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))
Tensor.manual_seed(0)
x = (Tensor.randn(1, T, D) * 3).half().shard(devs, axis=1).contiguous().realize()
fns = {"ring": lambda x: all_gather_q8(x, 1, ring=True), "oneshot": lambda x: all_gather_q8(x, 1, ring=False),
       "ringk": lambda x: ring_all_gather_q8(x), "rs": lambda p: sp_reduce_scatter(p)}
# (rs: the sequence parallel reduce scatter of every device's float partials (N, 1, T, D) to the rows (SP_RS: q4 default))
part = (Tensor.randn(N, 1, T, D) * 3).shard(devs, axis=0).contiguous().realize() if "rs" in getenv("MODES", "ring,oneshot,ringk") else None
def sync():
  for d in devs: Device[d].synchronize()
def per_device(t:Tensor) -> list[np.ndarray]:
  return [Tensor(t.uop.mselect(i)).bitcast(dtypes.uint16).numpy() for i in range(len(devs))]
ref = None
for mode in getenv("MODES", "ring,oneshot,ringk").split(","):
  fn = fns[mode]
  if mode == "ringk" and ring_all_gather_q8(x) is None:
    print("ringk: not supported here")
    continue
  jit = TinyJit(lambda x: fn(x).contiguous().realize())
  inp = part if mode == "rs" else x
  for _ in range(3): out = jit(inp)
  sync()
  st = time.perf_counter()
  for _ in range(K): out = jit(inp)
  sync()
  dt = (time.perf_counter() - st) / K
  outs = per_device(out) if mode != "rs" else []
  if ref is None: ref = outs
  same = mode != "rs" and all(np.array_equal(a, b) for a, b in zip(outs, ref))
  print(f"{mode:8s} {dt*1e3:8.3f} ms a gather ({T}x{D} half over {N}, {T*D*2/N/1e6*9/16:.2f} MB a part)  bits as the first: {same}", flush=True)
