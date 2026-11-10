#!/usr/bin/env python3
"""
the flag exchanges of batched decoding (all_gather_q8, q8 reduce_scatter, B=1 flag_partial_sum) alone, many times, with values that change
every exchange: a stale or torn peer payload (a missing L2 write back, a flag seen before its words) shows as
  - all_gather_q8 / partial_sum: the devices' copies differ (each device's own part is local, the others' come from the inboxes)
  - reduce_scatter: the launch's hash differs from a reference run's (REF=<file> of a run with the default write back; OUT=<file> writes one)
env: R (rows a device, 8 = B=16), ITERS (exchanges of each kind a launch), SECS (run time) or LAUNCHES, SEED
"""
import time, sys, hashlib
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels import amd_qwen as aq

N, R, D, ITERS, SECS, LAUNCHES = getenv("N", 6), getenv("R", 8), 2560, getenv("ITERS", 16), getenv("SECS", 60), getenv("LAUNCHES", 0)
devs = tuple(f"AMD:{i}" if i else "AMD" for i in range(N))

@TinyJit
def step(x0:Tensor, p0:Tensor, q0:Tensor) -> list[Tensor]:
  outs = []
  for i in range(ITERS):
    x = (x0 * (1 + i/64) + i/32).cast(dtypes.half)
    outs.append(aq.all_gather_q8(x, 1))
    p = (p0 * (1 - i/128) - i/16).cast(dtypes.half)
    outs.append(aq.reduce_scatter(p, 1))
    q = (q0 * (1 + i/32)).cast(dtypes.half)
    outs.append(aq.partial_sum(q))
  return [o.realize() for o in outs]

def bufs(t:Tensor) -> list[bytes]:
  b = t.uop.base.buffer
  return [bytes(x.as_memoryview()) for x in getattr(b, "bufs", [b])]

ref = open(sys.argv[1]).read().split() if len(sys.argv) > 1 and sys.argv[1] != "-" else None
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
t0, launch, bad, hashes = time.perf_counter(), 0, 0, []
while (launch < LAUNCHES) if LAUNCHES else (time.perf_counter() - t0 < SECS):
  rng = np.random.default_rng(getenv("SEED", 0) * 1000003 + launch)
  x0 = Tensor(rng.standard_normal((1, N*R, D), dtype=np.float32)).shard(devs, axis=1)
  p0 = Tensor(rng.standard_normal((N, 1, N*R, D), dtype=np.float32)).shard(devs, axis=0)
  q0 = Tensor(rng.standard_normal((N, 3*D), dtype=np.float32)).shard(devs, axis=0)
  res = step(x0, p0, q0)
  h = hashlib.md5()
  for k, o in enumerate(res):
    bs = bufs(o)
    if k % 3 != 1 and any(b != bs[0] for b in bs[1:]):
      bad += 1
      diff = [d for d, b in enumerate(bs) if b != bs[0]]
      print(f"MISMATCH launch {launch} iter {k//3} kind {('ag', 'rs', 'ps')[k%3]} devices {diff} differ from device 0", flush=True)
    for b in bs: h.update(b)
  hashes.append(h.hexdigest()[:16])
  if ref is not None and launch < len(ref) and ref[launch] != hashes[-1]:
    bad += 1
    print(f"MISMATCH launch {launch}: hash {hashes[-1]} != reference {ref[launch]}", flush=True)
  if out is not None: out.write(hashes[-1] + "\n")
  launch += 1
  if launch % 200 == 0: print(f"{time.perf_counter()-t0:7.1f}s launch {launch} exchanges {launch*ITERS*3} bad {bad}", flush=True)
print(f"done: {launch} launches, {launch*ITERS*3} exchanges ({launch*ITERS} of each kind), bad {bad}, "
      f"{'checked vs reference ' + str(min(len(ref), launch)) if ref else 'no reference'}", flush=True)
