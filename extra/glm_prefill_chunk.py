#!/usr/bin/env python3
"""GLM 5.3 Flash one-card small-loop prefill measurement (adapted from glm53_pfchunk).

NUM_BLOCKS limits model loading to the first blocks without MTP (debug only).
Measures one real chunk per arm at one context position, warmups excluded.
Env: MODEL, SHARD=1, MAXCTX=65536, NUM_BLOCKS=8, N=192, POS=0, REPS=3,
     PROFILE=1 (last rep's GPU classes), COPY_SPLIT=1 (timeline idle by class),
     CHECK=1 (cache-row hash evidence), TOP=60.
Run on the assigned card: AM_POWER_LIMIT=200 PYTHONPATH=. python3 extra/glm_prefill_chunk.py
"""
from __future__ import annotations
import collections
import json
import os
import random
import time

from tinygrad import Device, UOp
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.llm.glm5next import Transformer
import extra.kprof as kprof

started = time.perf_counter()
model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 1))
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
N, NJ, REPS = getenv("N", 192), getenv("NJ", getenv("N", 192)), getenv("REPS", 3)
POS = [int(x) for x in str(getenv("POS", "0")).split(",")]
print(f"LOADED blocks={len(model.blk)} shards={len(devs)} N={N} NJ={NJ} pos={POS} chunks={model.prefill_chunks()} "
      f"elapsed_s={time.perf_counter()-started:.3f}", flush=True)

def sync() -> None:
  for d in devs: Device[d].synchronize()

def vfree_mib(d: str) -> float:
  try: return sum(sz for sz, _, _, free in Device[d].iface.dev_impl.mm.pa_allocator.blocks.values() if free) / 2**20
  except Exception: return float("nan")

def mem(tag: str, pos: int|None = None) -> None:
  used = {d: GlobalCounters.mem_used_per_device.get(d, 0)/2**20 for d in devs}
  print("MEMORY " + json.dumps({"tag": tag, "pos": pos, "used_mib": used,
                                "vram_free_mib": {d: vfree_mib(d) for d in devs}}, sort_keys=True), flush=True)

sync()
mem("loaded")
random.seed(0)
prompt = [random.randrange(1000, 50000) for _ in range(NJ)]
st = time.perf_counter()
next(model.generate(prompt))
sync()
print(f"WARM chunk={NJ} elapsed_s={time.perf_counter()-st:.3f}", flush=True)
mem("warmed")
toks = [random.randrange(1000, 50000) for _ in range(max(POS) + NJ + 1)]
model._token_host[:len(toks)] = __import__("array").array("i", toks)
temp = None  # GLM5Next uses None for exact greedy sampling (not the older GLM DSA tensor pair)
v_start_pos, v_toks = UOp.variable("start_pos", 0, model.max_context-1), UOp.variable("toks", 1, NJ)
res: dict[int, list[float]] = collections.defaultdict(list)
for pos in POS:
  assert pos + N <= model.max_context
  for rep in range(REPS):
    last = rep == REPS-1
    if last and getenv("PROFILE"): kprof.reset(*devs)
    sp, nt = v_start_pos.bind(pos), UOp.variable("toks", 1, NJ).bind(N)
    sync()
    st = time.perf_counter()
    model(model._token_buf[:, sp:sp+nt], sp, temp)
    t_host = time.perf_counter()-st
    sync()
    dt = time.perf_counter()-st
    res[pos].append(dt)
    if getenv("CHECK"):
      import hashlib
      import numpy as np
      from tinygrad.llm.kernels.amd_glm import SEQ_BLK
      blk, nd = model.blk[-1], len(devs)
      arr = blk.cache_s.numpy().reshape(nd, -1, blk.cache_s.shape[-1])
      p = np.arange(pos, pos+N)
      rows = arr[(p//SEQ_BLK) % nd, (p//(SEQ_BLK*nd))*SEQ_BLK + p % SEQ_BLK]
      print(f"CHECK pos={pos} rep={rep} sha256={hashlib.sha256(rows.tobytes()).hexdigest()[:24]}", flush=True)
    print(f"CHUNK N={N} NJ={NJ} pos={pos} rep={rep} wall_ms={dt*1e3:.3f} host_ms={t_host*1e3:.3f}", flush=True)
    mem("chunk", pos)
  if getenv("PROFILE"):
    times = kprof.times(*devs)
    print(f"PROFILE pos={pos} gpu_ms={sum(v[1] for v in times.values())/len(devs)/1e3:.3f} "
          f"last_wall_s={res[pos][-1]*1e3:.3f}", flush=True)
    for name, (count, us) in sorted(times.items(), key=lambda kv: -kv[1][1])[:getenv("TOP", 60)]:
      print(f"  {us/len(devs)/1e3:9.3f} ms {count/len(devs):8.1f}x {name}", flush=True)
    if getenv("COPY_SPLIT"):
      seq = kprof.sequence_timed(*devs)
      def union(intervals: list[tuple[float, float]]) -> float:
        total, end = 0.0, float("-inf")
        for a, b in sorted(intervals):
          if b > end: total, end = total+b-max(a, end), b
        return total
      for d in sorted({d for d, *_ in seq}):
        kk = [(s, s+du) for dd, name, s, du in seq if dd == d and not name.startswith("copy")]
        cc = [(s, s+du) for dd, name, s, du in seq if dd == d and name.startswith("copy")]
        print(f"  {d} kernel_union_ms={union(kk)/1e3:.3f} copy_union_ms={union(cc)/1e3:.3f} "
              f"either_union_ms={union(kk+cc)/1e3:.3f}", flush=True)
for pos, vals in res.items():
  print(f"RESULT N={N} NJ={NJ} pos={pos} blocks={len(model.blk)} min_ms={min(vals)*1e3:.3f} "
        f"med_ms={sorted(vals)[len(vals)//2]*1e3:.3f} max_ms={max(vals)*1e3:.3f}", flush=True)
mem("end")
