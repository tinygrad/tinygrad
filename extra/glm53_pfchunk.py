#!/usr/bin/env python3
"""
GLM-5.3 prefill, the small loop: the time of one prefill chunk (generate's path: the sequence parallel part jits) at chosen start positions,
on the first NUM_BLOCKS blocks of the model (a load of minutes, not of the 79 blocks). The caches before a position are not filled (the chunk
reads what is there): the kernels' and the exchanges' times are a filled cache's, the indexer's lists are not real ones.
env: MODEL, SHARD (14), MAXCTX (262144), NUM_BLOCKS (8), N (2688 tokens a chunk), POS (0,65536,131072,258048 start positions), REPS (3),
     PARTS_T=1: every part jit synchronized and timed alone (the parts' GPU times; the wall of a chunk is then a little longer),
     PROFILE=1: kernel classes of the last rep at every position (TOP), COPY_SPLIT=1: a device's kernels / copies / idle on its timeline
run: PYTHONPATH=. MLA_SEQ_SHARD=1 NUM_BLOCKS=8 python3 extra/glm53_pfchunk.py
"""
import os, time, random, collections
from tinygrad import Device, UOp
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
import extra.kprof as kprof
from tinygrad.llm.kernels.amd_glm_dsa import p2p_late

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 14))
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
N, REPS = getenv("N", 2688), getenv("REPS", 3)
POS = [int(p) for p in str(getenv("POS", "0,65536,131072,258048")).split(",")]
print(f"loaded {len(model.blk)} blocks in {time.perf_counter()-t0:.1f} s, devices {len(devs)}, sizes {model.prefill_chunks()}", flush=True)
def sync():
  for d in devs: Device[d].synchronize()
random.seed(0)
prompt = [random.randrange(1000, 50000) for _ in range(N)]
st = time.perf_counter()
next(model.generate(prompt))
sync()
print(f"warm chunk {N} at 0 (jits made): {time.perf_counter()-st:.1f} s", flush=True)
# every position's tokens in the token buffer (generate wrote the prompt's)
toks = [random.randrange(1000, 50000) for _ in range(max(POS) + N + 1)]
model._token_host[:len(toks)] = __import__("array").array('i', toks)
temp = model._temps[(0.0, 1.0)]
v_start_pos, v_toks = UOp.variable("start_pos", 0, model.max_context-1), UOp.variable("toks", 1, N)

parts_t:list[float] = []
if getenv("PARTS_T"):
  from tinygrad.engine.jit import _TinyJit
  _call = _TinyJit.__call__
  def _timed(self, *a, **k):
    t = time.perf_counter()
    try: return _call(self, *a, **k)
    finally:
      sync()
      parts_t.append(time.perf_counter() - t)
  _TinyJit.__call__ = _timed  # type: ignore[method-assign]

def classify(name:str) -> str:
  n = name.split()[0] if name.startswith("copy") else name
  if n.startswith("copy"): return "copy " + (name.split(",")[0].split(" ", 1)[1] if "," in name else "")
  return n

res:dict[int, list[float]] = collections.defaultdict(list)
for pos in POS:
  assert pos + N <= model.max_context
  for r in range(REPS):
    last = r == REPS - 1
    if last and getenv("PROFILE"): kprof.reset(*devs)
    parts_t.clear()
    sp, nt = v_start_pos.bind(pos), v_toks.bind(N)
    sync()
    st = time.perf_counter()
    model(model._token_buf[:, sp:sp+nt], sp, temp)
    t_host = time.perf_counter() - st
    sync()
    dt = time.perf_counter() - st
    res[pos].append(dt)
    # (a p2p poll that ran out (P2P_AR_TIMEOUT_MS) in the chunk: its values wrong; the server's generate would run the prompt again)
    if getenv("CHECK"):
      # (an exactness check of A/B trees: the sampled token and the last block's latent cache rows of the chunk's positions)
      import hashlib, numpy as np
      from tinygrad.llm.kernels.amd_glm_dsa import SEQ_BLK
      blk, n = model.blk[-1], len(devs)
      arr = blk.cache_s.numpy().reshape(n, -1, blk.cache_s.shape[-1])
      p = np.arange(pos, pos + N)
      rows = arr[(p // SEQ_BLK) % n, (p // (SEQ_BLK*n))*SEQ_BLK + p % SEQ_BLK]
      print(f"CHECK pos {pos} rep {r}: token {model._token_host[pos+N]} cache {hashlib.sha256(rows.tobytes()).hexdigest()[:16]}", flush=True)
    if any(late:=p2p_late(devs, clear=True)): print(f"P2P RUNOUT at pos {pos} rep {r}: epochs {late}", flush=True)
    print(f"chunk {N} at {pos:7d} rep {r}: {dt*1e3:8.1f} ms (host {t_host*1e3:.1f} ms)" +
          (f" parts {' '.join(f'{p*1e3:.1f}' for p in parts_t)}" if parts_t else ""), flush=True)
  if getenv("PROFILE"):
    t = kprof.times(*devs)
    tot = sum(v[1] for v in t.values()) / len(devs)
    print(f"PROFILE pos {pos}: GPU time a device {tot/1e3:.1f} ms of {res[pos][-1]*1e3:.1f} ms wall", flush=True)
    for name, (c, us) in sorted(t.items(), key=lambda x: -x[1][1])[:getenv("TOP", 60)]:
      print(f"  {us/len(devs)/1e3:9.2f} ms  {c/len(devs):8.1f}x  {name}")
    if getenv("COPY_SPLIT"):
      seq = kprof.sequence_timed(*devs)
      def union(iv:list) -> float:
        tot, end = 0.0, float("-inf")
        for a, b in sorted(iv):
          if b > end: tot, end = tot + b - max(a, end), b
        return tot
      for d in sorted({d for d, *_ in seq}):
        kk = [(s, s + du) for dd, n, s, du in seq if dd == d and not n.startswith("copy")]
        cc = [(s, s + du) for dd, n, s, du in seq if dd == d and n.startswith("copy")]
        span = max(b for _, b in kk + cc) - min(a for a, _ in kk + cc)
        # (ksum > kernels: kernels ran at the same time (two compute queues))
        print(f"  {d:8s} span {span/1e3:8.1f} ms  kernels {union(kk)/1e3:8.1f}  copies {union(cc)/1e3:8.1f}  either {union(kk+cc)/1e3:8.1f}"
              f"  ksum {sum(b-a for a, b in kk)/1e3:8.1f}  csum {sum(b-a for a, b in cc)/1e3:8.1f}")
for pos, v in res.items(): print(f"RESULT N={N} pos={pos} blocks={len(model.blk)} min_ms={min(v)*1e3:.1f} med_ms={sorted(v)[len(v)//2]*1e3:.1f}")
