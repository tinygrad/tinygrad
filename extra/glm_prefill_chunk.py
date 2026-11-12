#!/usr/bin/env python3
"""GLM 5.3 Flash one-card small-loop prefill measurement (adapted from glm53_pfchunk).

NUM_BLOCKS limits model loading to the first blocks without MTP (debug only).
Measures one real chunk per arm at one context position, warmups excluded.
Env: MODEL, SHARD=1, MAXCTX=65536, NUM_BLOCKS=8, N=192, POS=0, REPS=3,
     PROFILE=1 (last rep's GPU classes), COPY_SPLIT=1 (timeline idle by class),
     CHECK=1 (cache-row hash evidence), TOP=60.
     CHECK hashes after the untimed first rep: the chunk's MLA cache rows (indexed on device), every gated deltanet
     block's recurrent state (they read the MoE outputs of the blocks before them) and the returned token.
     FILL=1: the context before POS is prefilled first (chunks of NJ, the same random tokens): the chunk at POS attends
     to real rows, not to an unwritten cache.
Run on the assigned card: AM_POWER_LIMIT=200 PYTHONPATH=. python3 extra/glm_prefill_chunk.py
"""
from __future__ import annotations
import collections
import json
import os
import random
import time

from tinygrad import Device, Tensor, UOp
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.llm.glm5next import Transformer, MLATransformerBlock
import tinygrad.llm.glm5next as g5
import extra.kprof as kprof

# A production shard holds ~11 of these 64 heads, where <=16 is supported. On one
# physical card evaluate all64 using that same kernel in four <=16 head groups.
# This is process-local profiling instrumentation, not a model/runtime change.
_original_sparse = g5.flash_attention_mla_sparse
def _profile_sparse(q: Tensor, cache: Tensor, index: Tensor, scale: float, valid_len, grouped=None):
  if q.shape[1] <= 16 or isinstance(q.device, tuple): return _original_sparse(q, cache, index, scale, valid_len, grouped)
  return Tensor.cat(*[_original_sparse(q[:, a:min(a+16, q.shape[1])], cache, index, scale, valid_len, grouped)
                      for a in range(0, q.shape[1], 16)], dim=1)
g5.flash_attention_mla_sparse = _profile_sparse

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
# CLOCKCAP="GFXCLK=2300": the gfx clock cap the server sets for prefill (serve.py CLOCKS, AMIdleClocks.set_caps); unset: none (the default)
if (cap:=getenv("CLOCKCAP", "")):
  from tinygrad.runtime.ops_amd import AMIdleClocks
  AMIdleClocks.set_caps(cap)
  print(f"CLOCKCAP {cap} devs={len(AMIdleClocks.devs)}", flush=True)
# SEQ_ONLY=1: the server's path instead (Transformer.seq_prefill: the region's jits, the slot's states, the env's cache formats): a prompt of
# SEQ_TOKENS (default 10 chunks of N) prefilled from 0, REPS times; the ms a chunk = its time over its chunks. PROFILE=1: the census of the last
if getenv("SEQ_ONLY"):
  assert model.seqs_supported(), "SEQ_ONLY: the server's sequences are not supported with this env/devices"
  L = getenv("SEQ_TOKENS", N * 10)
  stoks = [random.randrange(1000, 50000) for _ in range(L)]
  model.seq_prefill(0, 0, stoks, 0, 0)  # (the jits captured)
  sync()
  for rep in range(REPS):
    last = rep == REPS-1
    if last and getenv("PROFILE"): kprof.reset(*devs)
    st = time.perf_counter()
    tok = model.seq_prefill(0, 0, stoks, 0, 0)
    dt = time.perf_counter() - st
    print(f"SEQ L={L} N={N} rep={rep} ms={dt*1e3:.3f} ms_per_chunk={dt*1e3/(L/N):.3f} tok={tok}", flush=True)
    if rep == 0 and getenv("CHECK"):
      import hashlib
      from tinygrad.llm.glm5next import GatedDeltaNetBlock
      h = hashlib.sha256()
      # (the deltanet states of every block and the last token's hidden state (it read every layer's attention over all the rows); the
      # caches themselves are GBs here)
      for blk in model.blk:
        if isinstance(blk, GatedDeltaNetBlock) and hasattr(blk, "recurrent_state"): h.update(blk.recurrent_state.numpy().tobytes())
      h.update(model.seq_spec_h.numpy().tobytes())
      print(f"SEQCHECK sha256={h.hexdigest()[:24]}", flush=True)
  if getenv("PROFILE"):
    times = kprof.times(*devs)
    print(f"PROFILE seq gpu_ms_per_chunk={sum(v[1] for v in times.values())/len(devs)/1e3/(L/N):.3f}", flush=True)
    for name, (count, us) in sorted(times.items(), key=lambda kv: -kv[1][1])[:getenv("TOP", 60)]:
      print(f"  {us/len(devs)/1e3/(L/N):9.3f} ms {count/len(devs)/(L/N):8.1f}x {name}", flush=True)
  mem("end")
  raise SystemExit(0)
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
def fill(pos:int) -> None:
  st, p = time.perf_counter(), 0
  while p < pos:
    n = min(NJ, pos-p)
    sp = v_start_pos.bind(p)
    model(model._token_buf[:, sp:sp+UOp.variable("toks", 1, NJ).bind(n)], sp, temp)
    p += n
  sync()
  print(f"FILL pos={pos} elapsed_s={time.perf_counter()-st:.3f}", flush=True)

def check(pos:int, rep:int, out) -> None:
  import hashlib
  from tinygrad.llm.kernels.amd_glm import SEQ_BLK
  from tinygrad.llm.glm5next import GatedDeltaNetBlock
  h, nd, p = hashlib.sha256(), len(devs), Tensor.arange(pos, pos+N).to(devs[0])
  for blk in model.blk:
    if isinstance(blk, MLATransformerBlock) and hasattr(blk, "cache_s"):
      if nd == 1:
        cs = blk.cache_s.reshape(nd, -1, blk.cache_s.shape[-1])
        h.update(cs[(p//SEQ_BLK) % nd, (p//(SEQ_BLK*nd))*SEQ_BLK + p % SEQ_BLK].numpy().tobytes())
      else: h.update(blk.cache_s.numpy().tobytes())  # (sharded rows: the whole cache, every row written so far)
    elif isinstance(blk, MLATransformerBlock) and hasattr(blk, "cache_k"): h.update(blk.cache_k[:, :, pos:pos+N].numpy().tobytes())
    elif isinstance(blk, GatedDeltaNetBlock) and hasattr(blk, "recurrent_state"): h.update(blk.recurrent_state.numpy().tobytes())
  tok = out.numpy().tobytes() if isinstance(out, Tensor) else repr(out).encode()
  print(f"CHECK pos={pos} rep={rep} sha256={h.hexdigest()[:24]} tok={hashlib.sha256(tok).hexdigest()[:12]}", flush=True)

for pos in POS:
  assert pos + N <= model.max_context
  if getenv("FILL") and pos: fill(pos)
  for rep in range(REPS):
    last = rep == REPS-1
    if last and getenv("PROFILE"): kprof.reset(*devs)
    sp, nt = v_start_pos.bind(pos), UOp.variable("toks", 1, NJ).bind(N)
    sync()
    st = time.perf_counter()
    out = model(model._token_buf[:, sp:sp+nt], sp, temp)
    t_host = time.perf_counter()-st
    sync()
    dt = time.perf_counter()-st
    res[pos].append(dt)
    if getenv("CHECK") and rep == 0:
      try: check(pos, rep, out)
      except Exception as e: print(f"CHECK pos={pos} failed: {e!r}", flush=True)
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
# PYPROF=1: the host side of one chunk call (cProfile, the top functions by own time), after the timed reps
if getenv("PYPROF"):
  import cProfile, pstats, io
  pos = POS[0]
  sp, nt = v_start_pos.bind(pos), UOp.variable("toks", 1, NJ).bind(N)
  sync()
  pr = cProfile.Profile()
  pr.enable()
  model(model._token_buf[:, sp:sp+nt], sp, temp)
  pr.disable()
  sync()
  buf = io.StringIO()
  pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(25)
  print("PYPROF\n" + buf.getvalue(), flush=True)
# BURST=n: n chunks back to back (one sync at the end, as a server's consecutive chunks), the mean of BURST_REPS bursts
if (nb:=getenv("BURST", 0)):
  for pos in POS:
    sp, nt, bursts = v_start_pos.bind(pos), UOp.variable("toks", 1, NJ).bind(N), []
    for _ in range(getenv("BURST_REPS", 3)):
      sync()
      st = time.perf_counter()
      for _ in range(nb): model(model._token_buf[:, sp:sp+nt], sp, temp)
      sync()
      bursts.append((time.perf_counter()-st)/nb)
    print(f"BURST N={N} pos={pos} n={nb} ms_per_chunk={' '.join(f'{b*1e3:.3f}' for b in bursts)} "
          f"min={min(bursts)*1e3:.3f}", flush=True)
for pos, vals in res.items():
  warm = vals[1:] if len(vals) > 1 else vals  # the first invocation includes this chunk size's one-time JIT capture
  print(f"RESULT N={N} NJ={NJ} pos={pos} blocks={len(model.blk)} warm_min_ms={min(warm)*1e3:.3f} "
        f"warm_med_ms={sorted(warm)[len(warm)//2]*1e3:.3f} all_max_ms={max(vals)*1e3:.3f}", flush=True)
mem("end")
