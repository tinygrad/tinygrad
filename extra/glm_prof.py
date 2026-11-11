#!/usr/bin/env python3
"""
GLM (glm5next / glm-dsa) single sequence decode: step time and the GPU kernel times of NSTEPS steps (PROFILE=1), at a prompt of CTXP tokens.
env: MODEL (gguf), SHARD (6), MAXCTX (262144), CTXP (4096 prompt tokens: of CORPUS, a text file, else random), NSTEPS (64 tokens),
TOP (40 kernels printed), PREFILL_PROF=1: the kernel times of a prefill (a second prompt of CTXP tokens, the first warmed the jits)
run: PROFILE=1 PYTHONPATH=. MLA_SEQ_SHARD=1 python3 extra/glm_prof.py
"""
import os, time, random
from tinygrad import Device
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
import extra.kprof as kprof

t0 = time.perf_counter()
model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
random.seed(0)
if (cp:=os.environ.get("CORPUS")):
  # (text: the drafts of speculative decoding accepted as in real use, random tokens' rarely)
  from tinygrad.llm.cli import SimpleTokenizer
  prompt = SimpleTokenizer.from_gguf_kv(kv).encode(open(cp).read()[1000:1000+getenv("CTXP", 4096)*4])[:getenv("CTXP", 4096)]
else: prompt = [random.randrange(1000, 50000) for _ in range(getenv("CTXP", 4096))]
NS = getenv("NSTEPS", 64)
def sync():
  for d in devs: Device[d].synchronize()
def show(t:dict, n:int, what:str, wall:float):
  tot = sum(v[1] for v in t.values()) / len(devs)
  print(f"GPU kernel time per device per {what}: {tot/n/1e3:.3f} ms (of {wall/n*1e3:.3f} ms wall)")
  for name, (c, us) in sorted(t.items(), key=lambda x: -x[1][1])[:getenv("TOP", 40)]:
    print(f"  {us/len(devs)/n:9.1f} us/{what}  {c/len(devs)/n:8.2f}/{what}  {name}")
if getenv("PREFILL_PROF"):
  st = time.perf_counter()
  next(model.generate(prompt))
  sync()
  print(f"warm prefill {len(prompt)} tokens: {time.perf_counter()-st:.1f} s (jits made)", flush=True)
  prompt = [t + 1 for t in prompt]  # (another prompt: nothing cached)
  if getenv("PROFILE"): kprof.reset(*devs)
  st = time.perf_counter()
  tok = next(gen:=model.generate(prompt))
  sync()
  dt = time.perf_counter() - st
  print(f"prefill {len(prompt)} tokens: {dt:.2f} s, {len(prompt)/dt:.0f} tok/s, first token {tok}", flush=True)
  # (GREEDY: the greedy tokens after the prompt, an exactness check of A/B trees)
  if (ng:=getenv("GREEDY", 0)) and not getenv("PROFILE"): print("greedy", [tok] + [next(gen) for _ in range(ng - 1)], flush=True)
  if getenv("PROFILE"): show(kprof.times(*devs), len(prompt) // 1000, "1k tokens", dt)
  if getenv("PROFILE") and getenv("COPY_CTX"):
    # (the kernels around the copies of COPY_CTX MB or more: the last one before a copy on its device's timeline, the first one after it)
    import bisect, collections
    seq = kprof.sequence_timed(*devs)
    ks: dict[str, list] = collections.defaultdict(list)
    for d, n, st0, du in seq:
      if not n.startswith("copy"): ks[d].append((st0, st0 + du, n))
    ctx: collections.Counter = collections.Counter()
    for d, n, st0, du in seq:
      if not n.startswith("copy"): continue
      size, unit = n.split()[1], n.split()[2].rstrip(",")
      if float(size) * {"B": 1/2**20, "KB": 1/1024, "MB": 1, "GB": 1024}.get(unit, 1) < getenv("COPY_CTX", 1): continue
      ends = [e for _, e, _ in ks[d]]
      i = bisect.bisect_right(ends, st0) - 1
      j = bisect.bisect_left([s for s, _, _ in ks[d]], st0 + du)
      ctx[(n.split(",")[0], ks[d][i][2] if i >= 0 else "-", ks[d][j][2] if j < len(ks[d]) else "-")] += 1
    for (n, a, b), c in ctx.most_common(getenv("TOP", 40)): print(f"  {c:6d}  {n:16s}  after {a:40s} before {b}")
  raise SystemExit(0)
gen = model.generate(prompt)
st = time.perf_counter()
next(gen)
print(f"prefill {getenv('CTXP', 4096)} tokens: first token after {time.perf_counter()-st:.1f} s (total {time.perf_counter()-t0:.1f} s)",
      flush=True)
for _ in range(8): next(gen)  # (the decode jit captured)
sync()
if getenv("PROFILE"): kprof.reset(*devs)
st = time.perf_counter()
for _ in range(NS): next(gen)
sync()
dt = time.perf_counter() - st
print(f"decode: {NS} tokens in {dt*1e3:.1f} ms: {dt/NS*1e3:.2f} ms/token, {NS/dt:.1f} tok/s (spec drafts {model.config.spec_drafts})", flush=True)
if getenv("PROFILE"): show(kprof.times(*devs), NS, "token", dt)
