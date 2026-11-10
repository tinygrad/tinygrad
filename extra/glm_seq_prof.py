#!/usr/bin/env python3
"""
GLM_SEQ batched decode: the step time of G sequences decoded together (seq_step) and its GPU kernel times (PROFILE=1), every sequence a
corpus prompt of CTXP tokens prefilled on its own region (slot g). the steps' tokens are the model's (greedy; accepted drafts as in use).
env: MODEL, CORPUS, SHARD (6), MAXCTX (65536), GS (1,8,16,32: the step sizes measured, each needs SEQ_SIZES to have it and SEQ_SLOTS >= it),
     CTXP (4096), NSTEPS (32), TOP (40)
run: PROFILE=1 PYTHONPATH=. MLA_SEQ_SHARD=1 GLM_SEQ=1 SEQ_ROWS=... SEQ_SLOTS=32 SEQ_SIZES=1,2,4,8,16,32 python3 extra/glm_seq_prof.py
"""
import os, time, array
from tinygrad import Device
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
import extra.kprof as kprof

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
assert model.config.cache_rows, "GLM_SEQ=1 SEQ_ROWS=..."
tok = SimpleTokenizer.from_gguf_kv(kv)
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
k, CTXP, NS = model.config.spec_drafts, getenv("CTXP", 4096), getenv("NSTEPS", 32)
GS = [int(g) for g in str(getenv("GS", "1,8,16,32")).split(",")]
corpus = tok.encode(open(os.environ["CORPUS"]).read()[:getenv("CHARS", 2_000_000)])
span = -(-(CTXP + 2*NS*(k+1) + 64) // model.REGION_ALIGN) * model.REGION_ALIGN  # (a sequence's region: its prompt and the decoded tokens)
G = max(GS)
assert G * span <= model.seq_rows() and G <= model.seq_slots(), f"{G} regions of {span} rows: SEQ_ROWS >= {G*span}, SEQ_SLOTS >= {G}"
def sync():
  for d in devs: Device[d].synchronize()
st0 = time.perf_counter()
seqs = []
for g in range(G):
  ids = corpus[1000 + g*CTXP*2:1000 + g*CTXP*2 + CTXP]
  first = model.seq_prefill(g, g*span, list(ids), 0, 0)
  seqs.append(dict(slot=g, base=g*span, mtp=len(ids), j=0, par=1, y=[first] + [0]*k, n=0))
print(f"{G} prompts of {CTXP} prefilled in {time.perf_counter()-st0:.1f}s", flush=True)
def step(group:list[dict]):
  outs = model.seq_step([(s["slot"], s["base"], s["mtp"], s["j"], s["par"], s["y"], 0.0) for s in group])
  for s, (sampled, drafts) in zip(group, outs):
    j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
    st = s["mtp"] + s["j"]
    model._token_host[s["base"] + st + 1:s["base"] + st + 2 + j] = array.array('i', sampled[:j+1])
    s["mtp"], s["j"], s["par"], s["y"], s["n"] = st + 1, j, 1 - s["par"], sampled, s["n"] + j + 1
for g in GS:
  group = seqs[:g]
  for _ in range(3): step(group)  # (its jit captured)
  sync()
  n0 = sum(s["n"] for s in group)
  if getenv("PROFILE"): kprof.reset(*devs)
  st = time.perf_counter()
  for _ in range(NS): step(group)
  sync()
  dt = time.perf_counter() - st
  toks = sum(s["n"] for s in group) - n0
  print(f"G={g}: {dt/NS*1e3:.2f} ms/step, {toks/NS/g:.2f} tok/step/seq, {toks/dt:.0f} tok/s aggregate, {toks/dt/g:.1f} tok/s/seq", flush=True)
  if getenv("PROFILE"):
    t = kprof.times(*devs)
    tot = sum(v[1] for v in t.values()) / len(devs)
    print(f"  GPU kernel time per device per step: {tot/NS/1e3:.3f} ms", flush=True)
    for name, (c, us) in sorted(t.items(), key=lambda x: -x[1][1])[:getenv("TOP", 40)]:
      print(f"  {us/len(devs)/NS:9.1f} us/step  {c/len(devs)/NS:6.2f}/step  {name}", flush=True)
