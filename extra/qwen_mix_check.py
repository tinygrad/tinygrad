#!/usr/bin/env python3
"""Greedy short-suffix continuation through an ordinary prefill (MIX=0) or forced mixed decode steps (MIX=1).

A second sequence decodes beside the mixed one. The initial reply is deliberately not in the new prompt, so admission restores the
original prompt's checkpoint exactly as an OpenAI-style reasoning-drop turn does. Print RESULT with the mixed sequence's generated tokens.
env: MODEL, CORPUS, SHARD=6, MIX=1, L=4096, SUFFIX=24, GEN=48, and the server's SEQ_* / SPEC_DRAFTS settings.
"""
import os, sys, json, time, pathlib
from tinygrad.helpers import getenv, Context
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.engine import SeqEngine

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
with Context(DEBUG=0): model.warmup()
eng = SeqEngine(model)
text = pathlib.Path(os.environ["CORPUS"]).read_text()
corpus = tok.encode(text)
while len(corpus) < 400000: corpus += corpus
L, N, GEN = getenv("L", 4096), getenv("SUFFIX", 24), getenv("GEN", 48)
MIX, temp = getenv("MIX", 1), (0.0, 1.0)
ids = corpus[1000:1000+L]
got = eng.take(ids, 128)
assert got is not None
s, start, restore = got
assert start == 0
first = eng.prefill(s, ids, start, restore, temp)
sout = [first]
while len(sout) < 8: sout += eng.step([s])[0]
eng.stop(s, len(s.tokens))

bgs = []
for i in range(getenv("BG", 1)):
  bg_ids = corpus[100000+10000*i:100000+10000*i+4096]
  got = eng.take(bg_ids, 256)
  assert got is not None
  bg, start, restore = got
  assert start == 0
  eng.prefill(bg, bg_ids, start, restore, temp)
  bgs.append(bg)

nxt = ids + corpus[50000:50000+N]
got = eng.take(nxt, 128)
assert got is not None
s, start, restore = got
assert start == L and restore, (start, restore)
st = time.perf_counter()
out, mix_steps = [], 0
if MIX:
  eng.mix_start(s, nxt, start, restore, temp)
  while s.mix:
    out += eng.step([s, *bgs])[0]
    mix_steps += 1
else: out = [eng.prefill(s, nxt, start, restore, temp)]
while len(out) < GEN:
  out += eng.step([s, *bgs])[0]
print("RESULT " + json.dumps({"name": f"mix{N}", "MIX": MIX, "BG": len(bgs), "L": L, "tokens": out[:GEN], "mix_steps": mix_steps,
                              "secs": round(time.perf_counter()-st, 4)}), flush=True)
print("DONE", file=sys.stderr, flush=True)
