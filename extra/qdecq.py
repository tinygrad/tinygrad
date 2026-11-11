#!/usr/bin/env python3
"""
qdecq.py: the quality of batched decoding's numerics (teacher forced): G sequences of the fixed CORPUS (L tokens each, different offsets)
prefilled, then STEPS steps of speculative decoding with every draft and every step token forced to the corpus (QTF=1 in qwen4exp), the
rows' log probs of the vocab read back. prints a RESULT line a sequence as extra/qwen_kv_quality.py does (the target's log prob, the most
likely token and the TOP most likely (ids, log probs) of every position), so `python3 extra/qwen_kv_quality.py ref.out test.out` compares two
builds (mean NLL, top-1 agreement, KL on the ref's top tokens).
env: MODEL, CORPUS, G=16 (SEQ_SIZES must have it), L=4000, STEPS=40, TOP=64, the server's SEQ_*/SPEC_DRAFTS
"""
import os, json, pathlib, array
import numpy as np
os.environ["QTF"] = "1"
from tinygrad.helpers import getenv, Context
# (WARMUP_JITS: the warm-up makes the memory, the jits are captured on their first call (a run uses a few); WARMUP_JITS=all as a server)
os.environ.setdefault("WARMUP_JITS", "none")
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.engine import SeqEngine

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
if getenv("NOWARM"): model._finish_load()
else:
  with Context(DEBUG=0): model.warmup()
eng = SeqEngine(model)
corpus = tok.encode(pathlib.Path(os.environ["CORPUS"]).read_text()[:4_000_000])
G, L, STEPS, TOP = getenv("G", 16), getenv("L", 4000), getenv("STEPS", 40), getenv("TOP", 64)
k, V = model.config.spec_drafts, model.config.vocab_size
seqs, texts = [], []
for i in range(G):
  text = corpus[50000 + 60000*i: 50000 + 60000*i + L + STEPS*(k+1) + 2]
  ids = text[:L]
  s, start, restore = eng.take(ids, STEPS*(k+1) + 64)
  eng.prefill(s, ids, start, restore, (0.0, 1.0))
  # (the first token after the prompt forced too: the step's first row is the corpus')
  s.y[0] = text[L]
  model._token_host[s.base + L:s.base + L + 1] = array.array('i', [text[L]])
  seqs.append(s); texts.append(text)
res = [{"name": f"seq{i}", "lp": [], "top": [], "ids": [], "lps": []} for i in range(G)]
for step in range(STEPS):
  forced = []
  for s, text in zip(seqs, texts):
    p = s.mtp + s.j  # (the position of the step's first row's token)
    forced += text[p+1:p+1+k] + text[p+1:p+2+k]
  model._seq_host_tf[:G*(2*k+1)] = array.array('i', forced)
  starts = [s.mtp + s.j for s in seqs]
  got = eng.step(seqs)
  assert all(len(g) == k+1 for g in got), "every forced draft accepted"
  lp = np.frombuffer(model._seq_host_tfo, dtype=np.float32, count=G*(k+1)*V).reshape(G, k+1, V)
  for g, (p, text) in enumerate(zip(starts, texts)):
    for r in range(k+1):
      row = lp[g, r]
      top = np.argpartition(-row, TOP)[:TOP]
      top = top[np.argsort(-row[top])]
      res[g]["lp"].append(float(row[text[p+1+r]])); res[g]["top"].append(int(top[0]))
      res[g]["ids"].append(top.tolist()); res[g]["lps"].append(row[top].tolist())
for r in res: print("RESULT " + json.dumps(r), flush=True)
nll = -np.mean([x for r in res for x in r["lp"]])
print(f"positions {sum(len(r['lp']) for r in res)} mean NLL {nll:.4f}", flush=True)
