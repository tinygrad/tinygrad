#!/usr/bin/env python3
"""
the quality of a prefill numerics change (e.g. IQ2S_K256=1) for GLM: N prompts of L corpus tokens, the log probs of the token after each
(teacher forced: the next corpus token) and the TOP most likely ones, from the hidden state of the prompt's last token (_spec_h row 0).
  run:     MODEL=... CORPUS=file [N=48] [L=4096] [TOP=64] python3 extra/glm_prefill_quality.py > out
  compare: python3 extra/qwen_kv_quality.py ref.out test.out   (the same RESULT format: nll, top-1 agreement, KL(ref || test))
"""
import os, sys, json, time
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
N, L, TOP, V = getenv("N", 48), getenv("L", 4096), getenv("TOP", 64), model.config.vocab_size
corpus = tok.encode(open(os.environ["CORPUS"]).read()[:getenv("CHARS", 3_000_000)])
out = {"lp": [], "top": [], "ids": [], "lps": []}
st = time.perf_counter()
for i in range(N):
  off = 1000 + i * (len(corpus) - L - 2000) // N
  ids = corpus[off:off + L + 1]
  next(model.generate(list(ids[:L])))
  x = model.output(model._spec_h[0:1]).float().numpy()[0, :V].astype(np.float64)
  m = x.max()
  lp = x - (m + np.log(np.exp(x - m).sum()))
  best = np.argpartition(-lp, TOP)[:TOP]
  best = best[np.argsort(-lp[best])]
  out["lp"].append(round(float(lp[ids[L]]), 5))
  out["top"].append(int(best[0]))
  out["ids"].append([int(t) for t in best])
  out["lps"].append([round(float(lp[t]), 5) for t in best])
  print(f"{i}: {time.perf_counter()-st:.1f}s", file=sys.stderr, flush=True)
print("RESULT " + json.dumps({"name": f"prefill{L}", **out}), flush=True)
