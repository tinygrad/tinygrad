#!/usr/bin/env python3
"""
the quality of a format of the attention's cache (QSA_KV_BITS, QSA_KV_ROT): the model's log probs of the next token of a fixed text (teacher
forced, a token a prefill from the sequence's states) at a few contexts. prints a RESULT line a context: the target's log prob, the most likely
token and the TOP most likely ones (ids, log probs) of every position.
  run:     MODEL=... CORPUS=file [CTXS=2000,16000,64000,200000] [M=256] [TOP=64] python3 extra/qwen_kv_quality.py > out
  compare: python3 extra/qwen_kv_quality.py ref.out test.out  (the mean NLL, top-1 agreement and KL(ref || test) on the ref's top tokens)
"""
import os, sys, json, math, time, pathlib

def load(p:str) -> dict[str, dict]:
  return {d["name"]: d for d in (json.loads(l[7:]) for l in open(p) if l.startswith("RESULT "))}

def compare(ref:str, test:str):
  a, b = load(ref), load(test)
  tot = {"n": 0, "nll_a": 0.0, "nll_b": 0.0, "agree": 0, "kl": 0.0, "kl_max": 0.0}
  for k in a:
    if k not in b:
      print(k, "missing")
      continue
    x, y = a[k], b[k]
    n = min(len(x["lp"]), len(y["lp"]))
    agree, kl, klmax = 0, 0.0, 0.0
    for i in range(n):
      agree += x["top"][i] == y["top"][i]
      # KL(ref || test) on the ref's top tokens (a token not in the test's top: its log prob at most the test's last one there)
      ty = dict(zip(y["ids"][i], y["lps"][i]))
      floor = y["lps"][i][-1]
      ki = sum(math.exp(la) * (la - ty.get(t, floor)) for t, la in zip(x["ids"][i], x["lps"][i]))
      kl, klmax = kl + ki, max(klmax, ki)
    nll_a, nll_b = -sum(x["lp"][:n]) / n, -sum(y["lp"][:n]) / n
    print(f"{k:>10}: n {n:4d}  nll {nll_a:.4f} -> {nll_b:.4f} ({nll_b - nll_a:+.4f})  top-1 agree {agree/n:6.2%}  KL mean {kl/n:.5f} max {klmax:.4f}")
    for key, val in (("n", n), ("nll_a", nll_a * n), ("nll_b", nll_b * n), ("agree", agree), ("kl", kl)): tot[key] += val
    tot["kl_max"] = max(tot["kl_max"], klmax)
  n = tot["n"]
  if n: print(f"{'all':>10}: n {n:4d}  nll {tot['nll_a']/n:.4f} -> {tot['nll_b']/n:.4f} ({(tot['nll_b']-tot['nll_a'])/n:+.4f})  "
              f"ppl {math.exp(tot['nll_a']/n):.4f} -> {math.exp(tot['nll_b']/n):.4f}  top-1 agree {tot['agree']/n:6.2%}  KL mean {tot['kl']/n:.5f} "
              f"max {tot['kl_max']:.4f}")

if __name__ == "__main__" and len(sys.argv) == 3:
  compare(sys.argv[1], sys.argv[2])
  sys.exit(0)

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv, Context
# (WARMUP_JITS: the warm-up makes the memory, the jits are captured on their first call (a run uses a few); WARMUP_JITS=all as a server)
os.environ.setdefault("WARMUP_JITS", "none")
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.engine import SeqEngine

def log(*a): print(*a, file=sys.stderr, flush=True)

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
# (the logits of a prefill's last token: QSA_LP's buffer, before the jits are captured)
devs = model.token_embd.weight.device
model._lp_buf = Tensor.zeros(1, model.config.padded_vocab_size or model.config.vocab_size).shard(devs, 1).contiguous().realize()
st = time.perf_counter()
with Context(DEBUG=0): model.warmup()
log(f"warmup {time.perf_counter()-st:.1f}s")
eng = SeqEngine(model)
V, M, TOP = model.config.vocab_size, getenv("M", 256), getenv("TOP", 64)

corpus = tok.encode(pathlib.Path(os.environ["CORPUS"]).read_text()[:getenv("CHARS", 2_000_000)])
log(f"corpus {len(corpus)} tokens")

def logprobs() -> np.ndarray:
  x = model._lp_buf.numpy()[0, :V].astype(np.float64)
  m = x.max()
  return x - (m + np.log(np.exp(x - m).sum()))

for i, L0 in enumerate(int(x) for x in os.environ.get("CTXS", "2000,16000,64000,200000").split(",")):
  for s in [s for s in eng.seqs if s.active]: eng.stop(s, len(s.tokens))
  eng.forget()
  off = 1000 + 37000 * i
  ids = (corpus * (1 + (off + L0 + M + 1) // len(corpus)))[off:off + L0 + M + 1]
  st = time.perf_counter()
  got = eng.take(ids[:L0], M + 64)
  assert got is not None, f"no room for {L0}"
  s, start, restore = got
  eng.prefill(s, ids[:L0], start, restore, (0.0, 1.0))
  out = {"lp": [], "top": [], "ids": [], "lps": []}
  for t in range(L0, L0 + M):
    if t > L0: model.seq_prefill(s.slot, s.base, ids[:t], t - 1, 0)
    lp = logprobs()
    best = np.argpartition(-lp, TOP)[:TOP]
    best = best[np.argsort(-lp[best])]
    out["lp"].append(round(float(lp[ids[t]]), 5))
    out["top"].append(int(best[0]))
    out["ids"].append([int(x) for x in best])
    out["lps"].append([round(float(lp[x]), 5) for x in best])
  s.tokens, s.n = ids[:L0 + M - 1], L0 + M - 1
  name = f"ctx{L0}"
  print("RESULT " + json.dumps({"name": name, **out}), flush=True)
  log(f"{name}: nll {-sum(out['lp'])/M:.4f} ({time.perf_counter()-st:.1f}s)")
