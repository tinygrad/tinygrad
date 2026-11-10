#!/usr/bin/env python3
"""
GLM_SEQ decode quality against the main path, teacher forced: N corpus prompts of L tokens, then NTOK decode positions decoded by
speculative steps of the main path (forward_spec, _spec_call) and of the sequences sharing the caches (seq_step, one sequence), both forced
on the corpus continuation (a step's input tokens are the corpus', its drafts' rows count while they match it). at every position: the
log prob of the corpus token and the TOP most likely ones (from the verified rows' hidden states, the output head on the host side).
  run:     MODEL=... CORPUS=file GLM_SEQ=1 SEQ_ROWS=... [N=4] [L=4096] [NTOK=256] [TOP=64] [WARMUP=0]
           MAIN_OUT=main.out SEQ_OUT=seq.out python3 extra/glm_seq_quality.py
  compare: python3 extra/qwen_kv_quality.py main.out seq.out   (nll, top-1 agreement, KL(main || seq))
"""
import os, sys, json, time, array
import numpy as np
from tinygrad import Device
from tinygrad.helpers import getenv, Context
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
SEQ = bool(model.config.cache_rows)  # (GLM_SEQ=1 SEQ_ROWS=...: the sequences' path too, else the main path only (e.g. a prefill flag's A/B))
tok = SimpleTokenizer.from_gguf_kv(kv)
N, L, NTOK, TOP, V, k = getenv("N", 4), getenv("L", 4096), getenv("NTOK", 256), getenv("TOP", 64), model.config.vocab_size, model.config.spec_drafts
devs = model.token_embd.weight.device
DBG = getenv("DBG", 0)
# (the sequence's region: after the main path's rows (a multiple of the regions' alignment))
BASE = getenv("BASE", 98304)
devs = devs if isinstance(devs, tuple) else (devs,)
if getenv("WARMUP", 0): model.warmup()
corpus = tok.encode(open(os.environ["CORPUS"]).read()[:getenv("CHARS", 3_000_000)])

def raw_rows(t, a:int, b:int) -> np.ndarray:
  """rows a..b of a (rows, D) float state, read from its buffer (the first device's): a fresh tensor for the head (the head's q8 activation
  cache (amd_glm q8_quantize) is keyed by the input's graph: the same slice of the state read again gives its first quantization)"""
  for d in devs: Device[d].synchronize()
  buf = t.uop.storage_base.buffer
  buf = buf.bufs[0] if hasattr(buf, "bufs") else buf
  return np.frombuffer(bytearray(buf.as_memoryview()), dtype=np.float32).reshape(-1, t.shape[-1])[a:b].copy()

def rec(out:dict, h, target:int):
  from tinygrad import Tensor
  t, a, b = h
  from tinygrad.llm.kernels.amd_glm import q8_quantize_cache_clear
  x = model.output(Tensor(raw_rows(t, a, b)).to(devs)).float().numpy()[:, :V].astype(np.float64)
  q8_quantize_cache_clear()
  for r, t in zip(x, target if isinstance(target, list) else [target]):
    m = r.max()
    lp = r - (m + np.log(np.exp(r - m).sum()))
    best = np.argpartition(-lp, TOP)[:TOP]
    best = best[np.argsort(-lp[best])]
    out["lp"].append(round(float(lp[t]), 5))
    out["top"].append(int(best[0]))
    out["ids"].append([int(b) for b in best])
    out["lps"].append([round(float(lp[b]), 5) for b in best])

def forced(T:list[int], st:int, drafts:list[int]) -> int:
  """the drafts after position st that are the corpus' (its rows count)"""
  return next((i for i in range(k) if st+1+i >= len(T) or drafts[i] != T[st+1+i]), k)

def wait_out(host_out) -> list[int]:
  st = time.perf_counter()
  while min(o:=host_out.tolist()) < 0:
    if time.perf_counter() - st > 60: raise RuntimeError("a step's tokens: none after 60 s")
    time.sleep(0.0005)
  return o

# (QJIT=0: the steps eagerly (not supported by every kernel). the rows are read into fresh tensors (raw_rows) and checked (check))
QJIT = getenv("QJIT", 1)
bad = {"main": 0, "seq": 0, "rows": 0}
def check(name:str, out:dict, n:int, sampled:list[int]):
  """the recorded rows' argmax are the step's sampled tokens (else the rows read aren't the step's)"""
  bad["rows"] += n
  bad[name] += sum(t != s for t, s in zip(out["top"][-n:], sampled[:n]))

res:dict[str, list] = {"main": [], "seq": []}
st0 = time.perf_counter()
for i in range(N):
  off = 1000 + i * (len(corpus) - L - NTOK - 2000) // N
  ids, T = corpus[off:off+L], corpus[off:off+L+NTOK+k+2]  # (T: the corpus from the prompt's start, positions as the model's)
  # ** main path: the prompt prefilled by generate (its first sampled token not taken), then forced steps of forward_spec
  m = {"lp": [], "top": [], "ids": [], "lps": []}
  gen = model.generate(list(ids))
  t0 = next(gen)
  rec(m, (model._spec_h, 0, 1), T[L])
  check("main", m, 1, [t0])
  host, host_out = model._spec_host, model._spec_host_out
  mtp, j, par, y = L, 0, 1, [T[L]] + [0]*k
  while len(m["lp"]) < NTOK + 1:
    host[0:k+1] = array.array('i', y)
    host_out[:] = array.array('i', [-1] * (2*k+1))
    with Context(JIT=QJIT): model._spec_call(mtp, j, par, None)
    out = wait_out(host_out)
    st = mtp + j
    jj = forced(T, st, out[k+1:])
    if DBG: print(f"main st {st} j {j} par {par} y {y} out {out} jj {jj} T {T[st+1:st+2+k]}", file=sys.stderr)
    rec(m, (model._spec_h, par*(k+1), par*(k+1)+jj+1), T[st+1:st+2+jj])
    check("main", m, jj+1, out)
    mtp, j, par, y = st + 1, jj, 1 - par, [T[st+1+t] for t in range(k+1)]
  del gen
  for d in devs: Device[d].synchronize()
  # ** the sequences' path: the prompt prefilled on region BASE (slot 0), then forced seq_steps of it alone
  s = {"lp": [], "top": [], "ids": [], "lps": []}
  if not SEQ:
    res["main"].append({"name": f"p{i}", **{kk: v[:NTOK+1] for kk, v in m.items()}})
    print(f"prompt {i}: {time.perf_counter()-st0:.1f}s  rows not the step's: {bad}", file=sys.stderr, flush=True)
    continue
  H = 2*(k+1)
  first = model.seq_prefill(0, BASE, list(ids), 0, 0)
  rec(s, (model.seq_spec_h, 0, 1), T[L])
  check("seq", s, 1, [first])
  mtp, j, par, y = L, 0, 1, [T[L]] + [0]*k
  model._token_host[BASE+L:BASE+L+1] = array.array('i', [T[L]])
  while len(s["lp"]) < NTOK + 1:
    with Context(JIT=QJIT): sampled, drafts = model.seq_step([(0, BASE, mtp, j, par, y, 0.0)])[0]
    st = mtp + j
    jj = forced(T, st, drafts)
    if DBG: print(f"seq  st {st} j {j} par {par} y {y} out {sampled+drafts} jj {jj} T {T[st+1:st+2+k]}", file=sys.stderr)
    rec(s, (model.seq_spec_h, par*(k+1), par*(k+1)+jj+1), T[st+1:st+2+jj])
    check("seq", s, jj+1, sampled)
    model._token_host[BASE+st+1:BASE+st+2+jj] = array.array('i', T[st+1:st+2+jj])
    mtp, j, par, y = st + 1, jj, 1 - par, [T[st+1+t] for t in range(k+1)]
  for name, r in (("main", m), ("seq", s)): res[name].append({"name": f"p{i}", **{kk: v[:NTOK+1] for kk, v in r.items()}})
  agree = sum(a == b for a, b in zip(m["top"], s["top"]))
  print(f"prompt {i}: {time.perf_counter()-st0:.1f}s  top-1 agree {agree}/{len(m['top'])}  rows not the step's: {bad}", file=sys.stderr, flush=True)
for name, path in (("main", os.environ.get("MAIN_OUT", "main.out")), ("seq", os.environ.get("SEQ_OUT", "seq.out")))[:2 if SEQ else 1]:
  with open(path, "w") as f:
    for r in res[name]: f.write("RESULT " + json.dumps(r) + "\n")
