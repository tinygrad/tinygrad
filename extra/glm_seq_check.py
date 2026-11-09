#!/usr/bin/env python3
"""
GLM sequences sharing the caches (GLM_SEQ=1 SEQ_ROWS=n) against the main path (greedy, NTOK tokens), as SeqEngine drives them:
  1. NSEQ prompts prefilled in slots 0.. (regions REGION_ALIGN*8 apart), their checkpoints saved, all decoded at once (seq_step)
  2. sequence 0 alone: its checkpoint restored, the token after it from its hidden state (seq_first), decoded alone
  3. all of them again from their checkpoints, decoded at once
env: MODEL, SHARD (6), MAXCTX (65536), CTXP (4096 corpus tokens), CORPUS, NSEQ (2), NTOK (24), OFF (512: the prompts' corpus offsets apart)
"""
import os, time, array
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
N, CTXP, OFF, NSEQ = getenv("NTOK", 24), getenv("CTXP", 4096), getenv("OFF", 512), getenv("NSEQ", 2)
corpus = tok.encode(open(os.environ["CORPUS"]).read()[1000:1000+(CTXP+OFF*NSEQ)*5])
# (SAME=1: every sequence the first prompt: the rows of a step's sequences the same inputs, their tokens must be the same)
prompts = [corpus[(0 if getenv("SAME", 0) else i)*OFF:(0 if getenv("SAME", 0) else i)*OFF+CTXP] for i in range(NSEQ)]
refs = []
for p in prompts[:1] if getenv("SAME", 0) else prompts:
  st, out = time.perf_counter(), []
  for t in model.generate(list(p)):
    out.append(t)
    if len(out) == N: break
  refs.append(out)
  print(f"main path: {out[:8]}... ({time.perf_counter()-st:.1f} s)", flush=True)
if getenv("SAME", 0): refs = refs * NSEQ
k, ok = model.config.spec_drafts, True

def start(slot:int, base:int, p:list[int], first:int) -> dict:
  # (as SeqEngine.prefill: the MTP layer from the prompt's end, the hidden state of its last token row 0 of parity 0)
  return dict(slot=slot, base=base, mtp=len(p), j=0, par=1, y=[first] + [0]*k, toks=[first])

def decode(seqs:list[dict]) -> int:
  steps = 0
  while min(len(s["toks"]) for s in seqs) < N:
    out = model.seq_step([(s["slot"], s["base"], s["mtp"], s["j"], s["par"], s["y"], 0.0) for s in seqs])
    steps += 1
    for s, (sampled, drafts) in zip(seqs, out):
      j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
      st = s["mtp"] + s["j"]
      model._token_host[s["base"] + st + 1:s["base"] + st + 2 + j] = array.array('i', sampled[:j+1])
      s["mtp"], s["j"], s["par"], s["y"] = st + 1, j, 1 - s["par"], sampled
      s["toks"] += sampled[:j+1]
  return steps

def compare(name:str, seqs:list[dict], idxs:list[int]) -> bool:
  good = True
  for s, i in zip(seqs, idxs):
    same = s["toks"][:N] == refs[i]
    good &= same
    print(f"{name}: seq {i} slot {s['slot']} base {s['base']}: {'OK' if same else 'DIFFERS'}" +
          ("" if same else f"\n  seq  {s['toks'][:N]}\n  main {refs[i]}"), flush=True)
  return good

bases = [i * 3072 * 8 for i in range(NSEQ)]
seqs = []
for i, p in enumerate(prompts):
  st = time.perf_counter()
  t = model.seq_prefill(i, bases[i], list(p), 0, 0)
  model.seq_ckpt(True, i, len(p))
  print(f"prefill slot {i} base {bases[i]}: first token {t} ({time.perf_counter()-st:.1f} s) {'OK' if t == refs[i][0] else 'DIFFERS'}", flush=True)
  ok &= t == refs[i][0]
  seqs.append(start(i, bases[i], p, t))
st = time.perf_counter()
steps = decode(seqs)
print(f"1. {steps} steps of {NSEQ} sequences, {sum(len(s['toks']) for s in seqs)/steps:.2f} tokens a step ({time.perf_counter()-st:.1f} s)")
ok &= compare("1. together", seqs, list(range(NSEQ)))
# 2. sequence 0 from its checkpoint, alone
model.seq_ckpt(False, 0, len(prompts[0]))
t = model.seq_first(0, bases[0], len(prompts[0]))
s0 = start(0, bases[0], prompts[0], t)
decode([s0])
ok &= compare("2. restored alone", [s0], [0])
# 3. all from their checkpoints, together
seqs = []
for i, p in enumerate(prompts):
  model.seq_ckpt(False, i, len(p))
  seqs.append(start(i, bases[i], p, model.seq_first(i, bases[i], len(p))))
decode(seqs)
ok &= compare("3. restored together", seqs, list(range(NSEQ)))
# 4. from their checkpoints, joining one after the other (a step of 1, 2, 3 (padded to 4), .. sequences: as requests come to a server)
seqs, live = [], []
for i, p in enumerate(prompts):
  model.seq_ckpt(False, i, len(p))
  seqs.append(start(i, bases[i], p, model.seq_first(i, bases[i], len(p))))
for i in range(NSEQ):
  live.append(seqs[i])
  for _ in range(2 if i < NSEQ-1 else 0):
    out = model.seq_step([(s["slot"], s["base"], s["mtp"], s["j"], s["par"], s["y"], 0.0) for s in live])
    for s, (sampled, drafts) in zip(live, out):
      j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
      st = s["mtp"] + s["j"]
      model._token_host[s["base"] + st + 1:s["base"] + st + 2 + j] = array.array('i', sampled[:j+1])
      s["mtp"], s["j"], s["par"], s["y"] = st + 1, j, 1 - s["par"], sampled
      s["toks"] += sampled[:j+1]
decode(live)
ok &= compare("4. joining", seqs, list(range(NSEQ)))
print("ALL OK" if ok else "FAILED")
