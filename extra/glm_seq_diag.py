#!/usr/bin/env python3
"""
GLM_SEQ diagnosis: one prompt prefilled in NSEQ sequences (slots 0.., regions REGION_ALIGN*8 apart), STEPS steps of all of them at once: the
hidden states of the step's rows of every sequence compared with sequence 0's (the same inputs: a difference is the row's place in the step,
or its slot or region). env: MODEL, SHARD, MAXCTX, CTXP, CORPUS, NSEQ (4), STEPS (4), ORDER (e.g. 3,2,1,0: the sequences' order in a step)
"""
import os, array
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
CTXP, NSEQ = getenv("CTXP", 4096), getenv("NSEQ", 4)
p = tok.encode(open(os.environ["CORPUS"]).read()[1000:1000+CTXP*5])[:CTXP]
k = model.config.spec_drafts
order = [int(x) for x in str(getenv("ORDER", ",".join(str(i) for i in range(NSEQ)))).split(",")]
# (BASES: the regions of the slots, default REGION_ALIGN*8 apart in order)
bases = [int(x) for x in str(getenv("BASES", ",".join(str(i*3072*8) for i in range(NSEQ)))).split(",")]
seqs = []
for i in range(NSEQ):
  t = model.seq_prefill(i, bases[i], list(p), 0, 0)
  seqs.append(dict(slot=i, base=bases[i], mtp=len(p), j=0, par=1, y=[t] + [0]*k))
H = 2*(k+1)
from tinygrad.llm.glm5next import GatedDeltaNetBlock, MLATransformerBlock
def dump(tag:str, upto:int):
  if not getenv("DUMP", 0): return
  for li, b in enumerate(model.blk + [model.mtp]):
    parts = {}
    if isinstance(b, GatedDeltaNetBlock):
      S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
      rec, convs = b.pool_rec.numpy(), [c.numpy() for c in b.pool_conv]
      parts["rec"] = [rec[sl*S:(sl+1)*S] for sl in range(NSEQ)]
      if hasattr(b, "pool_rp"):
        rp = b.pool_rp.numpy()
        Q = rp.shape[0] // len(b.slot_rec)
        parts["rp"] = [rp[sl*Q:(sl+1)*Q] for sl in range(NSEQ)]
      for i, c in enumerate(convs): parts[f"conv{i}"] = [c[sl*SC:(sl+1)*SC] for sl in range(NSEQ)]
    if isinstance(b, MLATransformerBlock):
      if b.indexer is not None:
        R, ring = int(b.indexer.slot_ring[0].shape[0]), b.indexer.pool_ring.numpy()
        parts["ring"] = [ring[sl*R:(sl+1)*R] for sl in range(NSEQ)]
        pk, kp = b.indexer.cache_pk.numpy(), b.indexer.idx.kpool
        parts["pk"] = [pk[bases[sl]//kp:bases[sl]//kp + (len(p)+8)//kp] for sl in range(NSEQ)]
      nd = len(b.cache_s.device)
      cv = b.cache_s.numpy().reshape(nd, -1, b.cache_s.shape[-1])
      def rows_of(base, lo, hi): return [cv[(q//64) % nd, base//nd + (q//(64*nd))*64 + q % 64] for q in range(lo, hi)]
      import numpy as _np
      parts["cache"] = [_np.stack(rows_of(bases[sl], 0, upto)) for sl in range(NSEQ)]
    for name, xs in parts.items():
      def mx(a, b) -> float: return float(np.nan_to_num(np.abs(a.astype(np.float32) - b.astype(np.float32)), nan=1e9).max())
      ds = [mx(xs[sl], xs[0]) for sl in range(1, NSEQ)]
      if max(ds) > 0: print(f"  {tag}: block {li} {type(b).__name__} {name}: max |diff| vs slot 0 {ds}")
  print(f"  {tag}: dump done", flush=True)
dump("after prefill", len(p))
for step in range(getenv("STEPS", 4)):
  live = [seqs[i] for i in order]
  out = model.seq_step([(s["slot"], s["base"], s["mtp"], s["j"], s["par"], s["y"], 0.0) for s in live])
  hs = model.seq_spec_h.numpy()
  # (the step's rows: of parity par)
  rows = {s["slot"]: hs[s["slot"]*H + s["par"]*(k+1):s["slot"]*H + (s["par"]+1)*(k+1)] for s in live}
  ref = rows[0]
  print(f"step {step}: tokens {[o[0] for o in out]} drafts {[o[1] for o in out]}")
  for sl, r in rows.items():
    d = np.abs(r - ref)
    b0 = next(s['base'] for s in live if s['slot'] == sl)
    print(f"  slot {sl} base {b0}: hidden max |diff| vs slot 0 per row {[float(x) for x in d.max(axis=1)]} (|h| max {float(np.abs(ref).max()):.2f})")
  if step == 0: dump("after step 0", len(p) + k + 1)
  for s, (sampled, drafts) in zip(live, out):
    j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
    st = s["mtp"] + s["j"]
    model._token_host[s["base"] + st + 1:s["base"] + st + 2 + j] = array.array('i', sampled[:j+1])
    s["mtp"], s["j"], s["par"], s["y"] = st + 1, j, 1 - s["par"], sampled
# (DUMP=1: every block's states of the slots after a prefill and after a step, compared with slot 0's: the first block that differs)
