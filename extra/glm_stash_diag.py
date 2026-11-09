#!/usr/bin/env python3
"""
GLM stash (GLM_SEQ=1 SEQ_STASH_GB>0) on the devices: a prompt prefilled in slot 0 (region 0), stashed (seq_stash save, entry 0), restored
into slot 1 at another region (BASE): every MLA block's latent rows / pooled keys of the prompt and slot 1's states compared with the
originals; then the token after it (seq_first after its checkpoint) and NTOK greedy tokens of slot 1 vs slot 0's.
env: MODEL, SHARD (6), MAXCTX (65536), CTXP (4096), CORPUS, BASE (3072*16), NTOK (24)
"""
import os, array
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.glm5next import MLATransformerBlock, GatedDeltaNetBlock

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
CTXP, B, N = getenv("CTXP", 4096), getenv("BASE", 3072*16), getenv("NTOK", 24)
p = tok.encode(open(os.environ["CORPUS"]).read()[1000:1000+CTXP*5])[:CTXP]
k = model.config.spec_drafts
first = model.seq_prefill(0, 0, list(p), 0, 0)
model.seq_ckpt(True, 0, len(p))
print(f"stash: {model.stash_chunks} chunks, {model.stash_entries} entries", flush=True)
C = model.SEQ_MOVE
nch = -(-len(p) // C)
mla = [b for b in model.blk + [model.mtp] if isinstance(b, MLATransformerBlock)]
def region(b:MLATransformerBlock, base:int) -> tuple[np.ndarray, np.ndarray]:
  nd = len(b.cache_s.device)
  cv = b.cache_s.numpy().reshape(nd, -1, b.cache_s.shape[-1])
  lat = np.stack([cv[(q//64) % nd, base//nd + (q//(64*nd))*64 + q % 64] for q in range(len(p))])
  kp = b.indexer.idx.kpool
  pk = b.indexer.cache_pk.numpy()[base//kp:(base + len(p))//kp]
  return lat, pk
before = [region(b, 0) for b in mla[:3]]
model.seq_stash(True, 0, 0, [(0, 0, nch)], 0, len(p), 0)
model.seq_stash(False, 1, B, [(0, 0, nch)], 0, len(p), 0)
after = [region(b, B) for b in mla[:3]]
for i, ((l0, k0), (l1, k1)) in enumerate(zip(before, after)):
  print(f"MLA block {i}: latent max |diff| {float(np.abs(l0.astype(np.float32) - l1.astype(np.float32)).max())}, "
        f"pooled keys {float(np.abs(k0.astype(np.float32) - k1.astype(np.float32)).max())}", flush=True)
gdn = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)][0]
S = int(gdn.slot_rec[0].shape[0])
rec = gdn.pool_rec.numpy()
print(f"KDA block 0 rec state slot1 vs slot0 max |diff| {float(np.abs(rec[S:2*S] - rec[0:S]).max())}", flush=True)
def decode(slot:int, base:int, t:int) -> list[int]:
  s, toks = dict(mtp=len(p), j=0, par=1, y=[t] + [0]*k), [t]
  while len(toks) < N:
    sampled, drafts = model.seq_step([(slot, base, s["mtp"], s["j"], s["par"], s["y"], 0.0)])[0]
    j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
    st = s["mtp"] + s["j"]
    model._token_host[base + st + 1:base + st + 2 + j] = array.array('i', sampled[:j+1])
    s["mtp"], s["j"], s["par"], s["y"] = st + 1, j, 1 - s["par"], sampled
    toks += sampled[:j+1]
  return toks[:N]
model._token_host[B:B+len(p)] = array.array('i', p)
model.seq_ckpt(False, 1, len(p))
a = decode(1, B, model.seq_first(1, B, len(p)))
model.seq_ckpt(False, 0, len(p))
b = decode(0, 0, model.seq_first(0, 0, len(p)))
print(f"restored {a}\noriginal {b}\n{'EQUAL' if a == b else 'DIFFER'}")
