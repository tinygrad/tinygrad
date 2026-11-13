#!/usr/bin/env python3
"""
GLM-5.3 Flash (glm5next, GLM_SEQ): does a sequence's decode step compute the same function alone and in a step of B sequences?
One prompt (CORPUS from OFFSET, L tokens) prefilled into B slots (copies: the same inputs), then STEPS speculative steps of all B at once
(seq_step, teacher forced as extra/glm_seq_quality.py: the corpus' tokens written, a row counts while the model's drafts are the corpus').
For every B of BATCHES (a fresh prefill each) the step rows' hidden states of every slot are saved (OUT_<B>.npy, plus the prefill's last row),
and compared: slot s vs slot 0 inside a step (a row's place), and slot 0 vs B=1 (the step's size), for the steps whose drafts agree.
Rows a step = B*(k+1) (padded to the next SEQ_SIZES). One card: SHARD=1 NUM_BLOCKS=6 NUM_BLOCKS_MTP=1 (the MTP layer kept: drafts as deployed),
the sparse attention's heads in groups of 16 (as extra/glm_prefill_chunk.py).
env: MODEL, CORPUS, SHARD (1), MAXCTX (65536), L (2048), OFFSET (1000), STEPS (12), BATCHES (1,2,4,18), OUT (bn), PREFILL_ONLY (0)
"""
import os, array, json
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import getenv
import tinygrad.llm.glm5next as g5
from tinygrad.llm.glm5next import Transformer
from tinygrad.llm.cli import SimpleTokenizer

_original_sparse = g5.flash_attention_mla_sparse
def _sparse16(q:Tensor, cache:Tensor, index:Tensor, scale:float, valid_len, grouped=None):
  if q.shape[1] <= 16 or isinstance(q.device, tuple): return _original_sparse(q, cache, index, scale, valid_len, grouped)
  return Tensor.cat(*[_original_sparse(q[:, a:min(a+16, q.shape[1])], cache, index, scale, valid_len, grouped)
                      for a in range(0, q.shape[1], 16)], dim=1)
g5.flash_attention_mla_sparse = _sparse16

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 1))
tok = SimpleTokenizer.from_gguf_kv(kv)
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
k, L, STEPS, OFF = model.config.spec_drafts, getenv("L", 2048), getenv("STEPS", 12), getenv("OFFSET", 1000)
BATCHES = [int(b) for b in str(getenv("BATCHES", "1,2,4,18")).split(",")]
text = open(os.environ["CORPUS"]).read()
def prompt(off:int) -> list[int]: return tok.encode(text[off:off + (L + 64*STEPS) * 8])[:L + (k+1)*STEPS + 8]
T = prompt(OFF)
# (DIFF=1: the other slots get other prompts (OFFSET + g*DIFF_STRIDE chars): a step mixing sequences as served; slot 0 compared across B)
TS = [T] + [prompt(OFF + g*getenv("DIFF_STRIDE", 200000)) if getenv("DIFF", 0) else T for g in range(1, max(BATCHES))]
RA, H = model.REGION_ALIGN, 2*(k+1)
span = -(-(L + (k+1)*(STEPS+2) + 64) // RA) * RA
OUT = os.environ.get("OUT", "bn")
print(f"blocks {len(model.blk)} drafts {k} L {L} steps {STEPS} batches {BATCHES} sizes {model.SEQ_SIZES} span {span}", flush=True)

def hrows() -> np.ndarray:
  for d in devs: Device[d].synchronize()
  return model.seq_spec_h.numpy().reshape(-1, model.seq_spec_h.shape[-1]).astype(np.float32)

def forced(st:int, drafts:list[int], T:list[int]=T) -> int: return next((i for i in range(k) if drafts[i] != T[st+1+i]), k)

from tinygrad.llm.glm5next import GatedDeltaNetBlock, MLATransformerBlock
def states(base:int, g:int, lo:int, hi:int) -> dict[str, np.ndarray]:
  """(DUMP=1) slot g's states after a step: every block's recurrent state (and its replay rows), conv states, the cache rows lo..hi"""
  out = {}
  for li, b in enumerate(model.blk + [model.mtp]):
    if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "pool_rec"):
      def slot_of(t:Tensor) -> np.ndarray:
        S = int(t.shape[0]) // len(b.slot_rec)
        return t.numpy()[g*S:(g+1)*S].astype(np.float32)
      out[f"{li:02d}_rec"] = slot_of(b.pool_rec)
      if hasattr(b, "pool_rp"): out[f"{li:02d}_rp"] = slot_of(b.pool_rp)
      for i, c in enumerate(b.pool_conv): out[f"{li:02d}_conv{i}"] = slot_of(c)
    if isinstance(b, MLATransformerBlock) and hasattr(b, "cache_s") and b.cache_s.dtype != dtypes.uint32:
      nd, R = len(b.cache_s.device), int(b.cache_s.shape[0]) // len(b.cache_s.device)
      gr = [((q//64) % nd)*R + base//nd + (q//(64*nd))*64 + q % 64 for q in range(lo, hi)]  # (the rows of positions lo..hi, see seq_store)
      cv = b.cache_s.numpy()
      out[f"{li:02d}_cache"] = cv[gr].astype(np.float32)
  return out

res = {}
for B in BATCHES:
  assert (B-1)*getenv("SLOT_STRIDE", 1) < model.seq_slots() and B * span <= model.seq_rows(), f"SEQ_SLOTS >= {B}, SEQ_ROWS >= {B*span}"
  seqs, pre = [], []
  for g in range(B):
    Tg = TS[g]
    # (SLOT_STRIDE=s: sequence g in state slot g*s (default 1): with s = span // REGION_ALIGN a slot's index equals its region slot)
    sl = g * getenv("SLOT_STRIDE", 1)
    first = model.seq_prefill(sl, g*span, list(Tg[:L]), 0, 0)
    pre.append(hrows()[sl*H])  # (the slot's prefill writes its row 0: the hidden state of the prompt's last token)
    model._token_host[g*span+L:g*span+L+1] = array.array('i', [Tg[L]])
    seqs.append(dict(g=sl, base=g*span, mtp=L, j=0, par=1, y=[Tg[L]] + [0]*k, T=Tg))
  steps = []
  if not getenv("PREFILL_ONLY"):
    for s in range(STEPS):
      outs = model.seq_step([(q["g"], q["base"], q["mtp"], q["j"], q["par"], q["y"], 0.0) for q in seqs])
      h = hrows()
      rows = np.stack([h[q["g"]*H + q["par"]*(k+1): q["g"]*H + (q["par"]+1)*(k+1)] for q in seqs])  # (B, k+1, D)
      q0, (sampled, drafts) = seqs[0], outs[0]
      st = q0["mtp"] + q0["j"]
      jj = forced(st, drafts)
      steps.append(dict(st=st, jj=jj, drafts=list(drafts), sampled=list(sampled), rows=rows))
      if s == 0 and getenv("DUMP", 0):
        try: np.savez(f"{OUT}_{B}_dump0.npz", **states(0, 0, L - 4, L + k + 1))
        except Exception as e: print(f"DUMP failed: {e!r}", flush=True)
      for q, (sm, dr) in zip(seqs, outs):
        stq = q["mtp"] + q["j"]
        Tq = q["T"]
        jq = forced(stq, dr, Tq)
        model._token_host[q["base"]+stq+1:q["base"]+stq+2+jq] = array.array('i', Tq[stq+1:stq+2+jq])
        q["mtp"], q["j"], q["par"], q["y"] = stq + 1, jq, 1 - q["par"], [Tq[stq+1+t] for t in range(k+1)]
  res[B] = dict(pre=np.stack(pre), steps=steps)
  np.save(f"{OUT}_{B}_pre.npy", res[B]["pre"])
  if steps: np.save(f"{OUT}_{B}_rows.npy", np.stack([x["rows"] for x in steps]))
  with open(f"{OUT}_{B}_meta.json", "w") as f: json.dump([{kk: v for kk, v in x.items() if kk != "rows"} for x in steps], f)
  # (inside the step: every slot against slot 0 (the same inputs))
  inside = max((float(np.abs(x["rows"] - x["rows"][0:1]).max()) for x in steps), default=0.0)
  pin = float(np.abs(res[B]["pre"] - res[B]["pre"][0:1]).max())
  print(f"B={B}: prefill rows max|slot-slot0| {pin:.3g}; steps max|slot-slot0| {inside:.3g}; st/jj {[(x['st'], x['jj']) for x in steps]}",
        flush=True)

def rel(a:np.ndarray, b:np.ndarray) -> tuple[float, float]:
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)), float(np.abs(a - b).max())
ref = res[BATCHES[0]]
for B in BATCHES[1:]:
  r = res[B]
  if getenv("DUMP", 0) and os.path.exists(f"{OUT}_{B}_dump0.npz") and os.path.exists(f"{OUT}_{BATCHES[0]}_dump0.npz"):
    a, b = np.load(f"{OUT}_{B}_dump0.npz"), np.load(f"{OUT}_{BATCHES[0]}_dump0.npz")
    diffs = [(n, *rel(a[n], b[n])) for n in sorted(a.files)]
    print(f"B={B} vs B={BATCHES[0]} states after step 0 (name, rel L2, max): " +
          " ".join(f"{n}:{x:.2e}/{y:.2g}" for n, x, y in diffs), flush=True)
  print(f"B={B} vs B={BATCHES[0]}: prefill last row rel L2 / max {rel(r['pre'][0], ref['pre'][0])}", flush=True)
  for s, (x, y) in enumerate(zip(r["steps"], ref["steps"])):
    same = x["st"] == y["st"] and x["drafts"][:x["jj"]] == y["drafts"][:y["jj"]]
    n = min(x["jj"], y["jj"]) + 1
    if not same:
      print(f"  step {s}: structure differs (st {x['st']}/{y['st']} jj {x['jj']}/{y['jj']}): stop", flush=True)
      break
    per = [rel(x["rows"][0, i], y["rows"][0, i]) for i in range(n)]
    print(f"  step {s} st {x['st']}: rows 0..{n-1} rel L2 {[f'{a:.2e}' for a, _ in per]} max {[f'{b:.3g}' for _, b in per]} "
          f"tokens {'same' if x['sampled'][:n] == y['sampled'][:n] else 'DIFFER'}", flush=True)
