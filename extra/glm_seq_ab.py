#!/usr/bin/env python3
"""
GLM_SEQ against the main path on a server's prompt (the chat template's, as tools/glm_greedy_check.py's 4k one): generate's greedy tokens,
then the prompt prefilled on a region (seq_prefill, base BASE) and decoded alone (seq_step): the hidden state after the prompt and the tokens
compared. env: MODEL, SHARD (6), MAXCTX (262144), CORPUS, CHARS (13464), OFFSET (1000), NTOK (48), BASE (0)
"""
import os, json, time, array
import jinja2
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
env.globals['eos_token'] = tok.decode([tok.eos_id])
off, chars = getenv("OFFSET", 1000), getenv("CHARS", 13464)
text = open(os.environ["CORPUS"]).read()[off:off+chars]
msgs = [{"role": "user", "content": "Continue this source file exactly where it stops, no commentary:\n\n" + text}]
ids = tok.encode(env.from_string(tok.chat_template).render(messages=msgs, tools=None, add_generation_prompt=True, preserve_thinking=True))
N, k = getenv("NTOK", 48), model.config.spec_drafts
# (WARMUP=1: the jits made as a server's (model.warmup) first)
if getenv("WARMUP", 0): model.warmup()
# (ZERO=spec,mainspec,states,rp,rings,ckpt,caches after the warmup: the slots' state of that kind zeroed again: which leftovers change it)
if (zero:=str(getenv("ZERO", ""))):
  from tinygrad.llm.glm5next import GatedDeltaNetBlock, MLATransformerBlock
  zs:list = []
  kinds = set(zero.split(","))
  if "spec" in kinds: zs += [model.seq_spec_h]
  # (mainspec: the main path's hidden rows of speculative decoding (and its slots'): a step reads rows 1..k of the last one's)
  if "mainspec" in kinds: zs += [t for n in ("_spec_h", "pool_spec_h") if (t:=getattr(model, n, None)) is not None]
  for b in model.blk + [model.mtp]:
    if isinstance(b, GatedDeltaNetBlock):
      if "states" in kinds: zs += [b.pool_rec, *b.pool_conv]
      if "rp" in kinds and hasattr(b, "pool_rp"): zs.append(b.pool_rp)
      if "main" in kinds: zs += [b.recurrent_state, *b.conv_states]
    if isinstance(b, MLATransformerBlock) and b.indexer is not None and "rings" in kinds: zs += [b.indexer.pool_ring, b.indexer.ring]
  if "ckpt" in kinds: zs += [model.seq_ckpt_h] + [t for c in model.seq_ckpt_states for t in c]
  from tinygrad import Tensor
  Tensor.realize(*[t.assign(t.zeros_like()) for t in zs])
  print(f"zeroed {len(zs)} buffers ({zero})", flush=True)
# (FILL=x: the attention caches' rows (MLA latent rows, pooled keys, rings) filled with x first: tokens that change read unwritten rows)
if (fill:=getenv("FILL", "")) != "":
  from tinygrad import Tensor
  from tinygrad.llm.glm5next import MLATransformerBlock
  model._init_states()
  ts = []
  for b in model.blk + [model.mtp]:
    if isinstance(b, MLATransformerBlock):
      ts += [b.cache_s] + ([b.indexer.cache_pk, b.indexer.ring] if b.indexer is not None else [])
  import numpy as np
  from tinygrad import dtypes
  # (as the bits of a half: integer ops keep a split tensor's split, no NaN of the old values)
  bits = int(np.array([float(fill)], np.float16).view(np.uint16)[0])
  Tensor.realize(*[t.assign((t.bitcast(dtypes.uint16) * 0 + bits).bitcast(dtypes.half)) for t in ts if t.dtype == dtypes.half])
  print(f"filled {len(ts)} caches with {fill}", flush=True)
print(f"prompt {len(ids)} tokens", flush=True)
main = []
for t in model.generate(list(ids)):
  main.append(t)
  if len(main) == N: break
print(f"main: {main}", flush=True)
if not model.config.cache_rows: raise SystemExit(0)
base = getenv("BASE", 0)
first = model.seq_prefill(0, base, list(ids), 0, 0)
H = 2*(k+1)
s = dict(mtp=len(ids), j=0, par=1, y=[first] + [0]*k)
toks = [first]
while len(toks) < N:
  sampled, drafts = model.seq_step([(0, base, s["mtp"], s["j"], s["par"], s["y"], 0.0)])[0]
  j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
  st = s["mtp"] + s["j"]
  model._token_host[base + st + 1:base + st + 2 + j] = array.array('i', sampled[:j+1])
  s["mtp"], s["j"], s["par"], s["y"] = st + 1, j, 1 - s["par"], sampled
  toks += sampled[:j+1]
print(f"seq:  {toks[:N]}", flush=True)
d = next((i for i, (a, b) in enumerate(zip(main, toks)) if a != b), None)
print("EQUAL" if d is None else f"DIFFER at token {d}: main {tok.decode(main[:d+1])!r} seq {tok.decode(toks[:d+1])!r}")
