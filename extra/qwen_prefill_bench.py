#!/usr/bin/env python3
"""
the time of a warm turn's prefill (a conversation's new tokens after its cached context) by its parts, as the server runs it (SeqEngine): the
whole prefill's wall time and the host time of its calls (the state copies, the chunks' jits, the checkpoint, the wait for the token).
env: MODEL, CTX=50000 (the cached context), NEWS=32,64,108,192,384,576,1024,2048,4096 (new tokens a turn), REPS=5, plus the server's SEQ_*
PROFILE=1 VIZ=0: the device timeline too (one rep of every size, between profile markers)
"""
import os, sys, time, random, json, collections, statistics
from tinygrad import Device
from tinygrad.helpers import getenv, Context
from tinygrad.llm.model import Transformer
from tinygrad.llm.engine import SeqEngine

def log(*a): print(*a, file=sys.stderr, flush=True)

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
st = time.perf_counter()
with Context(DEBUG=0): model.warmup()
log(f"warmup {time.perf_counter()-st:.1f}s, chunks {model.prefill_chunks()}")
eng = SeqEngine(model)
devs = tuple(model.token_embd.weight.device)
def sync():
  for d in devs: Device[d].synchronize()

# the host time of the model's calls of a prefill
spent: collections.Counter = collections.Counter()
calls: collections.Counter = collections.Counter()
def timed(name:str):
  fn = getattr(model, name)
  def wrap(*a, **kw):
    st = time.perf_counter()
    try: return fn(*a, **kw)
    finally:
      spent[name] += time.perf_counter() - st
      calls[name] += 1
  setattr(model, name, wrap)
for name in ("_seq_copy_call", "_seq_chunk", "seq_ckpt", "_wait_host", "_chunk_plan"): timed(name)

rng = random.Random(0)
CTX, REPS = getenv("CTX", 50000), getenv("REPS", 5)
ids = [rng.randrange(1000, 100000) for _ in range(CTX)]
s, start, restore = eng.take(ids, 70000)
eng.prefill(s, ids, start, restore, (0.0, 1.0))
sync()
for n in [int(x) for x in os.environ.get("NEWS", "32,64,108,192,384,576,1024,2048,4096").split(",")]:
  times, parts = [], collections.Counter()
  for rep in range(REPS):
    # (the conversation's next turn: its tokens so far and n new ones, prefilled from where it is)
    new = s.tokens + [rng.randrange(1000, 100000) for _ in range(n)]
    eng.stop(s, len(s.tokens))
    got = eng.take(new, 64)
    assert got is not None
    s, start, restore = got
    spent.clear()
    calls.clear()
    sync()
    if getenv("PROFILE") and rep == REPS - 1:
      from tinygrad.helpers import profile_marker
      profile_marker(f"pf{n}")
    st = time.perf_counter()
    eng.prefill(s, new, start, restore, (0.0, 1.0))
    times.append(time.perf_counter() - st)
    parts += spent
  med = statistics.median(times)
  r = {"new": n, "start": start, "restore": restore, "ms_med": round(med*1e3, 1), "ms_min": round(min(times)*1e3, 1),
       "tok_s": round(n / med), "host_ms": {k: round(v / REPS * 1e3, 1) for k, v in parts.items()}, "calls": dict(calls)}
  print("PREFILL " + json.dumps(r), flush=True)
  log(r)
