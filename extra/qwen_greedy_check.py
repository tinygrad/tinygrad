#!/usr/bin/env python3
"""
greedy decoding of the qwen4exp model through SeqEngine (as the server runs it) on scenarios that cover the cache paths: long prompts, batched
steps, continuations from a sequence's end, from its checkpoint, after tokens dropped, from the stash, sliced prefills. prints a JSON line of every
scenario's tokens: two builds' outputs compared (diff) show a change of the caches doesn't change the model.
env: MODEL, SHARD=6, MAXCTX=262144, LENS=4096,65536,262000 (the long prompts), GEN=48 (tokens a reply), ONLY=names, the server's SEQ_* vars
"""
import os, sys, json, time, pathlib
from tinygrad.helpers import getenv, Context
# (WARMUP_JITS: the warm-up makes the memory, the jits are captured on their first call (a run uses a few); WARMUP_JITS=all as a server)
os.environ.setdefault("WARMUP_JITS", "none")
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.engine import SeqEngine

def log(*a): print(*a, file=sys.stderr, flush=True)

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
tok = SimpleTokenizer.from_gguf_kv(kv)
st = time.perf_counter()
with Context(DEBUG=0): model.warmup()
log(f"warmup {time.perf_counter()-st:.1f}s")

eng = SeqEngine(model)

# HANG_WATCH=s: a turn without progress (a prefill, a step) for that long is a hang (it can spin in a host submit without a timeout): the devices'
# queues and collectives reported, the process exits
busy = [time.perf_counter()]
if (HANG_WATCH:=getenv("HANG_WATCH", 0)):
  import threading
  def _watch():
    while time.sleep(2) or time.perf_counter() - busy[0] < HANG_WATCH: pass
    from tinygrad import Device
    from tinygrad.llm.kernels.amd_qwen import comm_state_report
    log(f"HANG_WATCH: no progress for {HANG_WATCH} s")
    for d in model.token_embd.weight.device:
      try: log("hang:", Device[d].iface.hang_report())
      except Exception as e: log(f"hang: {d}: {type(e).__name__}: {e}")
    try: log(comm_state_report())
    except Exception as e: log(f"comm state: {type(e).__name__}: {e}")
    os._exit(3)
  threading.Thread(target=_watch, daemon=True).start()
GEN, ONLY = getenv("GEN", 48), [x for x in os.environ.get("ONLY", "").split(",") if x]
TEMP = (0.0, 1.0)

# the corpus: the repo's python files (text the model continues meaningfully)
# (CORPUS: a file of it, the same prompts for two builds; else this tree's files, made into it)
root = pathlib.Path(__file__).resolve().parents[1]
if (cp:=os.environ.get("CORPUS")) and pathlib.Path(cp).exists(): text = pathlib.Path(cp).read_text()
else:
  text = "\n".join(p.read_text(errors="replace") for p in sorted(root.glob("tinygrad/**/*.py")) + sorted(root.glob("test/**/*.py")))
  if cp: pathlib.Path(cp).write_text(text)
corpus = tok.encode(text)
while len(corpus) < getenv("MAXCTX", 262144): corpus = corpus + corpus
log(f"corpus {len(corpus)} tokens")
def piece(at:int, n:int) -> list[int]: return corpus[at:at+n]

def gen(s, first:int, n:int=GEN) -> list[int]:
  out = [first]
  while len(out) < n:
    busy[0] = time.perf_counter()
    out += eng.step([s])[0]
  return out
def turn(ids:list[int], n:int=GEN, room:int=GEN+16, drop:int=0):
  got = eng.take(ids, room)
  assert got is not None, f"no room for {len(ids)}"
  s, start, restore = got
  busy[0] = time.perf_counter()
  out = gen(s, eng.prefill(s, ids, start, restore, TEMP), n)
  # (the reply's first n tokens kept: the next prompts the same whatever the drafts accepted)
  # (tokens dropped past the last step's accepted ones: its states are of those)
  kept = max(len(ids) + n - drop, len(s.tokens) - s.hrow % (model.config.spec_drafts + 1))
  eng.stop(s, kept)
  return s, out[:kept - len(ids)], start, restore
def reset():
  for s in [s for s in eng.seqs if s.active]: eng.stop(s, len(s.tokens))
  eng.forget()

def emit(name:str, **kw):
  print("RESULT " + json.dumps({"name": name, **kw}), flush=True)
  log(name, {k: (v if not isinstance(v, list) else (v[:12], len(v))) for k, v in kw.items()})

def run(name:str) -> bool: return not ONLY or name in ONLY

# long prompts, a sequence alone
for L in [int(x) for x in os.environ.get("LENS", "4096,65536,262000").split(",") if x]:
  if not run(f"long{L}"): continue
  reset()
  st = time.perf_counter()
  s, out, _, _ = turn(piece(1000, L))
  emit(f"long{L}", tokens=out, secs=round(time.perf_counter()-st, 1))

# turns of a conversation: from its end, after a dropped token, from its checkpoint (another reply), the same prompt again
if run("turns"):
  reset()
  A = piece(5000, 3001)
  s, o1, st1, r1 = turn(A)
  B = s.tokens + piece(20000, 517)                   # continues the sequence from its end
  s, o2, st2, r2 = turn(B, drop=1)                   # the last token dropped (a stop sequence)
  C = s.tokens + piece(30000, 203)
  s, o3, st3, r3 = turn(C)
  D = C + piece(40000, 301)                          # diverges at the end of C's prompt: from the checkpoint
  s, o4, st4, r4 = turn(D)
  s, o5, st5, r5 = turn(list(D))                     # the same prompt again: nothing prefilled
  emit("turns", tokens=[o1, o2, o3, o4, o5], starts=[st1, st2, st3, st4, st5], restores=[r1, r2, r3, r4, r5])

# sequences decoded together (a step of several), different lengths and offsets within a pool
if run("batch"):
  reset()
  seqs, outs = [], []
  for i, L in enumerate([1501, 2302, 4093, 7000, 9001, 513, 3333, 12345]):
    got = eng.take(piece(50000 + 9000*i, L), 4*GEN)
    assert got is not None
    s, start, restore = got
    outs.append([eng.prefill(s, piece(50000 + 9000*i, L), start, restore, TEMP)])
    seqs.append(s)
  for _ in range(GEN):
    for o, t in zip(outs, eng.step(seqs)): o += t
  for s in seqs: eng.stop(s, len(s.tokens))
  emit("batch", tokens=[o[:GEN] for o in outs])

# more conversations than slots: evicted to the stash, continued from there
if run("stash"):
  reset()
  n = model.seq_slots() + 3
  convs, outs = [], []
  for i in range(n):
    s, o, _, _ = turn(piece(100000 + 7000*i, 2000 + 37*i), n=16)
    convs.append(s.tokens)
    outs.append(o)
  starts = []
  for i in range(n):
    s, o, st, _ = turn(convs[i] + piece(150000 + 500*i, 97), n=16)
    outs.append(o)
    starts.append(st)
  emit("stash", tokens=outs, starts=starts, evictions=eng.evictions)

# a sequence's region moved (the regions compacted): the same tokens as not moved
if run("move"):
  outs = []
  for move in (False, True):
    reset()
    got = eng.take(piece(60000, 5001), 4*GEN)
    assert got is not None
    s, start, restore = got
    first = eng.prefill(s, piece(60000, 5001), start, restore, TEMP)
    if move:
      gap = max(b for b, c in eng._gaps(s) if c >= s.cap)
      eng._move(s, gap + (eng.rows - gap - s.cap) // eng.align * eng.align if gap + s.cap <= eng.rows else gap)
    outs.append(gen(s, first)[:GEN])
    eng.stop(s, len(s.tokens))
  emit("move", tokens=outs, same=outs[0] == outs[1])

# a long prompt prefilled in slices (the server's: other sequences' steps between them)
if run("slices"):
  reset()
  ids = piece(7, 20011)
  got = eng.take(ids, 4*GEN)
  assert got is not None
  s, start, restore = got
  for end in (4099, 9001, 15003):
    eng.prefill_slice(s, ids, start, restore, end)
    eng.settle()
    start, restore = end, False
  out = gen(s, eng.prefill(s, ids, start, restore, TEMP))
  eng.stop(s, len(s.tokens))
  emit("slices", tokens=out[:GEN])
print("DONE", flush=True)
