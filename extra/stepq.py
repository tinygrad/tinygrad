# quick decode step time: one sequence, random tokens, no warmup of the other jits (the step's own jit captured at its first call)
import os, time, random
from tinygrad import Device
from tinygrad.helpers import profile_marker
from tinygrad.llm.model import Transformer
fn = os.environ.get('QWEN_GGUF', '/raid/weights/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf')
t0 = time.perf_counter()
model,kv=Transformer.from_gguf(fn, 262144, shard=6)
model._finish_load()
devs = model.token_embd.weight.device
def sync():
  for d in devs: Device[d].synchronize()
k = model.config.spec_drafts
random.seed(0)
G, C = int(os.environ.get("G", 1)), int(os.environ.get("CTXP", 4000))
seqs = [(g, g*int(os.environ.get("GSTRIDE", 8192)), C, 0, 1, [random.randrange(1000, 50000) for _ in range(k+1)], 0.0) for g in range(G)]
outs = [model.seq_step(seqs) for _ in range(3)]
# (the steps' sampled tokens and drafts (temperature 0: greedy) hashed: two builds' steps compared)
import hashlib
print(f"ready {time.perf_counter()-t0:.1f}s outhash {hashlib.md5(repr(outs).encode()).hexdigest()[:12]}", flush=True)
if os.environ.get("METRICS"):
  import threading
  from tinygrad.runtime.ops_amd import AMIdleClocks
  def mon():
    time.sleep(0.5)
    for _ in range(2):
      print("metrics", [(m.get("power"), round(m.get("gfxclk", 0)), round(m.get("uclk", 0))) for m in AMIdleClocks.metrics()], flush=True)
      time.sleep(0.3)
  threading.Thread(target=mon, daemon=True).start()
NS = int(os.environ.get("NSTEPS", 100))
if os.environ.get("GSEQ"):
  gs = [int(g) for g in os.environ["GSEQ"].split(",")]
  allseqs = [(g, g*int(os.environ.get("GSTRIDE", 8192)), C, 0, 1, [random.randrange(1000, 50000) for _ in range(k+1)], 0.0) for g in range(max(gs))]
  for g in gs:
    t1 = time.perf_counter()
    model.seq_step(allseqs[:g])
    sync()
    print(f"  size {g}: {time.perf_counter()-t1:.1f}s", flush=True)
  sync()
  print("captured all sizes", flush=True)
  for it in range(NS):
    for g in gs: model.seq_step(allseqs[:g])
    if it % 20 == 0: print("iter", it, flush=True)
  sync()
  print("cycled ok", flush=True)
  raise SystemExit
if os.environ.get("HOSTT"):
  # (the host's part of a step: seq_step's start to its wait for the output (the submit), and the wait)
  orig_wait, tw = model._wait_host, []
  def timed_wait(*a, **kw):
    tw.append(time.perf_counter())
    r = orig_wait(*a, **kw)
    tw.append(time.perf_counter())
    return r
  model._wait_host = timed_wait
  sub, wait = [], []
  for _ in range(NS):
    tw.clear()
    t0 = time.perf_counter()
    model.seq_step(seqs)
    sub.append(tw[0]-t0)
    wait.append(tw[1]-tw[0])
  print(f"host submit {sorted(sub)[NS//2]*1000:.3f} ms wait {sorted(wait)[NS//2]*1000:.3f} ms (medians)", flush=True)
  model._wait_host = orig_wait
for rep in range(int(os.environ.get("REPS", 3))):
  sync()
  st = time.perf_counter()
  for _ in range(NS): model.seq_step(seqs)
  sync()
  print(f"step {(time.perf_counter()-st)/NS*1000:.3f} ms", flush=True)
if os.environ.get("PROF"):
  sync()
  profile_marker("a")
  for _ in range(2): model.seq_step(seqs)
  sync()
  profile_marker("b")
