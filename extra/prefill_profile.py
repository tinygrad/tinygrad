#!/usr/bin/env python3
"""Profile Qwen cold prefills and warm turns through SeqEngine.

Environment: MODEL (first GGUF shard), COLD=4096,261896, COLD_REPS=2,
WARM_CTX=100000, NEWS=1024, REPS=3, ROOM=8192, MAXCTX=262144, SHARD=6.
Run with PROFILE=1 to emit per-kernel GPU times in addition to wall/host times.
"""
import collections
import json
import os
import random
import statistics
import sys
import threading
import time

from tinygrad import Device
from tinygrad.helpers import Context, getenv
from tinygrad.llm.engine import SeqEngine
from tinygrad.llm.model import Transformer

output_lock = threading.Lock()
def emit(kind: str, row: dict):
  with output_lock: print(kind + " " + json.dumps(row, sort_keys=True), flush=True)


def log(*args): print(*args, file=sys.stderr, flush=True)


model, _ = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 262144), shard=getenv("SHARD", 6))
st = time.perf_counter()
with Context(DEBUG=0): model.warmup()
log(f"warmup {time.perf_counter()-st:.3f}s chunks={model.prefill_chunks()}")
eng = SeqEngine(model)
devs = tuple(model.token_embd.weight.device)

stop_metrics = None
metrics_thread = None
if getenv("METRICS"):
  from tinygrad.runtime.ops_amd import AMIdleClocks
  stop_metrics = threading.Event()
  def monitor_metrics():
    while not stop_metrics.wait(1.0):
      samples = [{key: round(value, 3) if isinstance(value, float) else value for key, value in metric.items()}
                 for metric in AMIdleClocks.metrics()]
      emit("METRICS", {"time": time.perf_counter(), "devices": samples})
  metrics_thread = threading.Thread(target=monitor_metrics, daemon=True)
  metrics_thread.start()

spent: collections.Counter = collections.Counter()
calls: collections.Counter = collections.Counter()
chunks: collections.Counter = collections.Counter()


def timed(name: str):
  fn = getattr(model, name)
  def wrap(*args, **kwargs):
    start = time.perf_counter()
    try: return fn(*args, **kwargs)
    finally:
      spent[name] += time.perf_counter() - start
      calls[name] += 1
      if name == "_seq_chunk": chunks[args[2]] += 1
  setattr(model, name, wrap)


for name in ("_seq_copy_call", "_seq_chunk", "seq_ckpt", "_wait_host", "_chunk_plan"):
  timed(name)


def sync():
  for device in devs: Device[device].synchronize()


def measure(label: str, seq, tokens: list[int], start: int, restore: int):
  spent.clear()
  calls.clear()
  chunks.clear()
  if getenv("PROFILE"):
    import kprof
    kprof.reset(*devs)
  sync()
  begin = time.perf_counter()
  eng.prefill(seq, tokens, start, restore, (0.0, 1.0))
  sync()
  elapsed = time.perf_counter() - begin
  row = {"label": label, "prompt": len(tokens), "start": start, "restore": restore,
         "new": len(tokens)-start, "ms": round(elapsed*1e3, 3), "tok_s": round((len(tokens)-start)/elapsed, 3),
         "host_ms": {key: round(value*1e3, 3) for key, value in spent.items()}, "calls": dict(calls),
         "chunks": {str(rows): chunks[rows] for rows in sorted(chunks)}}
  emit("CASE", row)
  if getenv("PROFILE"):
    kernel_times = kprof.times(*devs)
    for name, (count, micros) in sorted(kernel_times.items(), key=lambda item: -item[1][1]):
      emit("KERNEL", {"label": label, "name": name, "count": count,
                      "us_all_devices": round(micros, 3), "us_per_device": round(micros/len(devs), 3)})
  return elapsed


rng = random.Random(0)
seq = None

def reset_sequence():
  global seq
  if seq is not None:
    eng.stop(seq, len(seq.tokens))
    eng.forget()
    seq = None

cold_reps = getenv("COLD_REPS", 2)
for ctx in [int(x) for x in os.environ.get("COLD", "4096,261896").split(",") if x]:
  durations = []
  for rep in range(cold_reps):
    reset_sequence()
    tokens = [rng.randrange(1000, 100000) for _ in range(ctx)]
    got = eng.take(tokens, getenv("ROOM", 8192), force=True)
    assert got is not None
    seq, start, restore = got
    durations.append(measure(f"cold_{ctx}_r{rep}", seq, tokens, start, restore))
  emit("SUMMARY", {"label": f"cold_{ctx}", "ms_median": statistics.median(durations)*1e3,
                    "tok_s_median": ctx/statistics.median(durations)})

# Establish a cached prompt, then measure successive turns extending that same sequence.
reset_sequence()
warm_ctx = getenv("WARM_CTX", 100000)
tokens = [rng.randrange(1000, 100000) for _ in range(warm_ctx)]
got = eng.take(tokens, getenv("ROOM", 8192), force=True)
assert got is not None
seq, start, restore = got
measure(f"warm_base_{warm_ctx}", seq, tokens, start, restore)
for new_count in [int(x) for x in os.environ.get("NEWS", "1024").split(",") if x]:
  durations = []
  for rep in range(getenv("REPS", 3)):
    new_tokens = seq.tokens + [rng.randrange(1000, 100000) for _ in range(new_count)]
    eng.stop(seq, len(seq.tokens))
    got = eng.take(new_tokens, 64)
    assert got is not None
    seq, start, restore = got
    durations.append(measure(f"warm_{warm_ctx}_add_{new_count}_r{rep}", seq, new_tokens, start, restore))
  emit("SUMMARY", {"label": f"warm_{warm_ctx}_add_{new_count}", "ms_median": statistics.median(durations)*1e3,
                    "tok_s_median": new_count/statistics.median(durations)})

if stop_metrics is not None:
  stop_metrics.set()
  metrics_thread.join(timeout=2)
