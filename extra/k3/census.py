#!/usr/bin/env python3
# DEV=AMD:HIP python3 extra/k3/census.py trunk.gguf [blocks]: the K3 trunk cut to its first blocks (experts zero), decode ms/token and the
# kernels each gpu runs per token, by name: what the trunk's time per layer is made of
import collections, json, os, re, sys, time
from tinygrad import Device
from tinygrad.helpers import getenv
from tinygrad.runtime.support.hcq2 import HCQInfo
from tinygrad.llm.model import Transformer
from extra.k3.engine import cut_blocks

cut_blocks(NB:=int(sys.argv[2]) if len(sys.argv) > 2 else 5)

model, kv = Transformer.from_gguf(sys.argv[1], max_context=1024, shard=getenv("SHARD", 4))
toks, st = [], 0.0
for i, t in enumerate(model.generate([0, 1000, 2000])):
  if i == 3: st = time.perf_counter()
  toks.append(t)
  if i == 13: break
Device[Device.DEFAULT].synchronize()
print(f"{NB} blocks: {(time.perf_counter() - st) / 10 * 1e3:.2f} ms/token, tokens {toks}")
lin = next(iter(model.rollout_jit.captured._linears.values()))[0]  # (the linked linear: its hcq batches)
calls = [u for u in lin.toposort() if u.op.name == "CALL" and isinstance(getattr(u.arg, "aux", None), HCQInfo)]
if not calls: print("linear srcs:", collections.Counter((u.op, type(u.arg).__name__) for u in lin.src).most_common(8))
per = collections.Counter((d, re.sub(r"\x1b\[[0-9;]*m", "", n)) for c in calls for d, n, *_ in c.arg.aux.kernels)
print(f"{len(calls)} hcq batches, {sum(per.values())} kernels")
dev0 = collections.Counter({n: k for (d, n), k in per.items() if d in ("AMD", "AMD:0")})
if "OUT" in os.environ: json.dump(dev0, open(os.environ["OUT"], "w"))
for n, k in dev0.most_common(40): print(f"{k:5d}  {n}")

# PROFILE=1: the gpu time of the last token by kernel, and how much of each gpu's span is busy
from tinygrad.device import Compiled, ProfileGraphEvent
if (evs:=[e for e in Compiled.profile_events if isinstance(e, ProfileGraphEvent)]):
  ev, dur, span = max(evs[-6:], key=lambda e: len(e.ents)), collections.Counter(), {}
  for ent in ev.ents:
    st_, en_ = float(ev.sigs[ent.st_id]), float(ev.sigs[ent.en_id])
    name = re.sub(r"\x1b\[[0-9;]*m", "", str(ent.name))
    dur[(ent.device, name)] += en_ - st_
    lo, hi, busy = span.get(ent.device, (st_, en_, 0.0))
    span[ent.device] = (min(lo, st_), max(hi, en_), busy + en_ - st_)
  for d, (lo, hi, busy) in sorted(span.items()): print(f"{d}: span {hi - lo:.0f} us, busy {busy:.0f} us")
  g0 = collections.Counter({n: t for (d, n), t in dur.items() if d in ("AMD", "AMD:0")})
  for n, t in g0.most_common(30): print(f"{t:8.0f} us  {dev0[n]:4d}x  {n}")
  # per kernel instance: its bytes (the estimate) and time, the gemvs grouped by size
  kerns = [k for c in calls for k in c.arg.aux.kernels if k[0] == ev.ents[0].device]
  if len(kerns) == len(ev.ents):
    by = collections.defaultdict(lambda: [0, 0.0, 0])
    for k, ent in zip(kerns, ev.ents):
      name = re.sub(r"\x1b\[[0-9;]*m", "", str(ent.name))
      if ent.device in ("AMD", "AMD:0") and name.startswith("linear"):
        mem = int(k[2].mem) if isinstance(k[2].mem, int) else 0
        b = by[(name, mem)]
        b[0], b[1], b[2] = b[0] + 1, b[1] + float(ev.sigs[ent.en_id]) - float(ev.sigs[ent.st_id]), mem
    for (name, mem), (n, t, _) in sorted(by.items(), key=lambda e: -e[1][1]):
      print(f"{name:12s} {mem / 1e6:8.2f} MB x{n:3d}: {t / n:7.1f} us each, {mem * n / t / 1e3:6.0f} GB/s")
  else: print("kernels and profile entries differ", len(kerns), len(ev.ents))
