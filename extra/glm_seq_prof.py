#!/usr/bin/env python3
"""
GLM_SEQ batched decode: the step time of G sequences decoded together (seq_step) and its GPU kernel times (PROFILE=1), every sequence a
corpus prompt of CTXP tokens prefilled on its own region (slot g). the steps' tokens are the model's (greedy; accepted drafts as in use).
env: MODEL, CORPUS, SHARD (6), MAXCTX (65536), GS (1,8,16,32: the step sizes measured, each needs SEQ_SIZES to have it and SEQ_SLOTS >= it),
     CTXP (4096), NSTEPS (32), TOP (40), LIN_NAMES=1,
     SAMPLED=1: position sequences at CTXP with allocated but unfilled caches; feed distinct real corpus streams (temperature 1).
     Use WARMUP_STEPS=50 NSTEPS=300 for the board sampled test. Full real prompt fills remain the default.
     KSEQ=600 (last N dense kernels in step order), KSEQ_DEV=1 (device index)
Prints a rolling SHA256 over every target sample and draft, in order, for exact A/B trajectory comparisons.
run: PROFILE=1 PYTHONPATH=. MLA_SEQ_SHARD=1 GLM_SEQ=1 SEQ_ROWS=... SEQ_SLOTS=32 SEQ_SIZES=1,2,4,8,16,32 python3 extra/glm_seq_prof.py
"""
import os, time, array, hashlib
from tinygrad import Device
from tinygrad.helpers import getenv
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.glm5next import REGION_ALIGN
import extra.kprof as kprof

model, kv = Transformer.from_gguf(os.environ["MODEL"], getenv("MAXCTX", 65536), shard=getenv("SHARD", 6))
# (the derived weights as a server's warmup makes them: the routers' transposed float weights (skinny_matmul), else the routers of many
# rows run a generic kernel ~30x slower: 13.8 ms a step at 28 users)
if hasattr(model, "_finish_load"): model._finish_load()
assert model.config.cache_rows, "GLM_SEQ=1 SEQ_ROWS=..."
tok = SimpleTokenizer.from_gguf_kv(kv)
devs = model.token_embd.weight.device
devs = devs if isinstance(devs, tuple) else (devs,)
k, CTXP, NS = model.config.spec_drafts, getenv("CTXP", 4096), getenv("NSTEPS", 32)
GS = [int(g) for g in str(getenv("GS", "1,8,16,32")).split(",")]
corpus = tok.encode(open(os.environ["CORPUS"]).read()[:getenv("CHARS", 2_000_000)])
CYCLE = getenv("CYCLE", 0)
SAMPLED = bool(getenv("SAMPLED", 0))
WARM = getenv("WARMUP_STEPS", 50 if SAMPLED else 3)
REAL_STREAM = bool(getenv("REAL_STREAM", SAMPLED))
TEMP = getenv("SAMPLE_TEMP", 1.0 if SAMPLED else 0.0)
assert not SAMPLED or not CYCLE, "sampled unfilled contexts have no real checkpoint to restore"
room = min(NS, CYCLE) if CYCLE else NS + max(0, WARM-3)
span = -(-(CTXP + 2*room*(k+1) + 64) // REGION_ALIGN) * REGION_ALIGN  # (CYCLE: prompt checkpoints bound the rows too)
G = max(GS)
assert G * span <= model.seq_rows() and G <= model.seq_slots(), f"{G} regions of {span} rows: SEQ_ROWS >= {G*span}, SEQ_SLOTS >= {G}"
def sync():
  for d in devs: Device[d].synchronize()
st0 = time.perf_counter()
seqs = []
for g in range(G):
  # (sampled: the streams' starts wrap around the corpus, many users of 262k need more than it has)
  need, o = CTXP + 3*(NS+WARM) + k + 2, 1000 + g*CTXP*2
  if SAMPLED and o + need > len(corpus): o = 1000 + (o * 7919) % (len(corpus) - need - 1000)
  T = corpus[o:o + need]
  ids = T[:CTXP]
  assert len(ids) == CTXP, f"prompt {g}: got {len(ids)} tokens, need {CTXP} (increase CHARS)"
  if SAMPLED:
    seqs.append(dict(slot=g, base=g*span, mtp=CTXP, j=0, par=1, y=T[CTXP:CTXP+k+1], n=0,
                     outhash=hashlib.sha256(), stream=T))
  else:
    first = model.seq_prefill(g, g*span, list(ids), 0, 0)
    seqs.append(dict(slot=g, base=g*span, mtp=len(ids), j=0, par=1, y=[first] + [0]*k, n=0,
                     outhash=hashlib.sha256(), stream=T))
if SAMPLED: model._seq_init()  # normally seq_prefill/seq_step does this: allocate caches, slots and telemetry without filling them
if SAMPLED and getenv("SAMPLED_FILL", 0):
  # (SAMPLED_FILL=1: the indexer's pooled keys (GLM_SEQ: cache_pk, all the regions') random fp8 rows (a real context's spread of scores,
  # not an unfilled cache's all equal ones: the pool selection's time depends on them). the latent cache stays unfilled)
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.glm5next import MLATransformerBlock
  from tinygrad.llm.kernels.amd_glm import SEQ_BLK
  from tinygrad.device import Buffer, MultiBuffer
  rng, filled, arr = np.random.default_rng(1), 0, None
  for b in model.blk + [model.mtp]:
    if not isinstance(b, MLATransformerBlock) or b.indexer is None or b.indexer.cache_pk.dtype != dtypes.uint32: continue
    pk = b.indexer.cache_pk
    if arr is None:
      # (one array of every pool's row (the same for all the layers), by its pool: the same keys split (SEQ_IDX_SHARD) or not)
      sz = (int(pk.shape[0]), int(pk.shape[1])*4)
      arr = (rng.integers(0, 0x7F, size=sz, dtype=np.uint8) | (rng.integers(0, 2, size=sz, dtype=np.uint8) << 7)).view(np.uint32)
      arr[:, -1] = np.array([1e-3 / 256], dtype=np.float32).view(np.uint32)[0]
    if pk.uop.axis == 0:
      # (split: device d's local row l is pool (l//PB)*(PB*n) + d*PB + l%PB, see pool_index_shard)
      n, PB = len(devs), SEQ_BLK // b.indexer.idx.kpool
      l = np.arange(int(pk.shape[0]) // n)
      chunks = [arr[(l//PB)*(PB*n) + d*PB + l%PB] for d in range(n)]
    else: chunks = [arr] * len(devs)
    # (copied into the cache's buffers from the host: no device copy of the rows (39 users at 262k: a 54 MB temporary was the OOM))
    pkb = pk.realize().uop.base.buffer
    dbufs = pkb.bufs if isinstance(pkb, MultiBuffer) else [pkb]
    if all(db.nbytes == c.nbytes for db, c in zip(dbufs, chunks)) and getenv("SAMPLED_FILL_DIRECT", 1):
      for db, c in zip(dbufs, chunks):
        db.ensure_allocated().copy_from(Buffer("NPY", db.nbytes, opaque=memoryview(np.ascontiguousarray(c).view(np.uint8).reshape(-1))))
    else:
      pk.assign(Tensor(np.ascontiguousarray(np.concatenate(chunks))).shard(devs, axis=0) if pk.uop.axis == 0 else
                Tensor(np.ascontiguousarray(arr)).shard(devs) if len(devs) > 1 else Tensor(arr, device=devs[0])).realize()
    filled += 1
  sync()
  print(f"SAMPLED_FILL {filled} pooled key caches random fp8 rows", flush=True)
kv_card = model.stash_chunk_nbytes // model.SEQ_MOVE
if SAMPLED: print(f"SAMPLED_TEST {G} distinct real streams positioned at {CTXP}; caches allocated but unfilled; "
                  f"temperature={TEMP} warmup={WARM} steps={NS}", flush=True)
else: print(f"{G} prompts of {CTXP} prefilled in {time.perf_counter()-st0:.1f}s", flush=True)
print(f"RESIDENCY rows={model.seq_rows()} slots={model.seq_slots()} span={span} "
      f"{'positioned_rows' if SAMPLED else 'prompt_rows'}={G*CTXP} kv_bytes_per_row_card={kv_card} "
      f"kv_bytes_per_row_box={kv_card*len(devs)} stash_gb={getenv('SEQ_STASH_GB', 4.0)}", flush=True)
def restore(group:list[dict]):
  # Restore the actual prompt checkpoint, not cached/short prompt contents. Generated rows past the prompt are not read after this reset.
  for s in group:
    model._seq_copy_call(False, s["slot"], CTXP, 0)
    first = model.seq_first(s["slot"], s["base"], CTXP, 0.0)
    s["mtp"], s["j"], s["par"], s["y"] = CTXP, 0, 1, [first] + [0]*k

def step(group:list[dict]):
  outs = model.seq_step([(s["slot"], s["base"], s["mtp"], s["j"], s["par"], s["y"], TEMP) for s in group])
  for s, (sampled, drafts) in zip(group, outs):
    s["outhash"].update(array.array('i', [*sampled, *drafts]).tobytes())  # every target sample and draft, in order
    j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
    st = s["mtp"] + s["j"]
    incoming = s["stream"][st+1:st+2+j] if REAL_STREAM else sampled[:j+1]
    model._token_host[s["base"] + st + 1:s["base"] + st + 2 + j] = array.array('i', incoming)
    y = [s["stream"][st+1+t] for t in range(k+1)] if REAL_STREAM else sampled
    s["mtp"], s["j"], s["par"], s["y"], s["n"] = st + 1, j, 1 - s["par"], y, s["n"] + j + 1
for g in GS:
  group = seqs[:g]
  if SAMPLED:
    for s in group:
      s["mtp"], s["j"], s["par"], s["y"] = CTXP, 0, 1, s["stream"][CTXP:CTXP+k+1]
  for _ in range(WARM): step(group)  # (its jit captured)
  sync()
  n0 = sum(s["n"] for s in group)
  if CYCLE: restore(group)
  if getenv("PROFILE"): kprof.reset(*devs)
  if hasattr(model, "step_times"): model.step_times[:] = [0.0]*len(model.step_times)
  wall0 = time.time()
  decode_dt = 0.0
  done = 0
  while done < NS:
    st = time.perf_counter()
    for _ in range(min(NS-done, CYCLE or NS)): step(group)
    sync()
    decode_dt += time.perf_counter() - st
    done += min(NS-done, CYCLE or NS)
    if CYCLE and done < NS: restore(group)
  wall1 = time.time()
  dt = wall1-wall0
  toks = sum(s["n"] for s in group) - n0
  print(f"WINDOW start={wall0:.6f} end={wall1:.6f} elapsed={dt:.6f} decode_elapsed={decode_dt:.6f} "
        f"restore_elapsed={dt-decode_dt:.6f} accepted_tokens={toks} steps={NS} users={g} prompt_tokens={CTXP} "
        f"wall_tok_s={toks/dt:.3f}", flush=True)
  outhash = hashlib.sha256()
  for s in group: outhash.update(bytes.fromhex(s["outhash"].hexdigest()))
  print(f"G={g}: {decode_dt/NS*1e3:.2f} ms/step, {toks/NS/g:.2f} tok/step/seq, {toks/dt:.0f} tok/s aggregate, "
        f"{toks/dt/g:.1f} tok/s/seq, outhash {outhash.hexdigest()[:16]}", flush=True)
  if hasattr(model, "step_times"):
    # (the host's part of a step: the jit's launch (its kernels queued; the devices run meanwhile), then the wait for the sampled tokens)
    print(f"STEP_HOST launch {model.step_times[0]/NS*1e3:.2f} ms/step, wait {model.step_times[1]/NS*1e3:.2f} ms/step", flush=True)
  if getenv("PROFILE"):
    t = kprof.times(*devs)
    tot = sum(v[1] for v in t.values()) / len(devs)
    print(f"  GPU kernel time per device per step: {tot/NS/1e3:.3f} ms", flush=True)
    for name, (c, us) in sorted(t.items(), key=lambda x: -x[1][1])[:getenv("TOP", 40)]:
      print(f"  {us/len(devs)/NS:9.1f} us/step  {c/len(devs)/NS:6.2f}/step  {name}", flush=True)
    if (kdev:=getenv("KDEV", 0)):
      # (every device's time of the top KDEV kernels: the exchanges' spread is the wait for the slowest device's work before them)
      per = kprof.per_device(*devs)
      print("  KDEV devices " + " ".join(sorted(per)), flush=True)
      for name, _ in sorted(t.items(), key=lambda x: -x[1][1])[:kdev]:
        print(f"  KDEV {name[:40]:40s} " + " ".join(f"{per[d].get(name, 0.0)/NS:8.1f}" for d in sorted(per)), flush=True)
    if (nc:=getenv("CRIT", 0)):
      # (CRIT=n: the timeline of n more steps: every device's span and busy time, the time no device ran anything (the host: PROFILE=1
      # syncs a step's start on its launch), the critical path (kernels, copies, waits on another lane) and its kernels)
      kprof.reset(*devs)
      for _ in range(nc): step(group)
      sync()
      tl = kprof.sequence_timed(*devs)
      for d in sorted({d for d, *_ in tl}):
        ev = [(st0, st0 + du, n.startswith("copy")) for dd, n, st0, du in tl if dd == d]
        kk, cc = [(a, b) for a, b, c in ev if not c], [(a, b) for a, b, c in ev if c]
        span, _ = kprof.busy_all([(d, "", a, b - a) for a, b, _ in ev])
        print(f"  CRIT {d:8s} span {span/nc/1e3:7.2f} ms/step  kernels {kprof.busy_all([(d, '', a, b-a) for a, b in kk])[1]/nc/1e3:7.2f}"
              f"  copies {kprof.busy_all([(d, '', a, b-a) for a, b in cc])[1]/nc/1e3 if cc else 0.0:7.2f}", flush=True)
      span, busy = kprof.busy_all(tl)
      print(f"  CRIT all: span {span/nc/1e3:.2f} ms/step, some device busy {busy/nc/1e3:.2f}, none busy {(span-busy)/nc/1e3:.2f}", flush=True)
      cp = kprof.critical_path(tl)
      print(f"  CRIT path {cp['wall']/nc/1e3:.2f} ms/step = kernels {cp['kernels']/nc/1e3:.2f} + copies {cp['copies']/nc/1e3:.2f} + waits "
            f"{cp['wait']/nc/1e3:.2f} + lead {cp['lead']/nc/1e3:.2f}, {cp['switches']/nc:.0f} lane switches; waits "
            f"{dict((kk, round(v/nc/1e3, 2)) for kk, v in cp['waits'].items())}", flush=True)
      for (d, c), us in cp["devs"].most_common(12): print(f"  CRIT on {d:8s} {'copies ' if c else 'kernels'} {us/nc/1e3:7.2f} ms/step", flush=True)
      for n, us in cp["names"].most_common(getenv("TOP", 40)): print(f"  CRIT path {us/nc/1e3:7.2f} ms/step  {n}", flush=True)
    if (nseq:=getenv("KSEQ")):
      did = devs[getenv("KSEQ_DEV", 1)]
      seq = [(name, us) for dev, name, us in kprof.sequence(*devs) if dev == did and
             name.startswith(("q8_gemm", "linear_q6", "linear_q8_0"))][-nseq:]
      for i, (name, us) in enumerate(seq): print(f"  KSEQ {did} {i:4d} {us:9.2f} us  {name}", flush=True)
# (the memory every device holds at the end: the caches, the weights and the jit's buffers; the headroom for more users is the card's less this)
from tinygrad.helpers import GlobalCounters
print("MEM_USED GB " + " ".join(f"{GlobalCounters.mem_used_per_device.get(d, 0)/1e9:.3f}" for d in devs), flush=True)
if getenv("MEM_DUMP"):
  # (MEM_DUMP=n: device 0's allocated buffers by owner: the model's named tensors (weights, caches), the jits' shared scratch, the rest
  # by size (the n largest): where a user's memory outside the cache goes)
  import gc, collections
  from tinygrad.uop.ops import Ops
  from tinygrad.device import Buffer
  from tinygrad.nn.state import get_state_dict
  d0, names = devs[0], {}
  def bufs_of(u):
    try: b = u.buffer
    except Exception: return []
    return [x.base for x in (b.bufs if hasattr(b, "bufs") else [b]) if x.device == d0]
  for nm, t in get_state_dict(model, tensor_type=__import__("tinygrad").Tensor).items():
    for b in bufs_of(t.uop.base): names.setdefault(id(b), nm)
  for kk, u in getattr(model, "_scratch", {}).items():
    for b in bufs_of(u): names.setdefault(id(b), f"scratch[{kk}]")
  from tinygrad.engine.jit import _TinyJit
  for an, av in list(vars(model).items()):
    for jk, j in (av.items() if isinstance(av, dict) else [(None, av)]):
      if not isinstance(j, _TinyJit) or j.captured is None: continue
      for lin in [j.captured._linear] + [v[0] for v in j.captured._linears.values()]:
        for u in lin.toposort():
          if u.op is Ops.BUFFER:
            for b in bufs_of(u): names.setdefault(id(b), f"jit {an}[{jk}]")
  tot:collections.Counter = collections.Counter()
  rest:collections.Counter = collections.Counter()
  for o in gc.get_objects():
    if not isinstance(o, Buffer) or o._base is not None or o.device != d0 or not o.is_allocated(): continue
    nm = names.get(id(o))
    cls = "unnamed" if nm is None else "scratch" if nm.startswith("scratch") else "jit" if nm.startswith("jit") else \
      "cache" if any(w in nm for w in ("cache", "pool", "slot", "ring")) else "named"
    tot[cls] += o.nbytes
    rest[(nm if cls in ("scratch", "cache", "jit") else cls, o.nbytes, str(getattr(o, "dtype", "")))] += 1
  print("MEM_DUMP " + " ".join(f"{c}={v/1e9:.3f}GB" for c, v in tot.most_common()), flush=True)
  for (nm, nb, dt), c in sorted(rest.items(), key=lambda x: -x[0][1]*x[1])[:getenv("MEM_DUMP")]:
    print(f"MEM_DUMP {nb*c/1e6:10.1f} MB {c:4d} x {nb:12d} B {dt:10s} {nm}", flush=True)
