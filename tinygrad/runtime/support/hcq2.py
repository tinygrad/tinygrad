from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, TYPE_CHECKING
import functools, time, itertools, decimal, weakref, os, statistics
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, unwrap, PROFILE, VIZ
from tinygrad.helpers import to_tuple, ContextVar, Context, panic, partition, perf_counter_us
from tinygrad.device import Device, Buffer, MultiBuffer, BufferSpec, Compiled, LRUAllocator, DepsTracker
from tinygrad.device import ProfileGraphEntry, ProfileGraphEvent, ProfileDeviceEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu
from tinygrad.dtype import dtypes, DType
from tinygrad.runtime.support.memory import BumpAllocator, MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop, pm_flatten_linear
from tinygrad.engine.realize import lower_and_compile

if TYPE_CHECKING: from tinygrad.runtime.support.hcq import HCQBuffer # TODO: remove that

# *****************
# 0. helpers

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')
HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "CPU")
HCQ_DEVS = frozenset(("AMD", "CPU"))

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]

  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = () # (devices, name, estimates, timestamp slots, profile key)

  nargs:int = 0
  table:int = -1
  inputs:tuple[tuple[UOp, str], ...] = ()
  slots:tuple[tuple[str, int], ...] = () # per device, the position of its batch slots in the args

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, byte offset)
  if v.op in (Ops.BITCAST, Ops.AFTER): return unwrap_view(v.src[0])
  if v.op is not Ops.SHRINK: return v, 0
  base, off = unwrap_view(v.src[0])
  return base, off + v.src[1].val * v.dtype.itemsize

def select_lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

def to_name(*parts:str) -> str: return "_".join(parts).replace(":", "_").lower()

def timeline(devs:tuple[str, ...]) -> UOp: return UOp.placeholder((2,), dtypes.uint64, 0, device=devs, volatile=True, tag="timeline")
def timeline_value(devs:tuple[str, ...]) -> UOp: return timeline(devs).index(1).load()

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  fn = to_name("submit", (devs:=to_tuple(devs))[0].split(":")[0], queue.split(":")[0])
  return UOp.custom_function(fn, UOp(Ops.LINEAR, src=tuple(cmds), arg=(devs, queue)))

# *****************
# 0.1. prep: replace buffers with params

def replace_call_buffers(ctx:tuple[list[UOp], dict[UOp, int]], call:UOp) -> UOp|None:
  bufs, slots = ctx
  for s in call.src[1:]:
    if s.op is not Ops.PARAM and not s.is_bound_var and slots.setdefault(s, len(bufs)) == len(bufs): bufs.append(s)
  return call.replace(src=call.src[:1] + tuple(s if s.op is Ops.PARAM or s.is_bound_var else s.param_like(slots[s]) for s in call.src[1:]))
pm_replace_buffers = PatternMatcher([(UPat(Ops.CALL, name="call"), replace_call_buffers)])

# *****************
# 1.1. prep: staging copies

STAGING_SIZE, STAGING_SLOTS = (4 if os.getenv("CI") else 128) << 20, 2 # the staging mlocks into the device: ci runners cap locked memory at 8mb

@functools.cache
def _staging() -> Buffer: return Buffer("CPU", STAGING_SIZE, dtypes.uint8, preallocate=True)

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS - {"CPU"}) and not all_devices_in(b.device, HCQ_DEVS)

def stage_copy_ext(call:UOp) -> UOp|None:
  if (d:=next((d for b in call.src[1:] for d in to_tuple(b.device) if not d.startswith("CPU")), None)) is None: return None
  return pm.rewrite(call) if (pm:=getattr(Device[d], "pm_stage_copy", None)) is not None else None

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  assert src.dtype.itemsize == dst.dtype.itemsize, "staged copies must be dtype-size matched"
  base, it, copies = UOp.from_buffer(_staging()), src.dtype.itemsize, []
  chunk = (STAGING_SIZE // STAGING_SLOTS) // it
  for i, off in enumerate(range(0, src.max_numel(), chunk)):
    stage = base[(so:=(i % STAGING_SLOTS) * chunk * it):so + (n:=min(chunk, src.max_numel() - off)) * it]
    copies += [src[off:off+n].copy_to_device("CPU").call(stage, src[off:off+n]), stage.copy_to_device(dst.device).call(dst[off:off+n], stage)]
  return UOp(Ops.LINEAR, src=tuple(copies))

# *****************
# 1.2. prep: one call per device: the args pick their lane, the DEVICE axis binds to it

def unwrap_call(call:UOp) -> UOp|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY) or (n:=max(len(to_tuple(a.device)) for a in get_call_arg_uops(call))) == 1: return None
  dnum = UOp.variable("_device_num", 0, n - 1, dtypes.int)
  return UOp(Ops.LINEAR, src=tuple(call.replace(src=(call.src[0], *[a if a.is_bound_var else select_lane(a, i) for a in call.src[1:]], dnum.bind(i)))
                                   for i in range(n)))
pm_unwrap_multi = PatternMatcher([(UPat(Ops.CALL, name="call"), unwrap_call)])

# *****************
# 1.3. prep: kernel copies

def _get_enqueue_devs(call:UOp) -> Any|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=get_call_arg_uops(call)) or not all(all_devices_in(b.device, HCQ_DEVS) for b in bufs): return None
  if call.src[0].op is Ops.COPY: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: to_tuple(b.device)[0].startswith("CPU")).device # prio to enqueue on not CPU device
  # cpu has no queue (yet)
  return devs if all_devices_in(devs, HCQ_DEVS) and not to_tuple(devs)[0].startswith("CPU") else None

def copy_with_kernel(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  if (devs:=_get_enqueue_devs(call)) is None or Device[(dev:=to_tuple(devs)[0])].has_copy_queue: return None
  d, s = (UOp.param(i, dst.dtype, n:=dst.max_numel(), device=devs) for i in range(2))
  ast = d.index(r:=UOp.range(n, 0)).store(s.index(r).load()).end(r).sink(arg=KernelInfo(name="copy"), tag=1)
  return call.replace(src=(to_program(ast, Device[dev].renderer), dst, src))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY),), name="call", allow_any_len=True), stage_copy_ext),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src")), name="call"), copy_with_kernel)
])

# *****************
# 2. deps

class HCQDepsTracker(DepsTracker):
  # TODO: optimize
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    return (buf, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...], str]] # (call, devices, queue) per enqueued call
  profile:bool
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker)
  queues:dict[str, list[str]] = field(init=False)
  first:dict[tuple[str, str], int] = field(init=False); last:dict[tuple[str, str], int] = field(init=False) # noqa: E702
  signal_tags:set[int] = field(init=False)
  slots:dict[str, UOp] = field(init=False)

  def __post_init__(self):
    self.queues, self.first, self.last = {}, {}, {}
    for tag, (_, devs, q) in enumerate(self.batch):
      if q not in self.queues.setdefault(devs[0], []): self.queues[devs[0]].append(q)
      self.first.setdefault((devs[0], q), tag)
      self.last[(devs[0], q)] = tag
    self.signal_tags = {tag for (dev, q), tag in self.last.items() if q != self.epilogue_queue(dev)}
    self.slots = {dev: UOp.placeholder((len(qs) + 1 + (2 * len(self.batch) if self.profile else 0),), dtypes.uint64, device=(dev,), volatile=True,
                                       tag="slots") for dev, qs in self.queues.items()}

  def epilogue_queue(self, dev:str) -> str: return "COMPUTE:0" if len(self.queues[dev]) > 1 else self.queues[dev][0] # closes the device

  def slot(self, devs:tuple[str, ...], i:int) -> UOp: return self.slots[devs[0]].shrink(((i, i + 1),)) # not [i:i+1]: the slice path is 10x the cost
  def queue_signal(self, devs:tuple[str, ...], queue:str) -> UOp: return self.slot(devs, self.queues[devs[0]].index(queue))
  def sched_timeline(self, devs:tuple[str, ...]) -> UOp: return self.slot(devs, len(self.queues[devs[0]]))
  def stamps(self, devs:tuple[str, ...], tag:int) -> tuple[int, ...]: return (st:=len(self.queues[devs[0]])+1+2*tag, st + 1) if self.profile else ()

def _call_bufs(call:UOp) -> list[Any]:
  def dep_buf(b:UOp) -> Any:
    if (base:=(b.src[0] if b.op is Ops.MSELECT else b).storage_base).op is Ops.PARAM: return b if b.op is Ops.MSELECT else base
    return cast(MultiBuffer, base.buffer).bufs[b.arg] if b.op is Ops.MSELECT else base.buffer
  return [dep_buf(a) for a in get_call_arg_uops(call)]

def _wait_ins(ctx:BatchCtx, call:UOp, device:str, queue:str, tag:int) -> list[UOp]:
  bufs, write = _call_bufs(call), get_call_outs_ins(call)[0]
  latest:dict[tuple[str, str], int] = {} # (producer device, queue) -> the latest submit tag to wait on, same-queue submits are fifo
  for d, q, t in ctx.tracker.access_resources(bufs, list(range(len(bufs)) if write is None else write), (device, queue, tag)):
    if t < tag and (d, q) != (device, queue): latest[(d, q)] = max(latest.get((d, q), 0), t)
  ctx.signal_tags |= set(latest.values())
  return [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.queue_signal((d,), q), UOp.const(t + 1, dtypes.uint64))) for (d, q), t in latest.items()]

def _merge_queues(submits:list[UOp]) -> list[UOp]:
  # grouped by queues. can be sent in any order, sync convers that
  return [make_submit(*[c for s in submits if s.src[0].arg == k for c in s.src[0].src], devs=k[0], queue=k[1])
          for k in dedup([s.src[0].arg for s in submits])]

def _emit_submits(ctx:BatchCtx, call_waits:list[list[UOp]]) -> tuple[list[UOp], list[tuple]]:
  # one submit per call: timeline sync on first queue use, timestamps, the call, and a signal if someone waits on it
  src, kerns = [], []
  for tag, ((call, devices, queue), q) in enumerate(zip(ctx.batch, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if ctx.first[(devices[0], queue)] == tag:
      q = [UOp(Ops.INS, arg=("barrier", dtypes.void), src=()),
           UOp(Ops.INS, arg=("wait", dtypes.void), src=(timeline(devices), timeline_value(devices)))] + q

    # and make hcq call
    name, est = get_call_name(call, get_call_arg_uops(call)), estimate_uop(call)
    kerns.append((devices, name, est, ctx.stamps(devices, tag), getattr(call.src[0].arg, "profile_key", None)))

    ts_ins = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(ctx.slot(devices, i),)) for i in ctx.stamps(devices, tag)]
    q += ts_ins[:1] + [call] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags:
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.queue_signal(devices, queue), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_submit(*q, devs=devices, queue=queue))
  return src, kerns

def _epilogue(ctx:BatchCtx, dev:str) -> UOp:
  # one queue signals the timeline once the last call of every other queue signaled
  waits = [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.queue_signal((dev,), q), UOp.const(ctx.last[(dev, q)] + 1, dtypes.uint64)))
           for q in ctx.queues[dev] if q != ctx.epilogue_queue(dev)]
  bump = UOp(Ops.INS, arg=("store", dtypes.void), src=(timeline((dev,)), timeline_value((dev,)) + UOp.const(1, dtypes.uint64)))
  return make_submit(*waits, bump, devs=dev, queue=ctx.epilogue_queue(dev))

def _finalize_batch(ctx:BatchCtx) -> UOp:
  call_waits = [_wait_ins(ctx, c, d[0], q, tag) for tag, (c, d, q) in enumerate(ctx.batch)]
  submits, kerns = _emit_submits(ctx, call_waits)
  submits += [_epilogue(ctx, dev) for dev in ctx.queues]
  fence = UOp.custom_function("hcq_fence", *[ctx.sched_timeline((dev,)) for dev in ctx.queues],
                              *[ctx.queue_signal((dev,), q) for dev, qs in ctx.queues.items() for q in qs])
  merged = [m.replace(src=(*m.src, fence)) for m in _merge_queues(submits)]
  estimates = sum((estimate_uop(call) for call, _, _ in ctx.batch), start=Estimates()).simplify()
  return UOp.sink(*merged, arg=KernelInfo("hcq_submit", estimates=estimates), tag=1).call(aux=HCQInfo(tuple(ctx.queues), kernels=tuple(kerns)))

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  devs = [() if (d:=_get_enqueue_devs(c)) is None else tuple(Device.canonicalize(x) for x in to_tuple(d)) for c in l.src]
  queues = ["COMPUTE:0" if c.src[0].op is Ops.PROGRAM else "COPY:0" for c in l.src]
  srcs:list[UOp] = []
  for hcq, grp in itertools.groupby(zip(l.src, devs, queues), key=lambda e: bool(e[1])):
    srcs += [_finalize_batch(BatchCtx(list(grp), profile))] if hcq else [c for c, _, _ in grp]
  return l.replace(src=tuple(srcs))

# *****************
# 3. encode

@dataclass
class EncodeCtx:
  devs:tuple[str, ...]
  inputs:dict[tuple[UOp, str], int] = field(default_factory=dict)
  table:UOp = field(default_factory=lambda: UOp.placeholder((1,), dtypes.uint64, device="CPU", tag="inputs"))
  lt_patches:list[UOp] = field(default_factory=list)

class HWQueue:
  q_rewrite:PatternMatcher

  def __init__(self, ctx:EncodeCtx, submit:UOp):
    self.ctx, self.lin, self.deps = ctx, submit.src[0], list(submit.src[1:])
    self.devs, self.queue = self.lin.arg
    self.dev = Device[self.devs[0]]
    self.blob, self.patches = bytearray(), list[tuple[int, UOp]]()

  def q(self, *words) -> int:
    for w in words:
      c = w
      while isinstance(c, UOp) and c.op is Ops.CAST: c = c.src[0]
      if isinstance(c, UOp) and c.op is not Ops.CONST:
        self.patches.append((len(self.blob), w))
        self.blob += bytes(w.dtype.itemsize)
      else:
        v, n = (c.val, w.dtype.itemsize) if isinstance(w, UOp) else (c, 4)
        self.blob += (v & (1 << 8 * n) - 1).to_bytes(n, 'little')
    return len(self.blob)

  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")

def addrs_to_table(ctx:EncodeCtx, g:UOp) -> UOp|None:
  base, off = unwrap_view(g.src[0])
  param = base.src[0].base if base.op is Ops.MSELECT else base # unwrap mselects
  if param.op is not Ops.PARAM or param.tag is not None: return None
  slot = ctx.inputs.setdefault((base, to_tuple(g.arg)[0]), len(ctx.inputs))
  return ctx.table.index(slot).load() + UOp.const(off, dtypes.uint64)
pm_addrs_to_table = PatternMatcher([(UPat(Ops.GETADDR, name="g"), addrs_to_table)])

def _is_link_patch(w:UOp) -> bool:
  if w.op is Ops.GETADDR: return True
  if w.op in {Ops.LOAD, Ops.INDEX, Ops.PARAM} or w.is_variable: return False
  return all(_is_link_patch(s) for s in w.src)

def patch(buf:UOp, rows:list[tuple[int, UOp]], *deps:UOp) -> UOp: # the buffer after every row's word is stored at its byte offset
  groups:dict[tuple[DType, int], list[tuple[int, UOp]]] = {}
  for o, w in rows: groups.setdefault((w.dtype, o % w.dtype.itemsize), []).append((o, w))

  base, stores = buf.after(*deps), [] # the views hang off the buffer after its deps: the stores wait for them, the caller's after keeps the base flat
  for (dt, phase), grp in groups.items():
    view = base[phase:phase + (buf.max_numel() - phase) // dt.itemsize * dt.itemsize].bitcast(dt)
    stores.append(view.index(UOp.stack(*[UOp.const((o - phase) // dt.itemsize) for o, _ in grp])).store(UOp.stack(*[w for _, w in grp])))
  return buf.after(*deps, *stores)

def encode_submit(hq:HWQueue) -> UOp:
  # applying the rewrite
  for u in hq.lin.src: hq.q_rewrite.rewrite(u, ctx=hq)

  # merge blobs into one
  stream, views = len(hq.blob), {}
  for l in dedup([g.src[0] for _, w in hq.patches for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR]):
    hq.blob += bytes(-len(hq.blob) % 128)
    views[l] = (len(hq.blob), hq.q(*l.src))

  buf = UOp.placeholder((len(hq.blob),), dtypes.uint8, device=hq.devs, tag=to_name("cmdbuf", hq.queue))

  words = UOp.sink(*[w for _, w in hq.patches]).substitute({l: buf[o:e] for l, (o, e) in views.items()})
  words = graph_rewrite(words, pm_addrs_to_table, ctx=hq.ctx, name="addrs to table").src

  links, runtime = partition(list(zip([o for o, _ in hq.patches], words)), lambda r: _is_link_patch(r[1]))
  hq.ctx.lt_patches.append(patch(buf, links, buf.store(UOp(Ops.BINARY, arg=bytes(hq.blob)).bitcast(buf.dtype))))
  return hq.submit(patch(buf, runtime, *hq.deps).shrink(((0, stream),)))

# *****************
# 3.1. hcq special functions

def hcq_fence(ctx:EncodeCtx, f:UOp) -> UOp:
  lasts, sigs = f.src[:len(ctx.devs)], f.src[len(ctx.devs):]
  last:tuple[UOp, ...] = ()

  # wait for prev schedule to not collide
  # TODO: timeout?
  for i, dev in enumerate(ctx.devs):
    slots, off = unwrap_view(lasts[i])
    ctx.lt_patches.append(slots.after(slots.store(UOp(Ops.BINARY, arg=bytes(slots.max_numel() * slots.dtype.itemsize)).bitcast(slots.dtype))))
    done = timeline((dev,)).after(*last, loop:=UOp.loop(i)).index(0).load()
    waited = done.end(loop, done < slots.index(off // slots.dtype.itemsize).load())
    nxt = timeline_value((dev,)) + UOp.const(1, dtypes.uint64)
    last = (timeline((dev,)).after(waited).index(1).store(nxt), slots.after(waited).index(off // slots.dtype.itemsize).store(nxt))

  # re-arm the signals
  for sig in sigs:
    base, off = unwrap_view(sig)
    last = (base.after(*last).index(off // sig.dtype.itemsize).store(0),)
  return last[0].barrier(*last[1:])
pm_hcq_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence)])

# *****************
# 4. lower call

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo) or call.arg.aux.nargs: return None # not an hcq call, or lowered already

  # encode bodies
  ctx = EncodeCtx(call.arg.aux.device)
  pm = sum([Device[d].pm_encode for d in dedup([d.split(":")[0] for d in ctx.devs])], pm_hcq_encode)
  body = graph_rewrite(call.src[0], pm, ctx=ctx, walk=True, name="encode body")

  # resize table
  body = body.substitute({ctx.table: (table:=UOp.placeholder((len(ctx.inputs),), dtypes.uint64, device="CPU", tag="inputs"))})

  # the placeholders become the body's params in visit order, variables bind by name after them, the ranges renumber
  tops = body.toposort()
  bufs, alus = partition([u for u in tops if u.op is Ops.PARAM], lambda u: u.tag is not None)
  names = dedup([a.arg.name for a in alus])
  # bufs to params
  params = {b: UOp.param(i, b.dtype, b.shape, HCQ_RUNTIME_DEV.value, volatile=b.arg.volatile, name=f"{b.arg.name}_{i}") for i, b in enumerate(bufs)}
  # new slots for vars
  vals = {a: a.replace(arg=replace(a.arg, slot=len(bufs) + names.index(a.arg.name))) for a in alus}
  # reenum ranges
  rngs = {r: r.replace(arg=(i,)+r.arg[1:]) for i, r in enumerate(sorted([u for u in tops if u.op is Ops.RANGE], key=lambda r: r.arg))}
  # and sub all of them
  sink = body.substitute(params | vals | rngs)

  # move all lt-patches to the args
  patched = {p.src[0]: p for p in ctx.lt_patches}
  args = [patched.get(b, b) for b in bufs]

  if VIZ: graph_rewrite(UOp.sink(*args), PatternMatcher([]), name="View Link-Time Patches")
  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Body")

  info = replace(call.arg.aux, nargs=len(args), table=bufs.index(table) if table in bufs else -1, inputs=tuple(ctx.inputs),
                 slots=tuple((to_tuple(b.device)[0], i) for i, b in enumerate(bufs) if b.tag == "slots"))
  return call.replace(src=(sink, *args), arg=replace(call.arg, aux=info))
pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

hcq_compile_cache:dict[tuple[UOp, bool, bool], UOp] = {} # uops are hash-consed: the linear itself is the key, plus whether inputs bind

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  # TODO: this needs a cleanup
  bufmap = {s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}
  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear, profile, input_uops is None)))) is None:
    linear = graph_rewrite(linear.substitute(bufmap, walk=True), pm_unwrap_multi+pm_insert_copy_staging+pm_flatten_linear, name="prep calls")
    linear = sched_batches(linear, profile)
    linear = graph_rewrite(linear, pm_encode, walk=True, name="encode")
    with Context(EMULATED_DTYPES=""):
      final_linear = hcq_compile_cache[cache_key] = lower_and_compile(linear).substitute({v: k for k, v in bufmap.items()}, walk=True)
  return final_linear.substitute(bufmap, walk=True)

# *****************
# 5. bufferize placeholders

def bufferize_buf(ctx:bool, b:UOp) -> UOp|None: # ctx: a kept link (the jit's) owns the linear's buffers, a one-shot borrows ring slots
  if b.tag is None: return None # a param, not a placeholder

  dev = cast(HCQ2Compiled, Device[to_tuple(b.device)[0]])

  if b.arg.slot == 0 or b.tag == "program": r = cast(Buffer, unwrap(dev.pm_bufferize.rewrite(b, ctx=dev))) # device state and programs
  elif ctx: r = Buffer(dev.device, b.max_numel(), b.dtype, options=BufferSpec(host=b.arg.volatile, uncached=True, cpu_access=True), preallocate=True)
  else: r = dev.rt_view(b.max_numel() * b.dtype.itemsize, b.dtype, host=b.arg.volatile)

  return UOp.from_buffer(r, HCQ_RUNTIME_DEV.value)
pm_bufferize_placeholders = PatternMatcher([(UPat(Ops.PARAM, name="b"), bufferize_buf)])

# *****************
# 6. link

def resolve_getaddr(ctx:list[UOp], g:UOp) -> UOp|None:
  buf, off = unwrap_view(g.src[0])
  if buf.op not in {Ops.BUFFER, Ops.MSELECT}: return None
  ctx.append(buf) # add to refs
  return UOp.const(cast(Buffer, buf.buffer).get_buf(to_tuple(g.arg)[0]).va_addr + off, dtypes.uint64)

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  if getattr(b:=cast(Buffer, buf.buffer), '_hcq_written', None) is not blob.arg: # TODO: remove me
    cast(Any, b.ensure_allocated())._hcq_written = blob.arg
    b._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_words(buf:UOp, offs:UOp, ws:UOp) -> UOp:
  base, off = unwrap_view(buf)
  mv = cast(Buffer, base.buffer).ensure_allocated()._buf.cpu_view().view(fmt='B')
  for o, w in zip(offs.src, ws.src):
    n, at = w.dtype.itemsize, off + o.val * w.dtype.itemsize
    mv[at:at + n] = (w.val & (1 << 8 * n) - 1).to_bytes(n, 'little')
  return UOp(Ops.NOOP)

pm_link = PatternMatcher([
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  (UPat(GroupOp.ALU, src=UPat.cvar().or_casted(), name="a"),
    lambda a: UOp.const(exec_alu(a.op, a.dtype, [s.val for s in a.src], False), a.dtype)),
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat(name="buf").index(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="offs")).store(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="ws")),
    fold_words),
  (UPat(Ops.AFTER, name="a"), lambda a: None if a.is_bound_var else a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else
   panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

link_linear_cache:weakref.WeakKeyDictionary[UOp, UOp] = weakref.WeakKeyDictionary() # a baked link lives as long as its bound linear

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear)) is not None: return linked
  bufferized = graph_rewrite(linear, pm_bufferize_placeholders, ctx=cache, name="bufferize")
  linked = graph_rewrite(bufferized, pm_link, ctx=(refs:=list[UOp]()), bottom_up=False, name="link")
  if refs: linked = linked.replace(src=(linked.src[0].replace(src=linked.src[0].src + tuple(dedup(refs))), *linked.src[1:]))
  if cache: link_linear_cache[linear] = linked
  return linked

# *****************
# Device classes

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0
  wait_timeout_ms: float = 30000.0
  rt_nbytes: int = 64 << 20 # the pool every per-linear buffer is carved out of
  pm_encode: PatternMatcher = PatternMatcher([]) # the backend's own encode rules, matched by its submit names

  def __init__(self, device:str, allocator:HCQAllocator, compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.can_recover = can_recover

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="timeline"), lambda ctx: ctx.timeline),
      (UPat(Ops.PARAM, tag="program", name="b"),
       lambda ctx, b: ctx.prog_bufs.setdefault(b, Buffer(ctx.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)))),
    ])
    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.prog_bufs:dict[UOp, Buffer] = {}
    self.prof_ents:dict[tuple[Buffer, int], ProfileGraphEntry] = {} # (a batch's timestamps, start slot) -> entry, read at synchronize

  @functools.cached_property
  def timeline(self) -> Buffer: # [the signal, the value the last submitted batch signals]: zeroed host memory
    return Buffer(self.device, 2, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)

  def collect_prof(self):
    if PROFILE:
      es = list(self.prof_ents.items())
      sigs = [buf._buf.cpu_view().view(fmt='Q')[i]/decimal.Decimal(self.timestamp_divider) for (buf, _), e in es for i in (e.st_id, e.en_id)]
      Compiled.profile_events.append(ProfileGraphEvent([replace(e, st_id=2*i, en_id=2*i+1) for i,(_, e) in enumerate(es)], [], sigs))
    self.prof_ents.clear()

  def _at_profile_finalize(self): # the device clock against the host's: the median offset over a few tiny kernels
    from tinygrad.tensor import Tensor
    tdiffs = []
    for _ in range(5):
      with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
      if not (ents:=list(self.prof_ents.items())): return
      self.prof_ents.clear()
      st = perf_counter_us()
      self.synchronize()
      gpu = max(buf._buf.cpu_view().view(fmt='Q')[e.en_id] for (buf, _), e in ents)/decimal.Decimal(self.timestamp_divider)
      tdiffs.append((st+perf_counter_us())/2 - gpu)
    Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), self.device_props()))

  @functools.cache
  def rt_allocator(self, uncached:bool=True, host:bool=False) -> BumpAllocator: return BumpAllocator(self.rt_nbytes)

  @functools.cache
  def rt_buffer(self, uncached:bool=True, host:bool=False) -> Buffer:
    spec = BufferSpec(host=host, uncached=uncached, cpu_access=True)
    return Buffer(self.device, self.rt_allocator(uncached, host).size, dtypes.uint8, options=spec, preallocate=True)

  def rt_view(self, nbytes:int, dtype:DType=dtypes.uint8, uncached:bool=True, host:bool=False) -> Buffer: # a slot of the ring, wraps silently
    off = self.rt_allocator(uncached, host).alloc(max(nbytes, 1), alignment=128)
    return self.rt_buffer(uncached, host).view(nbytes // dtype.itemsize, dtype, off).ensure_allocated()

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done = time.perf_counter(), sig[0]
    while done < value:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif time.perf_counter() - st > (timeout or self.wait_timeout_ms) / 1000: self.on_device_hang()

  def synchronize(self, timeout:int|None=None):
    self._wait_signal(tl:=self.timeline._buf.cpu_view().view(fmt='Q'), tl[1], timeout)
    if self.prof_ents: self.collect_prof()

  def on_device_hang(self): raise RuntimeError(f"{self.device} hang detected")

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    super().finalize()

@dataclass
class HCQ2Buffer:
  va_addr:sint
  meta:Any=None
  view:MMIOInterface|None=None

  def offset(self, offset:int, size:int) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, meta=self.meta, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _as_buffer(self, buf:HCQBuffer) -> memoryview:
    return unwrap(buf.view).mv

  def _map(self, buf:HCQBuffer) -> HCQBuffer: # a mapping lives on the opaque, like hcq1: the lru hands the same one to many Buffers
    if self.dev not in buf.mapped_devs:
      if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
      buf.mappings[self.dev] = self._do_map(buf)
      buf.mapped_devs.append(self.dev)
    return buf.mappings[self.dev]

  def _do_unmap(self, mb): self.dev.iface.free(mb)

  @suppress_finalizing
  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    for dev in buf.mapped_devs: dev.synchronize()
    for d, mb in buf.mappings.items(): d.allocator._do_unmap(mb)
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer: return buf.offset(offset=offset, size=size)
