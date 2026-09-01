from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, TYPE_CHECKING
import functools, time, itertools, decimal
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, unwrap, PROFILE, VIZ
from tinygrad.helpers import to_tuple, ContextVar, Context, panic, partition
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer, DepsTracker
from tinygrad.device import ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, CallInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu
from tinygrad.uop.symbolic import const_arg
from tinygrad.dtype import dtypes, DType, AddrSpace
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
  queues:tuple[str, ...] = () # every queue the batch pushes on: the fence re-arms their signals
  estimates:Estimates = Estimates()

  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = ()

  args:tuple[tuple[UOp, UOp], ...] = () # placeholder -> the canonical body param it becomes, in call src order
  table:Any = None # the inputs table placeholder (its src position after lower)
  inputs:tuple[tuple[UOp, int, str], ...] = () # per table slot: (src, lane, device) the exec resolves to an address
  vals:tuple[tuple[str, int], ...] = () # bound values of the body variables (the queue byte size, table slots)

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_mstack(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.MSTACK: return tuple(x for s in u.src for x in unwrap_mstack(s))
  return unwrap_mstack(u.src[0]) if u.op is Ops.MSELECT else (u,)

def _lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

def to_name(*parts) -> str: # lowercase, "_"-joined: a device tuple gives its base name, a queue its type and index
  return "_".join(p for x in parts for p in (x[0].split(":")[:1] if isinstance(x, tuple) else str(x).split(":")) if p).lower()

def is_signal(tag) -> bool: return isinstance(tag, int) or tag.split("_")[0] in {"signal", "done", "timeline"}

def signal(devs, kind:str|int, queue:str="") -> UOp: # a u64 slot tagged {kind}_{device}_{queue}, an int kind is a profile timestamp
  tag = kind if isinstance(kind, int) else to_name(kind, to_tuple(devs), queue)
  return UOp.placeholder((1,), dtypes.uint64, 0, device=devs, volatile=True, tag=tag)

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp: # named submit_{base device}_{queue type}, the backend pm matches it
  return UOp.custom_function(to_name("submit", devs:=to_tuple(devs), queue.split(":")[0]), UOp(Ops.LINEAR, src=tuple(cmds), arg=(devs, queue)))

def make_call(name:str, body:UOp, info:HCQInfo) -> UOp: # a bare opaque CALL: .call would walk the whole body for its ranges assert
  return UOp(Ops.CALL, src=(body,), arg=CallInfo(None, name, False, False, info))

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

STAGING_SIZE, STAGING_SLOTS = 128 << 20, 2

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
# 1.2. prep: kernel copies

def _get_enqueue_devs(call:UOp) -> Any|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=call.src[1:]) or not all(all_devices_in(b.device, HCQ_DEVS) for b in bufs): return None
  if call.src[0].op is Ops.COPY: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: to_tuple(b.device)[0].startswith("CPU")).device # prio to enqueue on not CPU device
  # cpu-only calls don't batch: the host runs them synchronously, it has no queue (yet)
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
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    if isinstance(buf, UOp) and buf.op is Ops.MSELECT: buf = buf.src[0]
    return (buf.arg.slot, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...], str]]; profile:bool # noqa: E702 # (call, devices, queue) per enqueued call
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker); signal_tags:set[int] = field(default_factory=set) # noqa: E702

def _get_call_bufs_by_lane(call:UOp, devices:tuple[str, ...]) -> list[list[Any]]:
  def dep_buf(b:UOp) -> Any: return base if (base:=(b.src[0] if b.op is Ops.MSELECT else b).base).op is Ops.PARAM else b.buffer
  return [[dep_buf(_lane(a, lane)) for a in get_call_arg_uops(call)] for lane in range(len(devices))]

def _wait_ins(ctx:BatchCtx, bufs_by_lane:list[list[Any]], write, devices:tuple[str, ...], queue:str, tag:int) -> list[UOp]:
  latest:dict[tuple[str, str, int], int] = {} # (producer device, queue, consumer lane) -> the latest submit tag the lane waits on
  for lane, bufs in enumerate(bufs_by_lane):
    written = write if write is not None else list(range(len(bufs)))
    for d, q, t in ctx.tracker.access_resources(bufs, written, (devices[lane], queue, tag)):
      if t < tag and (d, q) != (devices[lane], queue): latest[(d, q, lane)] = max(latest.get((d, q, lane), 0), t) # same-queue submits are fifo
  ctx.signal_tags |= set(latest.values())
  waits = []
  for (d, q, lane), t in latest.items(): # the other lanes wait on a sentinel that is always signaled
    sig = UOp.mstack(*[signal(d, "signal", q) if i == lane else signal(dd, "sentinel") for i, dd in enumerate(devices)])
    waits.append(UOp(Ops.INS, arg=("wait", dtypes.void), src=(sig, UOp.const(t + 1, dtypes.uint64))))
  return waits

def _merge_queues(submits:list[UOp]) -> list[UOp]: # one submit per queue, the blocks only order through their gpu waits
  return [make_submit(*[c for s in submits if s.src[0].arg == k for c in s.src[0].src], devs=k[0], queue=k[1])
          for k in dedup([s.src[0].arg for s in submits])]

def _emit_submits(ctx:BatchCtx, call_waits:list[list[UOp]]) -> tuple[list[UOp], list[tuple]]:
  # one submit per call: timeline sync on first queue use, timestamps, the call, and a signal if someone waits on it
  src, kerns, seen_queues = [], [], set()
  for tag, ((call, devices, queue), q) in enumerate(zip(ctx.batch, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if (devices, queue) not in seen_queues:
      seen_queues.add((devices, queue))
      epoch = signal(devices, "value").index(0) - 1
      q = [UOp(Ops.INS, arg=("barrier", dtypes.void), src=()),
           UOp(Ops.INS, arg=("wait", dtypes.void), src=(signal(devices, "timeline"), epoch))] + q

    # and make hcq call
    name, info = get_call_name(call, get_call_arg_uops(call)), HCQInfo(devices, estimates=estimate_uop(call))
    ts_ids = [next(UOp.unique_num) for _ in range(2)] if ctx.profile else []
    kerns.append((devices, name, info.estimates, tuple(ts_ids), getattr(call.src[0].arg, "profile_key", None)))

    ts_ins = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(signal(devices, s),)) for s in ts_ids]
    q += ts_ins[:1] + [call.replace(arg=replace(call.arg, aux=info))] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags:
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(signal(devices, "signal", queue), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_submit(*q, devs=devices, queue=queue))
  return src, kerns

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...], str]], profile:bool) -> list[UOp]:
  if not batch: return []
  ctx, devices = BatchCtx(batch, profile), tuple(dedup([d for _, devs, _ in batch for d in devs]))
  call_waits = [_wait_ins(ctx, _get_call_bufs_by_lane(c, d), get_call_outs_ins(c)[0], d, q, tag) for tag, (c, d, q) in enumerate(batch)]
  submits, kerns = _emit_submits(ctx, call_waits)

  for devs in [tuple(g) for _, g in itertools.groupby(sorted(devices), key=lambda d: d.split(":")[0])]:
    queues = dedup([q for _, bdevs, q in batch if set(bdevs) & set(devs)])
    submits += [make_submit(UOp(Ops.INS, arg=("store", dtypes.void), src=(signal(devs, "done", q), UOp.const(1, dtypes.uint64))),
                            devs=devs, queue=q) for q in queues]
    submits += [make_submit(*[UOp(Ops.INS, arg=("wait", dtypes.void), src=(signal(devs, "done", q), UOp.const(1, dtypes.uint64))) for q in queues],
                UOp(Ops.INS, arg=("store", dtypes.void), src=(signal(devs, "timeline"), signal(devs, "value").index(0).load())),
                devs=devs, queue="COMPUTE:0" if len(queues) > 1 else queues[0])]

  fence = UOp.custom_function("hcq_fence")
  merged = [m.replace(src=(m.src[0].barrier(fence),)) for m in _merge_queues(submits)]
  fin = UOp.custom_function("hcq_finalizer")

  estimates = sum((estimate_uop(call) for call, _, _ in batch), start=Estimates()).simplify()
  return [make_call(f"hcq batch ({len(batch)})", UOp.sink(*merged, fin),
                    HCQInfo(devices, tuple(dedup([q for _, _, q in batch])), estimates, kernels=tuple(kerns)))]

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...], str]] = []
  for call in l.src:
    if (devs:=_get_enqueue_devs(call)) is not None: batch.append((call, to_tuple(devs), "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0"))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 3. encode

@dataclass
class EncodeCtx: # devs/queue plus everything on the device; the body's buffers register here as they are created
  dev:Any; devs:tuple[str, ...]; queue:str; queues:tuple[str, ...] = () # noqa: E702
  args:list[UOp] = field(default_factory=list)
  blobs:dict[UOp, tuple[bytes, list[tuple[int, UOp]]]] = field(default_factory=dict) # cmdbuf placeholder -> (bytes, link words at offsets)
  def __getattr__(self, name): return getattr(self.dev, name)

  def new_arg(self, u:UOp) -> UOp:
    if u not in self.args: self.args.append(u)
    return u

  def new_buf(self, shape:tuple[int, ...], dtype:DType, tag:str) -> UOp: # a scratch buffer owned by this linear
    return self.new_arg(UOp.placeholder(shape, dtype, next(UOp.unique_num), device=self.devs, tag=tag))

  def new_signal(self, kind:str, queue:str="") -> UOp: return self.new_arg(signal(self.devs, kind, queue))

class HWQueue: # a renderer for queues: q_rewrite turns each submit op into words, the blob packs them as they come
  q_rewrite:PatternMatcher

  def __init__(self, ctx:EncodeCtx):
    self.ctx, self.blob, self.patches, self.deps = ctx, bytearray(), list[tuple[int, UOp]](), list[UOp]()
  def __getattr__(self, name): return getattr(self.ctx, name)

  def encode(self, submit:UOp) -> UOp: return self.submit(self.render(submit.src[0]))

  def q(self, *words): # a word is an int, a const uop, or a patch uop the blob keeps a slot for
    for w in words:
      c = w
      while isinstance(c, UOp) and c.op is Ops.CAST: c = c.src[0]
      if isinstance(c, UOp) and c.op is not Ops.CONST:
        self.patches.append((len(self.blob), w))
        self.blob += bytes(w.dtype.itemsize)
      else:
        v, n = (c.val, w.dtype.itemsize) if isinstance(w, UOp) else (c, 4)
        self.blob += (v & (1 << 8 * n) - 1).to_bytes(n, 'little')

  def render(self, lin:UOp) -> UOp: # the blob bakes on a fresh placeholder, shrunk to the stream the submit copies
    if lin.op is Ops.BARRIER: self.deps, lin = list(lin.src[1:]), lin.src[0] # the barrier chains this submit on the prior phase
    self.devs, self.queue = lin.arg
    for u in lin.src:
      if u.op in {Ops.INS, Ops.CALL}: self.q_rewrite.rewrite(u, ctx=self)
      else: self.deps.append(u) # lowered sync blocks this submit pushes after

    # nested word linears (kernargs) pack into the tail, their getaddrs re-target to views of the buffer
    stream, offs = len(self.blob), {}
    for l in dedup([g.src[0] for _, w in self.patches for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR]):
      self.blob += bytes(-len(self.blob) % 128)
      start = len(self.blob)
      self.q(*l.src)
      offs[l] = (start, len(self.blob))

    # the blob is plain bytes on a fresh placeholder: the link words ride the ctx as (offset, word) rows,
    # the runtime ones the body stores every call, before the submit reads the stream
    buf = self.ctx.new_buf((len(self.blob),), dtypes.uint8, to_name("cmdbuf", self.devs, self.queue))
    placed = UOp(Ops.SINK, src=tuple(w for _, w in self.patches)).substitute({l: buf[o:e] for l, (o, e) in offs.items()}).src
    links, runtime = partition(list(zip([o for o, _ in self.patches], placed)), lambda p: is_link_value(p[1]))
    self.ctx.blobs[buf] = (bytes(self.blob), links)
    stores = [buf.after(*self.deps).shrink(((o, o + w.dtype.itemsize),)).bitcast(w.dtype).index(0).store(w) for o, w in runtime]
    return buf.after(*stores).shrink(((0, stream),))

  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")

def is_link_value(w:UOp) -> bool: # resolvable when the linear links: no variables, memory reads, or value params
  if w.op is Ops.GETADDR: return True
  if w.op in {Ops.LOAD, Ops.INDEX, Ops.PARAM} or w.is_variable: return False
  return all(is_link_value(s) for s in w.src)

def hcq_fence(ctx:EncodeCtx, f:UOp) -> UOp:
  done = ctx.new_signal("timeline").after(loop:=UOp.loop(0)).index(0).load()
  rst = (done.end(loop, done < ctx.new_signal("epoch").index(0).load()),)
  for q, kind in itertools.product(ctx.queues, ("signal", "done")):
    rst = (ctx.new_signal(kind, q).after(*rst).index(0).store(0),)
  return rst[0].barrier()

def hcq_finalizer(ctx:EncodeCtx, fin:UOp) -> UOp:
  epoch = (epoch_slot:=ctx.new_signal("value").index(0)).load()
  return UOp.barrier(epoch_slot.store(epoch + UOp.const(1, dtypes.uint64)), ctx.new_signal("epoch").index(0).store(epoch))

pm_hcq_encode = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence),
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_finalizer", name="fin"), hcq_finalizer),
])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo) or call.arg.aux.args: return None # not an hcq call, or lowered already

  # one ctx for the whole call: the body encodes with the hcq rules plus every device's own
  ctx = EncodeCtx(Device[(devs:=call.arg.aux.device)[0]], devs, "", call.arg.aux.queues)
  pms = [Device[d].pm_encode for d in dedup([d.split(":")[0] for d in devs])]
  body = graph_rewrite(call.src[0], functools.reduce(lambda a, b: a + b, pms, pm_hcq_encode), ctx=ctx, walk=True, name="encode body")

  # the blob and the link words ride the call arg as stores: the bytes, then one stacked store per word width at byte offsets
  patched:dict[UOp, UOp] = {}
  for cmdbuf, (blob, links) in ctx.blobs.items():
    stores = []
    for _, grp in itertools.groupby(sorted(links, key=lambda p: p[1].dtype.itemsize), key=lambda p: p[1].dtype.itemsize):
      offs, words = zip(*grp)
      stores.append(cmdbuf.index(UOp.stack(*[UOp.const(o) for o in offs])).store(UOp.stack(*words)))
    patched[cmdbuf] = cmdbuf.after(cmdbuf.store(UOp(Ops.BINARY, arg=blob).bitcast(cmdbuf.dtype)), *stores)

  # placeholders become canonical params of the body program, the call binds them in the same order
  tops = body.toposort(gate=lambda u: u.op is not Ops.PARAM)
  placeholders, alus = partition(dedup([s for u in tops for s in u.src if s.op is Ops.PARAM]), lambda s: s.addrspace is AddrSpace.GLOBAL)
  args = {b: UOp.param(i, b.dtype, shape=b.shape, device=HCQ_RUNTIME_DEV.value, volatile=b.arg.volatile, name=b.arg.name)
          for i, b in enumerate(placeholders)}
  # ALU params (variable values) bind by name at exec, re-slotted after the buffers so they render last
  vals = {a: a.replace(arg=replace(a.arg, slot=len(placeholders) + j)) for j, a in enumerate(alus)}
  rngs = {r: r.replace(arg=(i,)+r.arg[1:]) for i, r in enumerate(sorted([u for u in tops if u.op is Ops.RANGE], key=lambda r: r.arg))}
  sink = body.substitute(cast(dict[UOp, UOp], args) | vals | rngs).replace(arg=KernelInfo("hcq_submit"), tag=1)
  if VIZ: graph_rewrite(UOp.sink(*patched.values()), PatternMatcher([]), name="View Link-Time Patches")
  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Body")
  return call.replace(src=(sink, *[patched.get(b, b) for b in placeholders]),
                      arg=replace(call.arg, aux=replace(call.arg.aux, args=tuple(args.items()))))

pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

hcq_compile_cache:dict[tuple[UOp, bool], UOp] = {} # uops are hash-consed: the linear itself is the key

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear, profile)))) is None:
    # prep: only the schedule tracks deps on the real buffers, they go right back to params so the cache stays input-agnostic
    bufmap = {s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}
    linear = graph_rewrite(linear.substitute(bufmap, walk=True), pm_insert_copy_staging+pm_flatten_linear, name="insert copy staging")

    linear = sched_batches(linear, profile)
    if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View Schedule")
    linear = graph_rewrite(linear, pm_encode, walk=True, name="encode")

    # the unbind comes after encode: staged-copy bodies are opaque to substitute, encode lowers them into reachable getaddrs
    with Context(EMULATED_DTYPES=""):
      final_linear = hcq_compile_cache[cache_key] = lower_and_compile(linear).substitute({v: k for k,v in bufmap.items()}, walk=True)
  # bind this call's buffers: every input getaddr resolves at link
  return final_linear.substitute({s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}, walk=True)


# *****************
# 9. bufferize placeholders: replace placeholders with real buffers

def bufferize_buf(buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.mstack(*(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=dv), HCQ_RUNTIME_DEV.value) for dev in to_tuple(buf.device)))

# *****************
# 10. link: bufferize the placeholders, resolve the addresses, fold the words, the stores write the bytes

def _bufs(buf:UOp) -> list[Buffer]: # one Buffer per lane
  if buf.op is Ops.MSTACK: return [cast(Buffer, x.buffer) for x in buf.src]
  return list(m.bufs) if isinstance(m:=buf.buffer, MultiBuffer) else [m]

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, byte offset)
  if v.op in (Ops.BITCAST, Ops.AFTER): return unwrap_view(v.src[0])
  if v.op is not Ops.SHRINK: return v, 0
  base, off = unwrap_view(v.src[0])
  return base, off + v.src[1].val * v.dtype.itemsize

def resolve_getaddr(ctx:list[UOp], g:UOp) -> UOp|None: # the address per lane once the base is a real buffer
  buf, off = unwrap_view(g.src[0])
  if buf.op not in {Ops.BUFFER, Ops.MSTACK, Ops.MSELECT}: return None
  ctx.append(buf) # the address bakes into the blob, the linked linear refholds the buffer (amd scratch outlives its realloc)
  devs, bufs = to_tuple(g.arg), _bufs(buf)
  if len(bufs) == 1: bufs = bufs * len(devs) # one buffer shared by every lane
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = [x.get_buf(d).va_addr + off for x, d in zip(bufs, devs)]
  return UOp.const(addrs[0] if len(addrs) == 1 else tuple(addrs), dtypes.uint64)

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in _bufs(buf):
    if getattr(b, '_hcq_written', None) is not blob.arg: # programs are shared across linears, write them once
      cast(Any, b.ensure_allocated())._hcq_written = blob.arg
      b._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_words(buf:UOp, offs:UOp, words:UOp) -> UOp: # every word folded to a const (a lane stack on multi): write it at its byte offset
  for lane, b in enumerate(_bufs(buf)):
    mv = b.ensure_allocated()._buf.cpu_view().view(fmt='B')
    for o, w in zip(offs.src, words.src):
      v = w.src[lane] if w.op is Ops.STACK else w
      mv[o.val:o.val + w.dtype.itemsize] = (v.val & (1 << 8 * w.dtype.itemsize) - 1).to_bytes(w.dtype.itemsize, 'little')
  return UOp(Ops.NOOP)

def fold_alu(a:UOp) -> UOp: return UOp.const(exec_alu(a.op, a.dtype, [const_arg(s) for s in a.src], False), a.dtype)

word = UPat.any(UPat.cvar().or_casted(), UPat(Ops.STACK, src=UPat.cvar().or_casted())) # a const, or one per lane
pm_link = PatternMatcher([
  (UPat(Ops.PARAM, name="buf"), bufferize_buf),
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  (UPat(GroupOp.ALU, src=word, name="a"), fold_alu),
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat(name="buf").index(UPat(Ops.STACK, src=word, name="offs")).store(UPat(Ops.STACK, src=word, name="words")), fold_words),
  # written stores fold away, anything else left is a word the link couldn't resolve
  (UPat(Ops.AFTER, name="a"),
   lambda a: a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

link_linear_cache:dict[UOp, UOp] = {}

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear)) is not None: return linked
  refs:list[UOp] = []
  linked = graph_rewrite(linear, pm_link, ctx=refs, bottom_up=False, name="link")
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

    dev = (device,)
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag=to_name("sentinel", dev), name="b"), lambda ctx, b: ctx.signal(b.tag, (1 << 64) - 1)),
      (UPat(Ops.PARAM, tag=to_name("value", dev), name="b"), lambda ctx, b: ctx.signal(b.tag, 1, device="CPU")),
      (UPat(Ops.PARAM, tag=to_name("epoch", dev), name="b"), lambda ctx, b: ctx.signal(b.tag, device="CPU")),
      (UPat(Ops.PARAM, tag="program", name="b"), lambda ctx, b: ctx.prog_buffer(b)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.signal(b.tag) if is_signal(b.tag) else ctx.new_buffer(b)), # signals by tag, else the pool
    ])
    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.rt_allocator = BumpAllocator(self.rt_nbytes)
    self.prog_bufs:dict[UOp, Buffer] = {}
    self.prof_ents:dict[int, ProfileGraphEntry] = {}

  def collect_prof(self):
    if PROFILE:
      es = list(self.prof_ents.values())
      sigs = [self.signal(i)._buf.cpu_view().view(fmt='Q')[0]/decimal.Decimal(self.timestamp_divider) for e in es for i in (e.st_id, e.en_id)]
      Compiled.profile_events.append(ProfileGraphEvent([replace(e, st_id=2*i, en_id=2*i+1) for i,e in enumerate(es)], [], sigs))
    self.prof_ents.clear()

  def _at_profile_finalize(self):
    pass
    # from tinygrad.tensor import Tensor
    # tdiffs = []
    # for _ in range(5):
    #   with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
    #   if not (ents:=list(self.prof_ents.values())): return
    #   self.prof_ents.clear()
    #   st = perf_counter_us()
    #   self.synchronize()
    #   gpu = max(self.signal(e.en_id)._buf.cpu_view().view(fmt='Q')[0] for e in ents)/decimal.Decimal(self.timestamp_divider)
    #   tdiffs.append((st+perf_counter_us())/2 - gpu)
    # Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), self.device_props()))

  @functools.cache
  def rt_buffer(self, uncached:bool=True) -> Buffer:
    return Buffer(self.device, self.rt_allocator.size, dtypes.uint8, options=BufferSpec(uncached=uncached, cpu_access=True), preallocate=True)

  def rt_view(self, nbytes:int, dtype:DType=dtypes.uint8, uncached:bool=True) -> Buffer:
    return self.rt_buffer(uncached).view(nbytes // dtype.itemsize, dtype, self.rt_allocator.alloc(max(nbytes, 1), alignment=128)).ensure_allocated()

  def prog_buffer(self, b:UOp) -> Buffer: # program buffers are shared across linears, keyed on the placeholder
    if (buf:=self.prog_bufs.get(b)) is None:
      buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
    return buf

  def new_buffer(self, b:UOp) -> Buffer: return self.rt_view(b.max_numel() * b.dtype.itemsize, b.dtype)

  @functools.cache
  def signal(self, name:str|int, init_value:int=0, device:str|None=None) -> Buffer:
    buf = Buffer(device or self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf._buf.cpu_view().view(fmt='Q')[0] = init_value
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done = time.perf_counter(), sig[0]
    while done < value:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif time.perf_counter() - st > (timeout or self.wait_timeout_ms) / 1000: self.on_device_hang()

  def synchronize(self, timeout:int|None=None):
    sig = self.signal(to_name("timeline", (self.device,)))._buf.cpu_view().view(fmt='Q')
    tl = self.signal(to_name("value", (self.device,)), 1, device="CPU")._buf.cpu_view().view(fmt='Q')
    self._wait_signal(sig, tl[0] - 1, timeout)
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

  def _map(self, buf:HCQBuffer) -> HCQBuffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  def _do_unmap(self, mb): self.dev.iface.free(mb)

  @suppress_finalizing
  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    self.dev.synchronize()
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self._do_unmap(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer: return buf.offset(offset=offset, size=size)
