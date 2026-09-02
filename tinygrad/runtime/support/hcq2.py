from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, TYPE_CHECKING
import functools, time, itertools, decimal, weakref, collections, os
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, unwrap, PROFILE, VIZ
from tinygrad.helpers import to_tuple, ContextVar, Context, panic, partition
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator
from tinygrad.device import ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, CallInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu
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
  estimates:Estimates = Estimates()

  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = ()

  args:tuple[UOp, ...] = () # the placeholders the call binds, in the body's param order
  table:int = -1 # the inputs table's position in the args: the exec fills it with the address of every (input, device) in inputs
  inputs:tuple[tuple[UOp, str], ...] = ()

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, byte offset)
  if v.op in (Ops.BITCAST, Ops.AFTER): return unwrap_view(v.src[0])
  if v.op is not Ops.SHRINK: return v, 0
  base, off = unwrap_view(v.src[0])
  return base, off + v.src[1].val * v.dtype.itemsize

def _lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

def to_name(*parts) -> str: # lowercase, "_"-joined: "AMD:1" gives amd_1, a queue its type and index, a device tuple its first device
  return "_".join(p for x in parts for p in str(x[0] if isinstance(x, tuple) else x).split(":") if p).lower()

def is_signal(tag) -> bool: return isinstance(tag, int) or tag.split("_")[0] in {"signal", "done", "timeline"}

def signal(devs, kind:str|int, queue:str="") -> UOp: # a u64 slot tagged {kind}_{device}_{queue}, an int kind is a profile timestamp
  tag = kind if isinstance(kind, int) else to_name(kind, to_tuple(devs), queue)
  return UOp.placeholder((1,), dtypes.uint64, 0, device=to_tuple(devs), volatile=True, tag=tag)

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp: # named submit_{base device}_{queue type}, the backend pm matches it
  fn = to_name("submit", (devs:=to_tuple(devs))[0].split(":")[0], queue.split(":")[0])
  return UOp.custom_function(fn, UOp(Ops.LINEAR, src=tuple(cmds), arg=(devs, queue)))

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
  return UOp(Ops.LINEAR, src=tuple(call.replace(src=(call.src[0], *[a if a.is_bound_var else _lane(a, i) for a in call.src[1:]], dnum.bind(i)))
                                   for i in range(n)))
pm_unwrap_multi = PatternMatcher([(UPat(Ops.CALL, name="call"), unwrap_call)])

# *****************
# 1.3. prep: kernel copies

def _get_enqueue_devs(call:UOp) -> Any|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=get_call_arg_uops(call)) or not all(all_devices_in(b.device, HCQ_DEVS) for b in bufs): return None
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

class HCQDepsTracker: # per buffer base, the last write and the last read of every (dev, queue): fifo queues imply the earlier ones
  def __init__(self):
    self.writes:dict[Any, dict[Any, Any]] = collections.defaultdict(dict)
    self.reads:dict[Any, dict[Any, Any]] = collections.defaultdict(dict)

  @staticmethod
  def _key(buf:Any) -> Any: # a real buffer is its base, a param is its slot (per lane once mselected)
    if not isinstance(buf, UOp): return id(buf.base)
    return (buf.src[0].base.arg.slot, buf.arg) if buf.op is Ops.MSELECT else buf.arg.slot

  def access_resources(self, bufs:list[Any], write:list[int], dep:tuple[str, str, int]) -> list[tuple[str, str, int]]:
    waits:list[Any] = [] # a read waits for the writes, a write also for the reads: a superset of the exact ranges, never a miss
    for i, key in enumerate(map(self._key, bufs)):
      waits += list(self.writes[key].values()) + (list(self.reads[key].values()) if i in write else [])
      (self.writes if i in write else self.reads)[key][dep[:2]] = dep
    return dedup(waits)

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...], str]]; profile:bool # noqa: E702 # (call, devices, queue) per enqueued call
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker); signal_tags:set[int] = field(default_factory=set) # noqa: E702
  uid:int = field(default_factory=lambda: next(UOp.unique_num)) # the batch's slot number: it owns its signals like an hcq1 graph does

  def signal(self, devs, kind:str, queue:str) -> UOp:
    return UOp.placeholder((1,), dtypes.uint64, self.uid, device=to_tuple(devs), volatile=True, tag=to_name(kind, to_tuple(devs), queue))

def _call_bufs(call:UOp) -> list[Any]: # the dep resources: a param (or its mselect lane) as is, anything real as its Buffer
  def dep_buf(b:UOp) -> Any:
    base = (b.src[0] if b.op is Ops.MSELECT else b).base
    return (b if b.op is Ops.MSELECT else base) if base.op is Ops.PARAM else b.buffer
  return [dep_buf(a) for a in get_call_arg_uops(call)]

def _wait_ins(ctx:BatchCtx, call:UOp, device:str, queue:str, tag:int) -> list[UOp]:
  bufs, write = _call_bufs(call), get_call_outs_ins(call)[0]
  latest:dict[tuple[str, str], int] = {} # (producer device, queue) -> the latest submit tag to wait on, same-queue submits are fifo
  for d, q, t in ctx.tracker.access_resources(bufs, list(range(len(bufs)) if write is None else write), (device, queue, tag)):
    if t < tag and (d, q) != (device, queue): latest[(d, q)] = max(latest.get((d, q), 0), t)
  ctx.signal_tags |= set(latest.values())
  return [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.signal(d, "signal", q), UOp.const(t + 1, dtypes.uint64))) for (d, q), t in latest.items()]

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
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.signal(devices, "signal", queue), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_submit(*q, devs=devices, queue=queue))
  return src, kerns

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...], str]], profile:bool) -> list[UOp]:
  if not batch: return []
  ctx, devices = BatchCtx(batch, profile), tuple(dedup([d for _, devs, _ in batch for d in devs]))
  call_waits = [_wait_ins(ctx, c, d[0], q, tag) for tag, (c, d, q) in enumerate(batch)]
  submits, kerns = _emit_submits(ctx, call_waits)

  sigs:list[UOp] = [] # every slot the batch stores to, the fence re-arms them
  for dev in devices: # per device: every queue flags done, then one queue bumps the timeline
    queues = dedup([q for _, bdevs, q in batch if dev in bdevs])
    sigs += [ctx.signal(dev, kind, q) for q in queues for kind in ("signal", "done")]
    submits += [make_submit(UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.signal(dev, "done", q), UOp.const(1, dtypes.uint64))),
                            devs=dev, queue=q) for q in queues]
    submits += [make_submit(*[UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.signal(dev, "done", q), UOp.const(1, dtypes.uint64))) for q in queues],
                UOp(Ops.INS, arg=("store", dtypes.void), src=(signal(dev, "timeline"), signal(dev, "value").index(0).load())),
                devs=dev, queue="COMPUTE:0" if len(queues) > 1 else queues[0])]

  fence = UOp.custom_function("hcq_fence", *sigs)
  merged = [m.replace(src=(m.src[0].barrier(fence),)) for m in _merge_queues(submits)]
  fin = UOp.custom_function("hcq_finalizer")

  estimates = sum((estimate_uop(call) for call, _, _ in batch), start=Estimates()).simplify()
  return [make_call(f"hcq batch ({len(batch)})", UOp.sink(*merged, fin),
                    HCQInfo(devices, estimates, kernels=tuple(kerns)))]

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...], str]] = []
  for call in l.src:
    if (devs:=_get_enqueue_devs(call)) is not None: # canonical device names: "AMD:0" and "AMD" are one device with one set of signals
      batch.append((call, tuple(Device.canonicalize(d) for d in to_tuple(devs)), "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0"))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 3. encode

@dataclass
class EncodeCtx: # the batch's devices, the blobs its queues render and the inputs its body reads
  devs:tuple[str, ...]
  blobs:dict[UOp, tuple[bytes, list[tuple[int, UOp]]]] = field(default_factory=dict) # cmdbuf placeholder -> (bytes, link words at offsets)
  inputs:list[tuple[UOp, str]] = field(default_factory=list) # (input, device) per slot of the table the body reads addresses from
  table:UOp = field(default_factory=lambda: UOp.placeholder((1,), dtypes.uint64, device="CPU", tag="inputs")) # sized once encoded

  def new_buf(self, devs:tuple[str, ...], shape:tuple[int, ...], dtype:DType, tag:str) -> UOp: # a scratch buffer owned by this linear
    return UOp.placeholder(shape, dtype, next(UOp.unique_num), device=devs, tag=tag)

class HWQueue: # a renderer for queues: q_rewrite turns each submit op into words, the blob packs them as they come
  q_rewrite:PatternMatcher

  def __init__(self, ctx:EncodeCtx):
    self.ctx, self.blob, self.patches, self.deps = ctx, bytearray(), list[tuple[int, UOp]](), list[UOp]()
  def __getattr__(self, name): return getattr(self.dev, name) # the hardware bits of the submit's device

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
    self.dev = Device[self.devs[0]]
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
    buf = self.ctx.new_buf(self.devs, (len(self.blob),), dtypes.uint8, to_name("cmdbuf", self.devs, self.queue))
    placed = UOp(Ops.SINK, src=tuple(w for _, w in self.patches)).substitute({l: buf[o:e] for l, (o, e) in offs.items()})
    placed = graph_rewrite(placed, pm_input_addr, ctx=self.ctx, name="input addrs").src
    links, runtime = partition(list(zip([o for o, _ in self.patches], placed)), lambda p: is_link_value(p[1]))
    self.ctx.blobs[buf] = (bytes(self.blob), links)
    stores = [buf.after(*self.deps).shrink(((o, o + w.dtype.itemsize),)).bitcast(w.dtype).index(0).store(w) for o, w in runtime]
    return buf.after(*stores).shrink(((0, stream),))

  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")

def input_addr(ctx:EncodeCtx, g:UOp) -> UOp|None: # an input's address is only known at exec: the body reads it from the table
  base, off = unwrap_view(g.src[0])
  if (param:=base.src[0].base if base.op is Ops.MSELECT else base).op is not Ops.PARAM or param.tag is not None: return None
  if (key:=(base, to_tuple(g.arg)[0])) not in ctx.inputs: ctx.inputs.append(key)
  return ctx.table.index(ctx.inputs.index(key)).load() + UOp.const(off, dtypes.uint64)
pm_input_addr = PatternMatcher([(UPat(Ops.GETADDR, name="g"), input_addr)])

def is_link_value(w:UOp) -> bool: # resolvable when the linear links: no variables, memory reads, or value params
  if w.op is Ops.GETADDR: return True
  if w.op in {Ops.LOAD, Ops.INDEX, Ops.PARAM} or w.is_variable: return False
  return all(is_link_value(s) for s in w.src)

def hcq_fence(ctx:EncodeCtx, f:UOp) -> UOp: # spin until every device's timeline reaches its last epoch, then re-arm the batch's signals
  last:tuple[UOp, ...] = ()
  for i, dev in enumerate(ctx.devs):
    done = signal(dev, "timeline").after(*last, loop:=UOp.loop(i)).index(0).load()
    last = (done.end(loop, done < signal(dev, "epoch").index(0).load()),)
  for sig in f.src: last = (sig.after(*last).index(0).store(0),)
  return last[0].barrier()

def hcq_finalizer(ctx:EncodeCtx, fin:UOp) -> UOp: # per device: bump the epoch the next fence waits for
  stores = []
  for dev in ctx.devs:
    epoch = (epoch_slot:=signal(dev, "value").index(0)).load()
    stores += [epoch_slot.store(epoch + UOp.const(1, dtypes.uint64)), signal(dev, "epoch").index(0).store(epoch)]
  return UOp.barrier(*stores)

pm_hcq_encode = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence),
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_finalizer", name="fin"), hcq_finalizer),
])

# *****************
# 3.1. split: the body's placeholders become its params in visit order, a cmdbuf as its patched form; its variables and ranges collect

@dataclass
class SplitCtx:
  blobs:dict[UOp, tuple[bytes, list[tuple[int, UOp]]]]
  args:list[UOp] = field(default_factory=list); alus:list[UOp] = field(default_factory=list); ranges:list[UOp] = field(default_factory=list) # noqa: E702

def patch_cmdbuf(cmdbuf:UOp, blob:bytes, links:list[tuple[int, UOp]]) -> UOp: # the bytes, then one stacked store per word width at byte offsets
  stores = []
  for _, grp in itertools.groupby(sorted(links, key=lambda p: p[1].dtype.itemsize), key=lambda p: p[1].dtype.itemsize):
    offs, words = zip(*grp)
    stores.append(cmdbuf.index(UOp.stack(*[UOp.const(o) for o in offs])).store(UOp.stack(*words)))
  return cmdbuf.after(cmdbuf.store(UOp(Ops.BINARY, arg=blob).bitcast(cmdbuf.dtype)), *stores)

def split_param(ctx:SplitCtx, p:UOp) -> UOp|None: # a tagged param is a placeholder: the body's next param, bound by the call
  if p.addrspace is AddrSpace.ALU: ctx.alus.append(p)
  if p.tag is None: return None
  ctx.args.append(patch_cmdbuf(p, *ctx.blobs[p]) if p in ctx.blobs else p)
  return UOp.param(len(ctx.args) - 1, p.dtype, shape=p.shape, device=HCQ_RUNTIME_DEV.value, volatile=p.arg.volatile, name=p.arg.name)

def split_range(ctx:SplitCtx, r:UOp) -> None: ctx.ranges.append(r)

pm_split_body = PatternMatcher([(UPat(Ops.PARAM, name="p"), split_param), (UPat(Ops.RANGE, name="r"), split_range)])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo) or call.arg.aux.args: return None # not an hcq call, or lowered already

  # one ctx for the whole call: the body encodes with the hcq rules plus every device's own
  ctx = EncodeCtx(devs:=call.arg.aux.device)
  pms = [Device[d].pm_encode for d in dedup([d.split(":")[0] for d in devs])]
  body = graph_rewrite(call.src[0], functools.reduce(lambda a, b: a + b, pms, pm_hcq_encode), ctx=ctx, walk=True, name="encode body")

  # the table is sized now that every input is known, then the placeholders split out as the body's params
  table = UOp.placeholder((len(ctx.inputs),), dtypes.uint64, device="CPU", tag="inputs")
  body = graph_rewrite(body.substitute({ctx.table: table}), pm_split_body, split:=SplitCtx(ctx.blobs), name="split body")
  # a variable binds by name at exec: every program's copy of it is one param after the buffers; the ranges renumber in order
  names = dedup([a.arg.name for a in split.alus])
  vals = {a: a.replace(arg=replace(a.arg, slot=len(split.args) + names.index(a.arg.name))) for a in split.alus}
  rngs = {r: r.replace(arg=(i,)+r.arg[1:]) for i, r in enumerate(sorted(split.ranges, key=lambda r: r.arg))}
  sink = body.substitute(vals | rngs).replace(arg=KernelInfo("hcq_submit"), tag=1)
  if VIZ: graph_rewrite(UOp.sink(*split.args), PatternMatcher([]), name="View Link-Time Patches")
  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Body")
  info = replace(call.arg.aux, args=tuple(split.args), table=split.args.index(table) if ctx.inputs else -1, inputs=tuple(ctx.inputs))
  return call.replace(src=(sink, *split.args), arg=replace(call.arg, aux=info))

pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

hcq_compile_cache:dict[tuple[UOp, bool, bool], UOp] = {} # uops are hash-consed: the linear itself is the key, plus whether inputs bind

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  # the schedule and encode see the real buffers: their views are the deps, their addresses bake at link. unbound params (the jit's
  # inputs) are read from the inputs table at exec instead. the cache holds the input-agnostic param form, each call binds its own
  bufmap = {s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}
  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear, profile, input_uops is None)))) is None:
    linear = graph_rewrite(linear.substitute(bufmap, walk=True), pm_unwrap_multi+pm_insert_copy_staging+pm_flatten_linear, name="prep calls")
    linear = sched_batches(linear, profile)
    if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View Schedule")
    linear = graph_rewrite(linear, pm_encode, walk=True, name="encode")
    with Context(EMULATED_DTYPES=""):
      final_linear = hcq_compile_cache[cache_key] = lower_and_compile(linear).substitute({v: k for k, v in bufmap.items()}, walk=True)
  return final_linear.substitute(bufmap, walk=True)


# *****************
# 9. bufferize placeholders: replace placeholders with real buffers

def bufferize_buf(buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.from_buffer((dev:=Device[to_tuple(buf.device)[0]]).pm_bufferize.rewrite(buf, ctx=dev), HCQ_RUNTIME_DEV.value)

# *****************
# 10. link: bufferize the placeholders, resolve the addresses, fold the words, the stores write the bytes

def resolve_getaddr(ctx:list[UOp], g:UOp) -> UOp|None: # the address once the base is a real buffer
  buf, off = unwrap_view(g.src[0])
  if buf.op not in {Ops.BUFFER, Ops.MSELECT}: return None
  ctx.append(buf) # the address bakes into the blob: the linked linear refholds the buffer (the amd scratch outlives its realloc)
  return UOp.const(cast(Buffer, buf.buffer).get_buf(to_tuple(g.arg)[0]).va_addr + off, dtypes.uint64)

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  if getattr(b:=cast(Buffer, buf.buffer), '_hcq_written', None) is not blob.arg: # programs are shared across linears, write them once
    cast(Any, b.ensure_allocated())._hcq_written = blob.arg
    b._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_words(buf:UOp, offs:UOp, words:UOp) -> UOp: # every word folded to a const: write it at its byte offset
  mv = cast(Buffer, buf.buffer).ensure_allocated()._buf.cpu_view().view(fmt='B')
  for o, w in zip(offs.src, words.src):
    n = w.dtype.itemsize
    mv[o.val:o.val + n] = (w.val & (1 << 8 * n) - 1).to_bytes(n, 'little')
  return UOp(Ops.NOOP)

def fold_alu(a:UOp) -> UOp: return UOp.const(exec_alu(a.op, a.dtype, [s.val for s in a.src], False), a.dtype)

word = UPat.cvar().or_casted()
pm_link = PatternMatcher([
  (UPat(Ops.PARAM, name="buf"), bufferize_buf),
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  (UPat(GroupOp.ALU, src=word, name="a"), fold_alu),
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat(name="buf").index(UPat(Ops.STACK, src=word, name="offs")).store(UPat(Ops.STACK, src=word, name="words")), fold_words),
  # written stores fold away, anything else left (but a bound variable) is a word the link couldn't resolve
  (UPat(Ops.AFTER, name="a"), lambda a: None if a.is_bound_var else a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else
   panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

link_linear_cache:weakref.WeakKeyDictionary[UOp, UOp] = weakref.WeakKeyDictionary() # a baked link lives as long as its bound linear

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
      (UPat(Ops.PARAM, tag=to_name("value", dev), name="b"), lambda ctx, b: ctx.signal(b.tag, 1, device="CPU")),
      (UPat(Ops.PARAM, tag=to_name("epoch", dev), name="b"), lambda ctx, b: ctx.signal(b.tag, device="CPU")),
      (UPat(Ops.PARAM, tag="program", name="b"), lambda ctx, b: ctx.prog_buffer(b)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.signal(b.tag) if is_signal(b.tag) and not b.arg.slot else ctx.new_buffer(b)),
    ])
    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

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
  def rt_allocator(self, uncached:bool=True, host:bool=False) -> BumpAllocator: return BumpAllocator(self.rt_nbytes >> (6 if host else 0))

  @functools.cache
  def rt_buffer(self, uncached:bool=True, host:bool=False) -> Buffer:
    spec = BufferSpec(host=host, uncached=uncached, cpu_access=True)
    return Buffer(self.device, self.rt_allocator(uncached, host).size, dtypes.uint8, options=spec, preallocate=True)

  def rt_view(self, nbytes:int, dtype:DType=dtypes.uint8, uncached:bool=True, host:bool=False) -> Buffer: # a slot of the ring, wraps silently
    off = self.rt_allocator(uncached, host).alloc(max(nbytes, 1), alignment=128)
    return self.rt_buffer(uncached, host).view(nbytes // dtype.itemsize, dtype, off).ensure_allocated()

  def prog_buffer(self, b:UOp) -> Buffer: # program buffers are shared across linears, keyed on the placeholder
    if (buf:=self.prog_bufs.get(b)) is None:
      buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
    return buf

  def new_buffer(self, b:UOp) -> Buffer: # signals live in host memory: a peer polls them over pcie, not through its p2p window
    return self.rt_view(b.max_numel() * b.dtype.itemsize, b.dtype, host=is_signal(b.tag))

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
