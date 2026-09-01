from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, TYPE_CHECKING
import functools, time, collections, itertools, decimal
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, unwrap, PROFILE, all_same, VIZ
from tinygrad.helpers import to_tuple, ContextVar, Context, panic, partition
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer, DepsTracker
from tinygrad.device import ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, rewrite_group, GroupOp
from tinygrad.uop.symbolic import symbolic
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

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, element offset)
  return unwrap_view(v.src[0]) if v.op in (Ops.BITCAST, Ops.AFTER) else (v.src[0], v.src[1].val) if v.op is Ops.SHRINK else (v, 0)

def _lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp: # named submit_{base device}_{queue type}, the backend pm matches it
  fn = f"submit_{(devs:=to_tuple(devs))[0].split(':')[0]}_{queue.split(':')[0]}".lower()
  return UOp.custom_function(fn, UOp(Ops.LINEAR, src=tuple(cmds), arg=(devs, queue)))

def make_call(name:str, body:UOp, info:HCQInfo) -> UOp: return body.call(name=name, aux=info)

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

@dataclass(frozen=True)
class Dep: dev:str; queue:str; tag:int; lane:int # producer submit (dev, queue, tag) awaited by consumer lane # noqa: E702

signal_slots:dict[str, int] = collections.defaultdict(lambda: next(UOp.unique_num)) # slots share the uop counter, no collisions

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...]]]; batch_info:list[tuple[tuple[str, ...], str]]; profile:bool # noqa: E702
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker); signal_tags:set[int] = field(default_factory=set) # noqa: E702
  slots:dict[str, int] = field(default_factory=lambda: signal_slots)

  def signal(self, devs, slot:int=0, tag:str="signal") -> UOp:
    return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True, tag=tag)

def _get_call_bufs_by_lane(call:UOp, devices:tuple[str, ...]) -> list[list[Any]]:
  def dep_buf(b:UOp) -> Any: return base if (base:=(b.src[0] if b.op is Ops.MSELECT else b).base).op is Ops.PARAM else b.buffer
  return [[dep_buf(_lane(a, lane)) for a in get_call_arg_uops(call)] for lane in range(len(devices))]

def _wait_ins(ctx:BatchCtx, bufs_by_lane:list[list[Any]], write, devices:tuple[str, ...], queue:str, tag:int) -> list[UOp]:
  deps:list[Dep] = []
  for lane, bufs in enumerate(bufs_by_lane):
    written = write if write is not None else list(range(len(bufs)))
    deps += [Dep(d, q, t, lane) for d, q, t in ctx.tracker.access_resources(bufs, written, (devices[lane], queue, tag)) if t < tag]

  # same-queue submits are fifo-ordered, no wait needed
  if devices[0].split(":")[0] in {"AMD", "QCOM", "CPU"} or queue.startswith("COPY"):
    deps = [d for d in deps if (d.dev, d.queue) != (devices[d.lane], queue)]
  latest = {(d.dev, d.queue, d.lane): d for d in sorted(deps, key=lambda d: d.tag)}

  # keep only the latest signal
  rows:dict[tuple[str, int], dict[int, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
  for d in latest.values(): rows[(d.queue, d.tag)][d.lane].append(d.dev)
  waits = []
  for (dqueue, dtag), by_lane in rows.items():
    for ds in itertools.zip_longest(*(by_lane[lane] for lane in range(len(devices)))):
      sig = UOp.mstack(*[ctx.signal(d, tag="sentinel_signal") if dd is None else ctx.signal(dd, ctx.slots[dqueue]) for dd, d in zip(ds, devices)])
      waits.append(UOp(Ops.INS, arg=("wait", dtypes.void), src=(sig, UOp.const(dtag + 1, dtypes.uint64))))
  ctx.signal_tags |= {t for _, t in rows}
  return waits

def _merge_submits(subs:list[UOp]) -> UOp:
  if len(subs) == 1: return subs[0]
  devs, queue = subs[0].src[0].arg
  return make_submit(*[cmd for sub in subs for cmd in sub.src[0].src], devs=devs, queue=queue)

def _merge_queues(submits:list[UOp]) -> list[UOp]: # one submit per queue, the blocks only order through their gpu waits
  groups:dict[tuple, list[UOp]] = collections.defaultdict(list)
  for sub in submits: groups[sub.src[0].arg].append(sub)
  return [_merge_submits(g) for g in groups.values()]

def _emit_submits(ctx:BatchCtx, call_waits:list[list[UOp]]) -> tuple[list[UOp], list[tuple]]:
  # one submit per call: timeline sync on first queue use, timestamps, the call, and a signal if someone waits on it
  src, kerns, seen_queues = [], [], set()
  for tag, ((call, _), (devices, queue), q) in enumerate(zip(ctx.batch, ctx.batch_info, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if (devices, queue) not in seen_queues:
      seen_queues.add((devices, queue))
      epoch = ctx.signal(devices, tag="timeline_value").index(0) - 1
      q = [UOp(Ops.INS, arg=("barrier", dtypes.void), src=()),
           UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.signal(devices, tag="timeline_signal"), epoch))] + q

    # and make hcq call
    name, info = get_call_name(call, get_call_arg_uops(call)), HCQInfo(devices, estimate_uop(call))
    ts_ids = [next(UOp.unique_num) for _ in range(2)] if ctx.profile else []
    kerns.append((devices, name, info.estimates, tuple(ts_ids), make_call(name, call.src[0], info).key))

    ts_ins = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(ctx.signal(devices, s),)) for s in ts_ids]
    q += ts_ins[:1] + [call.replace(arg=replace(call.arg, aux=info))] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags:
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.signal(devices, ctx.slots[queue]), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_submit(*q, devs=devices, queue=queue))
  return src, kerns

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...]]], profile:bool) -> list[UOp]:
  if not batch: return []
  ctx = BatchCtx(batch, [(d, "COMPUTE:0" if c.src[0].op is Ops.PROGRAM else "COPY:0") for c, d in batch], profile)
  devices = tuple(dedup([d for devs, _ in ctx.batch_info for d in devs]))

  call_waits = [_wait_ins(ctx, _get_call_bufs_by_lane(c, d), get_call_outs_ins(c)[0], d, q, tag)
                for tag, ((c, _), (d, q)) in enumerate(zip(ctx.batch, ctx.batch_info))]
  submits, kerns = _emit_submits(ctx, call_waits)

  fences, resets, fins = [], [], []
  for _, group in itertools.groupby(sorted(devices), key=lambda d: d.split(":")[0]):
    devs = tuple(group)
    queues = dedup([q for bdevs, q in ctx.batch_info if set(bdevs) & set(devs)])

    fences.append(UOp.custom_function("hcq_fence", UOp(Ops.LINEAR, arg=(devs, tuple(queues)))))
    resets.append(UOp.custom_function("hcq_reset", UOp(Ops.LINEAR, arg=(devs, tuple(queues)))))
    submits += [make_submit(UOp(Ops.INS, arg=("store", dtypes.void),
                src=(ctx.signal(devs, signal_slots[f"{q}_done"]), UOp.const(1, dtypes.uint64))), devs=devs, queue=q) for q in queues]
    submits += [make_submit(*[UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.signal(devs, signal_slots[f"{q}_done"]),
                                                                           UOp.const(1, dtypes.uint64))) for q in queues],
                UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.signal(devs, tag="timeline_signal"),
                                                              ctx.signal(devs, tag="timeline_value").index(0).load())),
                devs=devs, queue="COMPUTE:0" if len(queues) > 1 else queues[0])]
    fins.append(UOp.custom_function("hcq_finalizer", UOp(Ops.LINEAR, arg=(devs, tuple(queues)))))

  estimates = sum((estimate_uop(call) for call, _ in batch), start=Estimates()).simplify()
  return [make_call(f"hcq batch ({len(batch)})", UOp.sink(*fences, *resets, *_merge_queues(submits), *fins),
                    HCQInfo(devices, estimates, kernels=tuple(kerns)))]

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...]]] = []
  for call in l.src:
    if (devs:=_get_enqueue_devs(call)) is not None: batch.append((call, to_tuple(devs)))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 3. encode

@dataclass
class EncodeCtx: # devs/queue plus everything on the device; the body's buffers register here as they are created
  dev:Any; devs:tuple[str, ...]; queue:str # noqa: E702
  args:list[UOp] = field(default_factory=list)
  blobs:dict[UOp, tuple[bytes, list[tuple[int, UOp]]]] = field(default_factory=dict) # cmdbuf placeholder -> (bytes, patch rows)
  def __getattr__(self, name): return getattr(self.dev, name)

  def new_arg(self, u:UOp) -> UOp: # a device buffer the exec binds per call
    if u not in self.args: self.args.append(u)
    return u

  def new_temp(self, shape:tuple[int, ...], dtype:DType, tag:str) -> UOp: # a scratch buffer owned by this linear
    return self.new_arg(UOp.placeholder(shape, dtype, next(UOp.unique_num), device=self.devs, tag=tag))

  def new_signal(self, slot:int=0, tag:str="signal") -> UOp:
    return self.new_arg(UOp.placeholder((1,), dtypes.uint64, slot, device=self.devs, volatile=True, tag=tag))

class HWQueue: # a renderer for queues: q_rewrite turns each submit op into words, the blob packs them as they come
  q_rewrite:PatternMatcher

  def __init__(self, ctx:EncodeCtx): self.ctx, self.blob, self.patches, self.deps = ctx, bytearray(), [], []
  def __getattr__(self, name): return getattr(self.ctx, name)

  def encode(self, submit:UOp) -> UOp: return self.submit(self.render(submit.src[0]))

  def q(self, *words): # a word is an int, a const uop, or a patch uop the blob keeps a slot for
    for w in words:
      c = (w.src[0] if w.op is Ops.CAST else w) if isinstance(w, UOp) else w
      if isinstance(c, UOp) and c.op is not Ops.CONST:
        self.patches.append((len(self.blob), w))
        self.blob += bytes(w.dtype.itemsize)
      else:
        v, n = (c.val, w.dtype.itemsize) if isinstance(w, UOp) else (c, 4)
        self.blob += (v & (1 << 8 * n) - 1).to_bytes(n, 'little')

  def render(self, lin:UOp) -> UOp: # the blob bakes on a fresh placeholder, shrunk to the stream the submit copies
    self.devs, self.queue = lin.arg
    for u in lin.src:
      if u.op in {Ops.INS, Ops.CALL}: self.q_rewrite.rewrite(u, ctx=self)
      else: self.deps.append(u) # lowered sync blocks this submit pushes after

    # nested word linears (kernargs) pack into the tail, their getaddrs re-target to views of the buffer
    stream, offs = len(self.blob), {}
    for l in dedup([g.src[0] for _, w in [*self.patches] for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR]):
      self.blob += bytes(-len(self.blob) % 128)
      start = len(self.blob)
      self.q(*l.src)
      offs[l] = (start, len(self.blob))

    # the blob is plain bytes on a fresh placeholder: the link patch words ride the ctx as (offset, word) rows,
    # the runtime ones the body stores every call, before the submit reads the stream
    buf = self.ctx.new_temp((len(self.blob),), dtypes.uint8, "cmdbuf")
    placed = UOp(Ops.GROUP, src=tuple(w for _, w in self.patches)).substitute({l: buf[o:e] for l, (o, e) in offs.items()}).src
    links, runtime = partition(list(zip([o for o, _ in self.patches], placed)), lambda p: is_link_value(p[1]))
    self.ctx.blobs[buf] = (bytes(self.blob), links)
    stores = [buf.shrink(((o, o + w.dtype.itemsize),)).bitcast(w.dtype).index(0).store(w) for o, w in runtime]
    return buf.after(*stores).shrink(((0, stream),))

  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")

def is_link_value(w:UOp) -> bool: # resolvable when the linear links: no variables or memory reads, no input-param addresses
  if w.op is Ops.GETADDR: return all(b.op is not Ops.PARAM or b.tag is not None for b in unwrap_mstack(w.buf_uop))
  if w.op in {Ops.LOAD, Ops.INDEX} or w.is_variable: return False
  return all(is_link_value(s) for s in w.src)

def hcq_fence(ctx:EncodeCtx, f:UOp) -> UOp: # spin until the group's timeline reaches the last recorded epoch
  done = ctx.new_signal(tag="timeline_signal").after(loop:=UOp.loop(0)).index(0).load()
  return done.end(loop, done < ctx.new_signal(tag="epoch").index(0).load())

def hcq_reset(ctx:EncodeCtx, r:UOp) -> UOp: # zero the batch signals, the fence made them safe to reuse
  rst:tuple[UOp, ...] = ()
  for slot in [signal_slots[k] for q in r.src[0].arg[1] for k in (q, f"{q}_done")]:
    rst = (ctx.new_signal(slot).after(*rst).index(0).store(0),)
  return rst[0]

def hcq_finalizer(ctx:EncodeCtx, fin:UOp) -> UOp: # the host bumps the timeline and records the epoch for the next fence
  epoch = (epoch_slot:=ctx.new_signal(tag="timeline_value").index(0)).load()
  return UOp.barrier(epoch_slot.store(epoch + UOp.const(1, dtypes.uint64)), ctx.new_signal(tag="epoch").index(0).store(epoch))

pm_hcq_encode = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence),
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_reset", name="r"), hcq_reset),
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_finalizer", name="fin"), hcq_finalizer),
])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo) or call.arg.aux.args: return None # not an hcq call, or lowered already

  # one ctx for the whole call: the body encodes with the hcq rules plus every device's own
  ctx = EncodeCtx(Device[(devs:=call.arg.aux.device)[0]], devs, "")
  pms = [Device[d].pm_encode for d in dedup([d.split(":")[0] for d in devs])]
  body = graph_rewrite(call.src[0], functools.reduce(lambda a, b: a + b, pms, pm_hcq_encode), ctx=ctx, walk=True, name="encode body")

  # the link patch words ride the call arg as stacked stores at byte offsets (one per word width), next to the blob bytes
  patched:dict[UOp, UOp] = {}
  for cmdbuf, (data, links) in ctx.blobs.items():
    binary = cmdbuf.store(UOp(Ops.BINARY, src=(), arg=data).bitcast(cmdbuf.dtype))
    stores = []
    for width in dedup([w.dtype.itemsize for _, w in links]):
      grp = [(o, w) for o, w in links if w.dtype.itemsize == width]
      stores.append(cmdbuf.index(UOp.stack(*[UOp.const(o, dtypes.int) for o, _ in grp])).store(UOp.stack(*[w for _, w in grp])))
    patched[cmdbuf] = cmdbuf.after(binary, *stores)

  # placeholders become canonical params of the body program, the call binds them in the same order
  tops = body.toposort(gate=lambda u: u.op is not Ops.PARAM)
  placeholders = dedup([s for u in tops for s in u.src if s.op is Ops.PARAM])
  args = {b: UOp.param(i, b.dtype, shape=b.shape, device=HCQ_RUNTIME_DEV.value, volatile=b.arg.volatile) for i, b in enumerate(placeholders)}
  rngs = {r: r.replace(arg=(i,)+r.arg[1:]) for i, r in enumerate(sorted([u for u in tops if u.op is Ops.RANGE], key=lambda r: r.arg))}
  sink = body.substitute(cast(dict[UOp, UOp], args) | rngs).replace(arg=KernelInfo("hcq_submit"), tag=1)
  return call.replace(src=(sink, *[patched.get(b, b) for b in placeholders]),
                      arg=replace(call.arg, aux=replace(call.arg.aux, args=tuple(args.items()))))

pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])



hcq_compile_cache:dict[tuple[bytes, bool], UOp] = {}

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear.key, profile)))) is None:
    # prep
    linear = linear.substitute({s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}, walk=True)
    linear = graph_rewrite(linear, pm_insert_copy_staging+pm_flatten_linear, name="insert copy staging")

    # schedule and encode on real buffers: the input addresses bake into the blobs at link
    linear = sched_batches(linear, profile)
    if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View Schedule")
    linear = graph_rewrite(linear, pm_encode, walk=True, name="encode")

    with Context(EMULATED_DTYPES=""): final_linear = hcq_compile_cache[cache_key] = lower_and_compile(linear)
  return final_linear


# *****************
# 9. bufferize placeholders: replace placeholders with real buffers

def bufferize_buf(ctx:tuple[bool, list[UOp]], buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.mstack(*(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=(dv, ctx[0])), HCQ_RUNTIME_DEV.value)
                      for dev in to_tuple(buf.device)))
pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="buf"), bufferize_buf)])

# *****************
# 10. link: bufferize the placeholders, then the patch stores fold into plain memory writes

def push_stack(op:UOp) -> UOp|None:
  if not (ns:=[s for s in op.src if s.op is Ops.STACK]) or not all_same([len(s.src) for s in ns]): return None
  return UOp(Ops.STACK, src=tuple(op.replace(src=tuple(s.src[i] if s.op is Ops.STACK else s for s in op.src)) for i in range(len(ns[0].src))))

def _bufs(buf:UOp) -> list[Buffer]: # one Buffer per lane
  if buf.op is Ops.MSTACK: return [cast(Buffer, x.buffer) for x in buf.src]
  return list(m.bufs) if isinstance(m:=buf.buffer, MultiBuffer) else [m]

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in _bufs(buf):
    if getattr(b, '_hcq_written', None) is not blob.arg: # programs are shared across linears, write them once
      cast(Any, b.ensure_allocated())._hcq_written = blob.arg
      b._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def resolve_getaddr(ctx:tuple[bool, list[UOp]], buf:UOp, g:UOp) -> UOp:
  ctx[1].append(buf) # the address bakes into the blob, the linked linear refholds the buffer (amd scratch outlives its realloc)
  devs, bufs = to_tuple(g.arg), _bufs(buf)
  if len(bufs) == 1: bufs = bufs * len(devs) # one buffer shared by every lane
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(x.get_buf(d).va_addr, dtypes.uint64) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, src=addrs)

def resolve_getaddr_view(bv:UOp, g:UOp) -> UOp:
  addr = UOp(Ops.GETADDR, src=(bv.src[0],), arg=g.arg)
  return addr if bv.op is Ops.BITCAST else addr + UOp.const(bv.src[1].val * bv.dtype.itemsize, dtypes.uint64)

def fold_stack_store(view:UOp, idx:UOp, val:UOp) -> UOp|None: # once every patch word resolved, they write in one go
  uncast = lambda w: w.src[0] if w.op is Ops.CAST else w # symbolic keeps casted consts, the row width masks anyway # noqa: E731
  vals, idxs = [uncast(w) for w in val.src], [uncast(i) for i in idx.src]
  if any(w.op is not Ops.CONST for w in vals + idxs): return None
  buf, start = unwrap_view(view)
  for b in _bufs(buf):
    mv = b.ensure_allocated()._buf.cpu_view().view(fmt='B')
    for i, w0, w in zip(idxs, val.src, vals): # the width comes from each word, the index is in view elements
      width = w0.dtype.itemsize
      mv[(bo:=(start + i.arg) * view.dtype.itemsize):bo + width] = (w.arg & (1 << 8 * width) - 1).to_bytes(width, 'little')
  return UOp(Ops.NOOP)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU | {Ops.CAST}, name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(Ops.AFTER, name="a"),), name="g"), lambda a, g: g.replace(src=(a.src[0],))),
  (UPat(Ops.GETADDR, src=(UPat((Ops.SHRINK, Ops.BITCAST), name="bv"),), name="g"), resolve_getaddr_view),
  (UPat(Ops.GETADDR, src=(UPat((Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), name="buf"),), name="g"), resolve_getaddr),

  # folders
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat((Ops.BITCAST, Ops.SHRINK, Ops.BUFFER, Ops.MSTACK), name="view")
    .index(UPat(Ops.STACK, name="idx")).store(UPat(Ops.STACK, name="val")), fold_stack_store),
])

pm_assert_no_afters = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: panic(RuntimeError, f"AFTER left at hcq_link: {a.src[0].op}"))])

link_linear_cache:dict[bytes, UOp] = {}

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear_key:=linear.key)) is not None: return linked
  refs:list[UOp] = []
  linear = graph_rewrite(linear, pm_resolve_patches+symbolic+pm_assert_no_afters, bpm=pm_bufferize, ctx=(cache, refs), bottom_up=False,
                         name="resolve patches")
  if refs: linear = linear.replace(src=(linear.src[0].replace(src=linear.src[0].src + tuple(dedup(refs))), *linear.src[1:]))
  if cache: link_linear_cache[linear_key] = linear
  return linear

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
      (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].signal("timeline")),
      (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].signal("value", 1, device="CPU")),
      (UPat(Ops.PARAM, tag="epoch", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot, device="CPU")),
      (UPat(Ops.PARAM, tag="signal", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if b.tag is None else ctx[0].new_buffer(b, cache=ctx[1]))
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

  def new_buffer(self, b:UOp, cache:bool) -> Buffer:
    if b.tag == "program": # program buffers are shared across linears, keyed on the placeholder
      if (buf:=self.prog_bufs.get(b)) is None:
        buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
      return buf
    return self.rt_view(b.max_numel() * b.dtype.itemsize, b.dtype)

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
    sig = self.signal("timeline")._buf.cpu_view().view(fmt='Q')
    tl = self.signal("value", 1, device="CPU")._buf.cpu_view().view(fmt='Q')
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
