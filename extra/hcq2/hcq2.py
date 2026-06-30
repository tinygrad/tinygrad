from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any
import struct, functools, time, collections, importlib, itertools, weakref
from dataclasses import replace, dataclass, field
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, DEBUG, dedup, flatten, pluralize
from tinygrad.helpers import to_tuple, round_up
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites, GroupOp
from tinygrad.uop.symbolic import symbolic_simple, symbolic
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop, pm_flatten_linear
from tinygrad.engine.jit import DepsTracker

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0 # GPU timestamp counter ticks per microsecond; override per device

  def __init__(self, device:str, allocator:'HCQAllocator', compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    # default pm bufferize
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.BUFFER, tag="timeline_signal"), lambda ctx: ctx.timeline_signal()),
      (UPat(Ops.BUFFER, tag="timeline_value"), lambda ctx: ctx.timeline_value()),
      (UPat(Ops.BUFFER, tag="sentinel_signal"), lambda ctx: ctx.timeline_signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.BUFFER, name="b"), lambda ctx, b:
        Buffer(ctx.device, b.max_numel(), b.dtype, options=BufferSpec(host=False, uncached=True, cpu_access=True, nolru=True))), # TODO: remove nolru
    ])

    super().__init__(device, allocator, compilers, lambda *a, **kw: None, None, arch=arch)

  @functools.cache
  def timeline_signal(self, queue:str|None=None, init_value:int=0) -> Buffer:
    buf = Buffer(self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf._buf.cpu_view().mv.cast('Q')[0] = init_value
    return buf

  @functools.cache
  def timeline_value(self, queue:str|None=None, init_value:int=1) -> Buffer:
    buf = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    buf.as_memoryview(force_zero_copy=True).cast('Q')[0] = init_value
    return buf

  @functools.cached_property
  def timestamps_buf(self) -> Buffer:
    return Buffer(self.device, 0x1000, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal()._buf.cpu_view().mv.cast('Q')
    tl = self.timeline_value().as_memoryview(force_zero_copy=True).cast('Q')
    st = time.perf_counter()
    while sig[0] < tl[0] - 1:
      if time.perf_counter() - st > (timeout or 3000) / 1000: self.on_device_hang()

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def _select_iface(self):
    assert (v:=getenv(k:=f'{type(self).__name__[:-6].upper()}_IFACE', "")) == "",  \
      f"{k}={v} is deprecated, use DEV={replace(DEV.target(type(self).__name__[:-6]), interface=v)} instead"
    assert hasattr(self, "ifaces"), "must have ifaces to select an iface"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fall back to mock ifaces
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # if the device has an interface, call device_fini to clean up resources
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

class HCQ2Buffer:
  def __init__(self, va_addr:sint, size:int, meta:Any=None, _base:HCQ2Buffer|None=None, view:MMIOInterface|None=None, owner:HCQ2Compiled|None=None):
    self.va_addr, self.size, self.meta, self._base, self.view, self.owner = va_addr, size, meta, _base, view, owner

  def offset(self, offset:int=0, size:int|None=None) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def base(self) -> HCQ2Buffer: return self._base or self

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _map(self, buf:HCQ2Buffer) -> HCQ2Buffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    self.dev.synchronize()
    if options is not None and options.external_ptr is not None: return
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self.dev.iface.free(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)

  def _wrap(self, dev:str, sz:int, opaque:HCQ2Buffer) -> Buffer:
    return Buffer(dev, sz, dtypes.uint8, opaque=opaque, options=BufferSpec(external_ptr=1))

  def _copy(self, dst:Buffer, src:Buffer):
    from tinygrad.engine.realize import run_linear
    su = UOp.from_buffer(src)
    run_linear(UOp(Ops.LINEAR, dtypes.void, (su.copy_to_device(dst.device).call(UOp.from_buffer(dst), su),)), update_stats=False)

  def _copyin(self, dest:HCQ2Buffer, src:memoryview):
    s = Buffer(self.dev.device, len(src), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    s._buf.cpu_view()[:len(src)] = src
    self._copy(self._wrap(self.dev.device, len(src), dest), s)

  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    d = Buffer(self.dev.device, len(dest), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    self._copy(d, self._wrap(self.dev.device, len(dest), src))
    self.dev.synchronize()
    dest[:] = d._buf.cpu_view()[:len(dest)]

  # def _as_buffer(self, buf): return buf.cpu_view().mv

# *****************
# 0. helpers

HCQ_DEVS = frozenset(("AMD",))
HCQ_P2P_DEVS = HCQ_DEVS | frozenset(("CPU",))

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_after(uop):
  while uop.op is Ops.AFTER: uop = uop.src[0]
  return uop

def make_getaddr(u, device=None):
  if unwrap_after(u).op not in (Ops.BUFFER, Ops.SLICE, Ops.BINARY, Ops.MSTACK, Ops.MSELECT, Ops.PARAM): return u
  return UOp(Ops.GETADDR, dtypes.uint64, src=(u, UOp(Ops.DEVICE, arg=device or to_tuple(u.device)[0])))

def make_ins(op, *srcs):
  return UOp(Ops.INS, dtypes.void, tuple(UOp.const(dtypes.uint32, s) if isinstance(s, int) else s.cast(dtypes.uint32) for s in srcs), op)

def make_patch(buf:UOp, off:sint, val:UOp, dtype=None) -> UOp:
  dt = dtype or val.dtype
  return UOp(Ops.SHRINK, buf.dtype.base, (buf, UOp.const(dtypes.int, off), UOp.const(dtypes.int, dt.itemsize))).bitcast(dt).store(val.cast(dt))

def make_cmdbuf(lin, devs, tag):
  blob, patches = b'', []
  for s in (s for ins in lin.src for s in ins.src):
    if s.op is not Ops.CONST: patches.append((len(blob), s))
    blob += struct.pack(f'<{s.dtype.fmt}', s.arg if s.op is Ops.CONST else 0x0)
  buf = UOp.new_buffer(devs, len(blob), dtypes.uint8).rtag(tag)
  afters = dedup(u for _, s in patches for u in s.toposort() if u.op is Ops.AFTER)
  deps = tuple(d for p in afters for d in p.src[1:])
  cmdbuf = buf.after(buf.store(UOp(Ops.BINARY, dtypes.void, src=(), arg=blob)),
                     *[make_patch(buf, off, s) for off, s in patches], *deps)
  return cmdbuf.substitute({p: p.src[0] for p in afters}) if afters else cmdbuf

def make_mstack(uops): return uops[0] if len(uops) == 1 else UOp(Ops.MSTACK, uops[0].dtype, tuple(uops))

def make_signal(devs, queue=None, sentinel=False):
  return UOp.new_buffer(devs, 1, dtypes.uint64).rtag("sentinel_signal" if sentinel else (queue, "timeline_signal") if queue else "timeline_signal")
def make_signal_value(devs, queue=None): return UOp.new_buffer(devs, 1, dtypes.uint64).rtag((queue, "timeline_value") if queue else "timeline_value")

def get_submit(ast:UOp) -> UOp: return next(u for u in ast.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_cmdbuf")

@functools.cache
def get_backend_pm(device:str|tuple[str,...], pm:str) -> PatternMatcher|None:
  device = to_tuple(device)[0].split(":")[0].lower()
  try:
    importlib.import_module(f'tinygrad.runtime.ops_{device}') # TODO: remove that
    return getattr(importlib.import_module(f'extra.hcq2.ops_{device}2'), pm)
  except ImportError: return None

@dataclass(frozen=True)
class HCQInfo:
  name:str
  estimates:Estimates
  device:tuple[str, ...]
  queue:str

  params:tuple[int, ...] = ()
  inputs:int|None = None

# *****************
# 0.1. prep: replace buffers with params

def replace_buffer(ctx:dict[UOp, UOp], b:UOp) -> UOp:
  if (p:=ctx.get(b)) is None: ctx[b] = p = b.param_like(len(ctx))
  return p
pm_replace_buffers = PatternMatcher([(UPat(Ops.BUFFER, name="b"), replace_buffer)])

# *****************
# 1.1. prep: staging copies

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS) and not all_devices_in(b.device, HCQ_P2P_DEVS)

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  stage = UOp.new_buffer("CPU", src.max_numel() * src.dtype.base.itemsize, dtypes.uint8)
  return UOp(Ops.LINEAR, dtypes.void, (src.copy_to_device("CPU").call(stage, src), stage.copy_to_device(dst.device).call(dst, stage)))
pm_insert_copy_staging = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy)])

# *****************
# 2.1. early hcq calls tracking

def tag_hcq_calls(ctx:itertools.count, call:UOp):
  if (hcq_devs:=next((b.device for b in call.src[1:] if all_devices_in(b.device, HCQ_DEVS)), None)) is None: return None

  queue = "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0"
  info = HCQInfo(get_call_name(call, get_call_arg_uops(call)), estimate_uop(call), to_tuple(hcq_devs), queue)
  return call.replace(arg=replace(call.arg, aux=info)).rtag(next(ctx))
pm_tag_hcq_calls = PatternMatcher([(UPat(Ops.CALL, name="call"), tag_hcq_calls)])

# *****************
# 2.2. early deps tracking
# device.timeline_signal/value are the per-device schedule epoch. Before a schedule queue accesses memory owned by device N for the first time,
# it waits for device[N].timeline_signal >= device[N].timeline_value - 1. This orders the schedule after all prior schedules that touched device N.
#
# queue.timeline_signal/value are per-queue progress counters used only inside a schedule.
# Only the owner queue signals its queue.timeline_signal. Values are monotonic.
#
# At schedule end, one finalizer queue per touched device[N] waits for every active queue on device[N] to reach its schedule-local
# final queue.timeline value, then signals device[N].timeline_signal with the schedule's reserved device epoch. After that, buffers/transients
# for device N from this schedule are safe for the next schedule
#
# C programs reserve and bump timeline values, then patch command buffers with the concrete wait/signal values.

class HCQDepsTracker(DepsTracker):
  def _key(self, u:UOp) -> tuple[Any, int, int]:
    if u.op is Ops.SLICE: return (r:=self._key(u.src[0]))[0], r[1]+(o:=u.src[1].arg*u.dtype.itemsize), r[1]+o+u.arg*u.dtype.itemsize
    if u.op is Ops.MSELECT:
      if u.src[0].op is Ops.MSTACK: return self._key(u.src[0].src[u.arg])
      if u.src[0].op is Ops.PARAM: return ((r:=self._key(u.src[0]))[0], u.arg), r[1], r[2]
      assert False
      # b = u.src[0].buffer.bufs[u.arg]
      # return id(b.base), b.offset, b.offset+b.nbytes
    return (u.arg.slot, u.arg.name), 0, u.max_numel() * u.dtype.base.itemsize

# *****************
# 2.1. deps tracking

def sched_sync(ctx:HCQDepsTracker, call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo): return None

  refs = get_call_arg_uops(call)
  outs, _ = get_call_outs_ins(call)
  devices, queue = call.arg.aux.device, call.arg.aux.queue

  dep_lanes:list[tuple[UOp, int]] = []
  for lane, d in enumerate(devices):
    lane_refs = [b.mselect(lane) for b in refs] if len(devices) > 1 else refs
    for dep, dlane in ctx.access_resources(lane_refs, outs, (call, lane)): dep_lanes.append((dep, dlane, lane))

  if devices[0].split(":")[0] in {"AMD", "QCOM"} or queue.startswith("COPY"):
    dep_lanes = [(dep, lane) for dep, dlane, lane in dep_lanes if (dep.arg.aux.device[dlane], dep.arg.aux.queue) != (devices[lane], queue)]

  latest = {(dep.arg, lane): dep for dep, lane in sorted(dep_lanes, key=lambda x: x[0].tag)}
  deps:dict[UOp, tuple[int, ...]] = collections.defaultdict(tuple)
  for (_, lane), dep in latest.items(): deps[dep] += (lane,)

  return call.after(*deps, arg=tuple(deps.values()))
pm_sched_sync = PatternMatcher([(UPat(Ops.CALL, name="call"), sched_sync)])

# *****************
# 2.2. merge queues

def _merged_hcq_call(calls:list[UOp]):
  info = replace(unwrap_after(calls[0]).arg.aux, estimates=sum((unwrap_after(c).arg.aux.estimates for c in calls), start=Estimates()))
  cmdbuf = make_submit(*calls, devs=info.device, queue=info.queue)
  return UOp.custom_function("hcq", cmdbuf.sink()).call(name="hcq", aux=info)

def merge_queues(linear:UOp) -> UOp:
  new_src:list[UOp] = []
  opened_qs:dict[tuple[tuple[str, ...], str], list[UOp]] = {} # (devs, queue) -> list of calls, kept in submit order

  for call in linear.src:
    if not isinstance(unwrap_after(call).arg.aux, HCQInfo):
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in list(opened_qs)] + [call]
      continue

    devices, queue = unwrap_after(call).arg.aux.device, unwrap_after(call).arg.aux.queue

    if (old:=opened_qs.pop((devices, queue), None)) is not None: new_rec = old + [call]
    else:
      # no such queue opened: close every open submit on this queue that shares a device, so submit order is kept
      closing = [k for k in opened_qs if k[1] == queue and set(k[0]) & set(devs)]
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in closing]
      new_rec = [call]
    opened_qs[(devices, queue)] = new_rec
  return linear.replace(src=tuple(new_src + [_merged_hcq_call(c) for c in opened_qs.values()]))
pm_merge_queues = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), merge_queues)])

# *****************
# 2.2. finalizer

def add_finalizer(ctx:itertools.count, linear:UOp) -> UOp:
  # collect by device type
  parts:dict[str, list[UOp]] = collections.defaultdict(list)
  for call in linear.src:
    if (c:=unwrap_after(call)).src[0].op is not Ops.CUSTOM_FUNCTION or c.src[0].arg != "hcq": continue
    parts[c.arg.aux.device[0].split(':')[0]].append(unwrap_after(get_submit(call).src[0].src[0]))

  nbump = next(ctx)
  finalizers = []
  for calls in parts.values():
    devs = tuple(dedup(d for call in calls for d in unwrap_after(call).arg.aux.device))
    zero = UOp.const(dtypes.int, 0)
    tl = make_signal_value(devs)

    # split each (multi-device) call into per-device deps, then store the device timeline value into the device signal after them
    deps:dict[UOp, tuple[int, ...]] = collections.defaultdict(tuple)
    for call in calls:
      for d in unwrap_after(call).arg.aux.device: deps[call] += (devs.index(d),)
    store = make_signal(devs).store(tl.index(zero)).after(*deps, arg=tuple(deps.values())).rtag("deps")
    submit = make_submit(store, devs=devs, queue="COMPUTE:0")

    upd = [(tl, 1)] + [(make_signal_value(devs, queue=qn), nbump) for qn in dedup([unwrap_after(call).arg.aux.queue for call in calls])]
    patches = [s.after(submit).index(zero, dtype=s.dtype.ptr()).store(s.index(zero) + inc) for s, inc in upd]
    finalizers.append(UOp.custom_function("hcq", UOp.barrier(*patches).sink()).call(aux=HCQInfo("hcq finalizer", Estimates(), devs, "COMPUTE:0")))
  return linear.replace(src=linear.src + tuple(finalizers))
pm_add_finalizer = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), add_finalizer)])

# *****************
# 2.4. global sync

def add_global_sync(ctx:set[tuple[str, ...]], submit:UOp, q:UOp) -> UOp|None:
  if (devs:=q.arg[0]) in ctx: return None
  ctx.add(devs)

  # some devices from a command buffer might be used for the first time this schedule, so we wait for their global timeline epoch.
  wait = make_signal(devs).wait(make_signal_value(devs).index(UOp.const(dtypes.int, 0)) - 1)
  return submit.replace(src=(q.replace(src=(UOp(Ops.BARRIER, dtypes.void), wait, *q.src)),))
pm_add_global_sync = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_global_sync)])

# *****************
# 2.4. replace params with per-submit input address loads

def replace_params(call:UOp) -> UOp|None:
  if not (params:={u:u.arg.slot for u in call.src[0].toposort(enter_calls=False) if u.op is Ops.PARAM and u.addrspace is AddrSpace.GLOBAL}): return None

  # fill new info
  hcqinfo = replace(call.arg.aux, params=tuple(sorted(set(params.values()))), inputs=len(get_call_arg_uops(call)))

  inputs = UOp.new_buffer(get_submit(call.src[0]).src[0].arg[0], len(hcqinfo.params), dtypes.uint64).rtag("inputs")

  slot2idx = {s:i for i,s in enumerate(hcqinfo.params)}
  body = call.src[0].substitute({u:inputs.index(UOp.const(dtypes.int, slot2idx[s])).load() for u,s in params.items()})

  return call.replace(src=(body, *call.src[1:], inputs), arg=replace(call.arg, aux=hcqinfo))
pm_replace_params = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), replace_params)])

# *****************
# 3.1. lower loads/stores

def add_loads(ctx:set[int], submit:UOp, q:UOp) -> UOp|None:
  if (deps:=next((s for s in q.src if s.op is Ops.AFTER), None)) is None: return None
  cur_devs = q.arg[0]

  waits = []
  for lanes, dep in zip(deps.arg, deps.src[1:]):
    devs, queue = dep.arg.aux.device, dep.arg.aux.queue
    ctx.add(dep.tag) # mark op to update signal.

    # for lanes that need this dep, wait on the dep device's signal/value; other lanes get a passing sentinel
    lanes = set(lanes)
    sig = make_mstack([make_signal(devs[j] if j in lanes else d, queue=queue, sentinel=j not in lanes) for j, d in enumerate(cur_devs)])
    val = make_mstack([make_signal_value(devs[j] if j in lanes else d, queue=queue) for j, d in enumerate(cur_devs)]).index(UOp.const(dtypes.int, 0))
    waits.append(sig.wait(val + dep.tag))
  new_src = flatten(waits+[deps.src[0]] if s is deps else [s] for s in q.src)
  return submit.replace(src=(q.replace(src=tuple(new_src)),))
pm_add_inner_loads = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_loads)])

def add_stores(ctx:set[int], submit:UOp, q:UOp) -> UOp|None:
  devs, queue = q.arg
  new_src = []
  for op in q.src:
    new_src.append(op)
    if (sigval:=unwrap_after(op).tag) in ctx:
      new_src.append(make_signal(devs, queue=queue).store(make_signal_value(devs, queue=queue).index(UOp.const(dtypes.int, 0)) + sigval))
  return submit.replace(src=(q.replace(src=tuple(new_src)),))
pm_add_inner_stores = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_stores)])

# *****************
# 2.1. hcq lowering: programs

def encode_kernargs_clike(call:UOp, prg:UOp, devs:str|tuple[str, ...]) -> UOp:
  data, info = prg.arg
  call_args = get_call_arg_uops(call)
  # buf = UOp.placeholder((data.kernargs_alloc_size,), dtypes.uint8, 0).rtag("kernargs")
  buf = UOp.new_buffer(devs, data.kernargs_alloc_size, dtypes.uint8).rtag("kernargs")
  patches = [make_patch(buf, i*8, make_getaddr(call_args[gi], devs)) for i,gi in enumerate(info.globals)] \
          + [make_patch(buf, len(info.globals)*8 + i*4, v, dtypes.uint32) for i,v in enumerate(info.vars)]
  return buf.after(*patches)

# def encode_program_bytes(data, image_bytes) -> UOp:
#   buf = UOp.placeholder((len(image_bytes),), dtypes.uint8, 0).rtag("program")
#   blob = UOp(Ops.BINARY, dtypes.void, src=(), arg=image_bytes)
#   return prg.replace(src=(buf.after(buf.store(blob)),), arg=(data, prg.arg))

# def prep_program(call:UOp, prg:UOp) -> UOp|None:
#   if (pm:=get_backend_pm(call.src[1].device, "pm_prep_program")) is None or (lowered:=pm.rewrite(prg)) is None: return None

#   data, image_bytes = lowered
#   buf = UOp.placeholder((len(image_bytes),), dtypes.uint8, 0).rtag("program")
#   blob = UOp(Ops.BINARY, dtypes.void, src=(), arg=image_bytes)
#   return prg.replace(src=(buf.after(buf.store(blob)),), arg=(data, prg.arg)).call(*call.src[1:], aux=HCQInfo.from_call(call))

# pm_prep_runtime = PatternMatcher([
#   # bind generic PROGRAM device to the call's actual dev(s), then run device-specific lowering
#   (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"),),
#     name="call", allow_any_len=True), prep_program),
# ])

# *****************
# 2.2. hcq lowering: ops to ir

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  return UOp.custom_function("submit_cmdbuf", UOp(Ops.LINEAR, dtypes.void, src=tuple(cmds), arg=(to_tuple(devs), queue)))

def make_hcq_call(call:UOp, root:UOp) -> UOp:
  return UOp.custom_function("hcq", root.sink()).call(aux=HCQInfo.from_call(call) if (aux:=call.arg.aux) is None else aux)

# def lower_program(call:UOp, prg:UOp) -> UOp:
#   if (hcq_dev:=next((b.device for b in call.src[1:] if b.device.split(":")[0] in HCQ_DEVS), None)) is None: return None
#   return make_hcq_call(call, make_submit(call, devs=hcq_dev, queue="COMPUTE:0"))

# def lower_copy(call:UOp, copy:UOp) -> UOp|None:
#   dst, src = call.src[1], call.src[2]
#   if (hcq_dev:=next((b.device for b in (dst, src) if b.device.split(":")[0] in HCQ_DEVS), None)) is None: return None

#   cp_op = UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.max_numel() * src.dtype.base.itemsize).call(*call.src[1:]) # TODO: spec
#   return make_hcq_call(call, make_submit(cp_op, devs=hcq_dev, queue="COPY:0"))

# pm_lower_ops = PatternMatcher([
#   (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), lower_program),
#   (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
# ])

def encode_cmdbuf(submit:UOp, lin:UOp) -> UOp|None:
  if (pm:=Device.get_class(lin.arg[0][0]).pm_lower) is None: return None
  return graph_rewrite(submit, pm, name=f"encode {lin.arg[0]}", enter_calls=True)
pm_encode_cmdbufs = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), encode_cmdbuf)])

# *****************
# 5.2. lift patches to the command buffer (root)

# def lift_patches_to_cmdbuf(cmdbuf:UOp) -> UOp|None:
#   if not (patches:=dedup(u for store in cmdbuf.src[1:] for u in store.toposort() if u.op is Ops.AFTER)): return None
#   deps = tuple(d for p in patches for d in p.src[1:])
#   return cmdbuf.replace(src=cmdbuf.src + deps).substitute({p: p.src[0] for p in patches})
# pm_lift_patches_to_cmdbuf = PatternMatcher([
#   (UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, tag={"compute", "copy"}),), allow_any_len=True, name="cmdbuf"), lift_patches_to_cmdbuf),
# ])

# *****************
# 5.3. pack placeholders buffers

def pack_hcq_placeholders(call:UOp) -> UOp|None:
  bufs = [b for b in call.src[0].toposort() if b.op is Ops.BUFFER and b.tag in (maxtags:={"scratch"}) | (sumtags:={"program", "kernargs"})]

  off_per_buf:dict[UOp, int] = {}
  size_per_tag:dict[str, int] = {}
  for b in bufs:
    bsz = b.max_numel()
    if b.tag in maxtags: size_per_tag[b.tag] = max(size_per_tag.get(b.tag, 0), bsz)
    elif b.tag in sumtags:
      off_per_buf[b] = round_up(size_per_tag.get(b.tag, 0), {"program": 0x1000}.get(b.tag, 128))
      size_per_tag[b.tag] = off_per_buf[b] + bsz

  count_per_tag = collections.Counter(b.tag for b in bufs)
  ref_bufs = {b.tag:b for b in bufs if count_per_tag[b.tag] > 1}
  bases = {tag:UOp.new_buffer(b.device, size_per_tag[tag], b.dtype).rtag(tag) for tag,b in ref_bufs.items()}
  subs = {b:UOp(Ops.SLICE, b.dtype, (bases[b.tag], UOp.const(dtypes.weakint, off_per_buf.get(b, 0))), b.max_numel()) for b in bufs if b.tag in bases}
  return call.replace(src=(call.src[0].substitute(subs, walk=True), *call.src[1:])) if subs else None
pm_pack_placeholders = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), pack_hcq_placeholders)])

# *****************
# 5.4. capture buffers reachable from each hcq call as BIND, so we don't drop their refs

def hold_call_buffers(call:UOp) -> UOp|None:
  if not (bufs:=tuple(dedup(u for u in call.src[0].toposort() if u.op is Ops.BUFFER and u not in call.src))): return None
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=bufs),))
pm_hold_call_buffers = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), hold_call_buffers)])

# *****************
# 6. bufferize placeholders: replace placeholders with real buffers.

def bufferize_buf(buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  uops = tuple(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=dv), "CPU") for dev in to_tuple(buf.device))
  return make_mstack(uops)
pm_bufferize = PatternMatcher([(UPat(Ops.BUFFER, name="buf"), bufferize_buf)])

# *****************
# 7. resolve patches

def push_stack(op, s): return UOp(Ops.STACK, op.dtype.scalar().vec(len(s.src)),
  tuple(op.replace(dtype=op.dtype.scalar(), src=tuple(x if y is s else y for y in op.src)) for x in s.src))

def fold_blob_store(buf:UOp, blob:UOp) -> UOp:
  for b in (mb.bufs if isinstance((mb:=buf.buffer), MultiBuffer) else (mb,)): b.ensure_allocated()._buf.cpu_view().mv.cast('B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(buf:UOp, off:UOp, val:UOp) -> UOp:
  for b, v in zip((bs:=mb.bufs if isinstance((mb:=buf.buffer), MultiBuffer) else (mb,)), val.src if val.op is Ops.STACK else (val,)*len(bs)):
    struct.pack_into(f'<{v.dtype.fmt}', b.ensure_allocated()._buf.cpu_view().mv.cast('B'), off.arg * buf.dtype.base.itemsize, v.arg)
  return UOp(Ops.NOOP)

def resolve_getaddr(buf:UOp, g:UOp) -> UOp|None:
  if buf.op not in (Ops.BUFFER, Ops.MSTACK, Ops.MSELECT): return None
  devs, b = to_tuple(g.src[1].arg), buf.buffer
  bufs = tuple(cast(Buffer, x.buffer) for x in buf.src) if buf.op is Ops.MSTACK else tuple(b.bufs if isinstance(b, MultiBuffer) else (b,)*len(devs))
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(dtypes.uint64, x.get_buf(d).va_addr) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, dtypes.uint64.vec(len(addrs)), addrs)

def resolve_getaddr_slice(bv:UOp, dev:UOp) -> UOp:
  itemsize = bv.src[0].dtype.itemsize if unwrap_after(bv.src[0]).op in (Ops.BUFFER, Ops.SLICE, Ops.MSTACK, Ops.MSELECT) else bv.dtype.itemsize
  return UOp(Ops.GETADDR, dtypes.uint64, src=(bv.src[0], dev)) + UOp.const(dtypes.uint64, bv.src[1].arg * itemsize)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU, src=[UPat(Ops.STACK, name="s"), UPat(Ops.CONST)], name="op"), push_stack),
  (UPat(Ops.CAST, src=(UPat(Ops.STACK, name="s"),), name="op"), push_stack),

  # shrink on slice is shrink on base at offset
  (UPat(Ops.SHRINK, src=(UPat(Ops.SLICE, name="bv"), UPat(), UPat()), name="shr"),
    lambda shr, bv: shr.replace(src=(bv.src[0], shr.src[1] + bv.src[1].cast(shr.src[1].dtype), shr.src[2]))),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(Ops.SLICE, name="bv"), UPat(Ops.DEVICE, name="dev"))), resolve_getaddr_slice), # getaddr(slice(x)) -> offset+getaddr(x)
  (UPat(Ops.GETADDR, src=(UPat(name="buf"), UPat(Ops.DEVICE)), name="g"), resolve_getaddr),

  # folders
  (UPat({Ops.BUFFER, Ops.SLICE, Ops.MSTACK}, name="buf").store(UPat(Ops.BINARY, name="blob")), fold_blob_store),
  (UPat(Ops.SHRINK, src=(UPat({Ops.BUFFER, Ops.SLICE, Ops.MSTACK}, name="buf"), UPat.cvar("off"), UPat(Ops.CONST))).bitcast()
    .store(UPat.any(UPat.cvar("val"), UPat(Ops.STACK, name="val"))), fold_const_store),
]) + symbolic_simple

# *****************
# 8. callify hcq programs

def to_param(bufs:list[UOp], ref:UOp) -> UOp:
  if ref not in bufs: bufs.append(ref)
  return UOp.placeholder((ref.buffer.size,), ref.dtype, bufs.index(ref))
pm_to_param = PatternMatcher([(UPat({Ops.MSELECT, Ops.MSTACK, Ops.BUFFER}, name="r"), lambda ctx, r: to_param(ctx, r))])

def parametrize_host_buffers(call:UOp) -> UOp:
  # preserve original order of args
  body = graph_rewrite(call.src[0], pm_to_param, ctx=(bufs:=list(get_call_arg_uops(call))), bottom_up=True, name="parametrize host buffers")

  # move vars to new slots
  var_slots = {nm:len(bufs)+i for i,nm in enumerate(sorted({v.expr for v in body.variables() if v.op is Ops.PARAM}))}
  body = body.substitute({v:v.replace(arg=replace(v.arg, slot=var_slots[v.expr])) for v in body.variables() if v.op is Ops.PARAM})

  return call.replace(src=(body, *bufs) + tuple(x for x in call.src[1:] if x.op is Ops.BIND))
pm_parametrize_host_buffers = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), parametrize_host_buffers)])

def callify_hcq(call:UOp) -> UOp:
  prg = to_program(call.src[0].src[0].replace(arg=KernelInfo("hcq_submit"), tag=1), Device["CPU"].renderer)
  return UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(prg,), arg="hcq").call(*call.src[1:], aux=call.arg.aux)
pm_callify_hcq = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), callify_hcq)])

@track_rewrites(lambda _,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp) -> UOp:
  # linear = graph_rewrite(linear, pm_replace_buffers, ctx=(buffers_map:={}), walk=True, enter_calls=True, name="replace buffer")

  # schedule
  linear = graph_rewrite(linear, pm_insert_copy_staging + pm_flatten_linear, name="insert copy staging")
  linear = graph_rewrite(linear, pm_tag_hcq_calls, ctx=(enumerator:=itertools.count(0)), walk=True, name="tag hcq calls")
  linear = graph_rewrite(linear, pm_sched_sync, ctx=(deps_ctx:=HCQDepsTracker()), walk=True, name="schedule sync")
  linear = graph_rewrite(linear, pm_merge_queues, walk=True, name="merge queues")
  linear = graph_rewrite(linear, pm_add_finalizer, ctx=enumerator, walk=True, name="add finalizer")
  linear = graph_rewrite(linear, pm_add_global_sync, ctx=set(), walk=True, name="add global sync", enter_calls=True)
  linear = graph_rewrite(linear, pm_replace_params, walk=True, name="replace params")

  # lowering to hcq ir
  linear = graph_rewrite(linear, pm_add_inner_loads, ctx=(waited:=set()), walk=True, name="add loads", enter_calls=True)
  linear = graph_rewrite(linear, pm_add_inner_stores, ctx=waited, walk=True, name="add stores", enter_calls=True)
  linear = graph_rewrite(linear, pm_encode_cmdbufs, walk=True, name="encode cmdbufs", enter_calls=True)
  linear = graph_rewrite(linear, pm_hold_call_buffers, walk=True, name="hold call buffers")
  # linear = graph_rewrite(linear, pm_resolve_patches, bottom_up=False, name="simplify patches", enter_calls=True)
  return linear
  # linear = graph_rewrite(linear, pm_lift_patches_to_cmdbuf, name="lift patches to cmdbuf", enter_calls=True)

  # linear = graph_rewrite(linear, pm_prep_runtime, name="prepare runtime")

  exit(0)

  return linear

  linear = graph_rewrite(linear, pm_prep_runtime, name="prepare runtime")
  linear = graph_rewrite(linear, pm_lower_ops, name="lower ops into hcq ir")

  exit(0)

  linear = graph_rewrite(linear, pm_schedule_inner_sync, ctx=(deps_ctx:=DepsCtx()), walk=True, name="schedule inner sync")

  linear = graph_rewrite(linear, pm_add_finalizer, ctx=deps_ctx, walk=True, name="add finalizer")
  linear = graph_rewrite(linear, pm_add_inner_loads, ctx=(waited:=set()), walk=True, name="add loads", enter_calls=True)
  linear = graph_rewrite(linear, pm_add_inner_stores, ctx=waited, walk=True, name="add stores", enter_calls=True)
  linear = graph_rewrite(linear, pm_merge_queues, name="merge queues")
  linear = graph_rewrite(linear, pm_add_global_sync, ctx=set(), walk=True, name="add global sync", enter_calls=True)
  linear = graph_rewrite(linear, pm_annotate_devs, name="annotate devs")
  linear = graph_rewrite(linear, pm_replace_params, name="replace params")
  linear = graph_rewrite(linear, pm_encode_cmdbufs, walk=True, name="encode cmdbufs", enter_calls=True)
  # linear = graph_rewrite(linear, pm_lift_patches_to_cmdbuf, name="lift patches to cmdbuf", enter_calls=True)
  # linear = graph_rewrite(linear, pm_pack_placeholders, walk=True, name="pack placeholders")
  return graph_rewrite(linear, pm_hold_call_buffers, walk=True, name="hold call buffers")

@track_rewrites(lambda _,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_bufferize, bottom_up=True, walk=True, name="bufferize placeholders", enter_calls=True)
  linear = graph_rewrite(linear, pm_resolve_patches, bottom_up=False, name="simplify patches", enter_calls=True)
  linear = graph_rewrite(linear, pm_parametrize_host_buffers, walk=True, name="parametrize host buffers")
  return graph_rewrite(linear, pm_callify_hcq, name="callify hcq")
