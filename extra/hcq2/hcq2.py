from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any, TYPE_CHECKING
import struct, functools, time, collections, importlib, itertools, weakref
from dataclasses import replace
if TYPE_CHECKING: from tinygrad.engine.realize import ExecContext
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, mv_address, round_up, DEBUG, dedup, pluralize
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites, GroupOp
from tinygrad.uop.symbolic import symbolic_simple, symbolic
from tinygrad.dtype import dtypes, DType
from dataclasses import dataclass, field
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, track_stats, get_call_arg_uops, resolve_params, pm_flatten_linear
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
        Buffer(ctx.device, b.arg, b.dtype, options=BufferSpec(host=True, uncached=True, cpu_access=True, nolru=True))), # TODO: remove nolru
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
    self.dev.iface.dev_impl.mm.unmap_range(int(mb.va_addr), round_up(mb.size, 0x1000))

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

def unwrap_after(uop):
  while uop.op is Ops.AFTER: uop = uop.src[0]
  return uop

def make_mstack(uops): return uops[0] if len(uops) == 1 else UOp(Ops.MSTACK, uops[0].dtype, tuple(uops))

def make_signal(devs, queue=None, sentinel=False):
  return UOp.new_buffer(devs, 1, dtypes.uint64).rtag("sentinel_signal" if sentinel else (queue, "timeline_signal") if queue else "timeline_signal")
def make_signal_value(devs, queue=None): return UOp.new_buffer(devs, 1, dtypes.uint64).rtag((queue, "timeline_value") if queue else "timeline_value")

class HCQEncoder:
  def __init__(self): self.blob, self.patches = b'', []

  def get_dev_addr(self, uop:UOp) -> UOp:
    if unwrap_after(uop).op not in (Ops.BUFFER, Ops.SLICE, Ops.BINARY, Ops.MSTACK, Ops.MSELECT): return uop
    return UOp(Ops.GETADDR, dtypes.uint64, src=(uop, UOp(Ops.DEVICE, arg=self.dev.device)))

  def append(self, *data, dtype=dtypes.uint32):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{dtype.fmt}', d)
      else:
        self.patches.append((len(self.blob), self.get_dev_addr(d), dtype))
        self.blob += struct.pack(f'<{dtype.fmt}', 0)

  def q(self, *values): self.append(*values)

  def uop(self, dev:str|tuple[str, ...], tag:str|None=None) -> UOp:
    buf = UOp.new_buffer(dev, len(self.blob), dtypes.uint8)
    if tag: buf = buf.rtag(tag)
    blob_uop = UOp(Ops.BINARY, dtypes.void, src=(), arg=self.blob)
    stores = [buf.index(UOp.const(dtypes.int, off), dtype=buf.dtype.ptr()).cast(dt.ptr()).store(val.cast(dt)) for off, val, dt in self.patches]
    return buf.after(buf.store(blob_uop), *stores)

# *****************
# 0. helpers

HCQ_DEVS = frozenset(("AMD",))
HCQ_P2P_DEVS = HCQ_DEVS | frozenset(("CPU",))

def to_tuple(d): return d if isinstance(d, tuple) else (d,)

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

# *****************
# 1.1. prep runtimes: staging copies

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS) and not all_devices_in(b.device, HCQ_P2P_DEVS)

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  stage = UOp.new_buffer("CPU", src.buffer.nbytes, dtypes.uint8)
  return UOp(Ops.LINEAR, dtypes.void, (src.copy_to_device("CPU").call(stage, src), stage.copy_to_device(dst.device).call(dst, stage)))
pm_insert_copy_staging = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy)])

# *****************
# 1.2. prep runtimes: programs/kernargs

@functools.cache
def get_pm_prep_program(name:str) -> PatternMatcher|None:
  try:
    importlib.import_module(f'tinygrad.runtime.ops_{name.lower()}') # TODO: remove that
    return importlib.import_module(f'extra.hcq2.ops_{name.lower()}2').pm_prep_program
  except ImportError: return None

def prep_program(call:UOp, prg:UOp) -> UOp|None:
  dev = call.src[1].device
  if (pm:=get_pm_prep_program(to_tuple(dev)[0].split(":")[0])) is None or (lowered:=pm.rewrite(prg)) is None: return None

  data, image_bytes = lowered
  buf = UOp.new_buffer(dev, len(image_bytes), dtypes.uint8).rtag("program")
  blob = UOp(Ops.BINARY, dtypes.void, src=(), arg=image_bytes)
  return call.replace(src=(prg.replace(src=(buf.after(buf.store(blob)),), arg=(data, prg.arg)),) + call.src[1:])

def prep_kernargs(call:UOp, prg:UOp) -> UOp:
  data, info = prg.arg
  patches = [(i*dtypes.uint64.itemsize, UOp(Ops.GETADDR, dtypes.uint64, src=(call.src[1+gi], UOp(Ops.DEVICE, arg=call.src[1+gi].device))),
              dtypes.uint64) for i,gi in enumerate(info.globals)] \
          + [(len(info.globals)*dtypes.uint64.itemsize + i*dtypes.uint32.itemsize, v, dtypes.uint32) for i,v in enumerate(info.vars)]

  buf = UOp.new_buffer(call.src[1].device, data.kernargs_alloc_size, dtypes.uint8).rtag("kernargs")
  kernargs = buf.after(*tuple(buf.index(UOp.const(dtypes.int, o), dtype=buf.dtype.ptr()).cast(dt.ptr()).store(val.cast(dt)) for o, val, dt in patches))

  return call.replace(src=(prg.replace(src=prg.src + (kernargs,), arg=(data, info)),) + call.src[1:])

pm_prep_runtime = PatternMatcher([
  # bind generic PROGRAM device to the call's actual dev(s), then run device-specific lowering
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"),),
    name="call", allow_any_len=True), prep_program),

  # lower kernargs (PROGRAM.src[0] is now AFTER(BUFFER, COPY) — the lowered program image)
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.AFTER),), name="prg"),), name="call", allow_any_len=True), prep_kernargs),
])

# *****************
# 2. lowering to hcq ir

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  devs:tuple[str, ...] = to_tuple(devs)
  cmds = tuple([cmd.replace(arg=(devs, queue)).rtag("hcq_cmd") if cmd.op is Ops.CALL else cmd for cmd in cmds])
  queue = UOp(Ops.LINEAR, dtypes.void, src=cmds, arg=(devs, queue))
  return UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(queue,), arg="submit")

def lower_program(call:UOp, prg:UOp) -> UOp:
  return make_submit(prg.call(*call.src[1:]), devs=call.src[1].device, queue="COMPUTE:0").sink().call().rtag("hcq")

def lower_copy(call:UOp, copy:UOp) -> UOp|None:
  dst, src = call.src[1], call.src[2]
  if (hcq_dev:=next((b.device for b in (dst, src) if b.device.split(":")[0] in HCQ_DEVS), None)) is None: return None

  cp_op = UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes)
  return make_submit(cp_op.call(*call.src[1:]), devs=hcq_dev, queue="COPY:0").sink().call().rtag("hcq")

pm_lower_ops = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.AFTER), UPat(Ops.AFTER)), name="prg"),), name="call", allow_any_len=True), lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
])

# *****************
# 3.1. deps tracking
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

@dataclass
class DepsCtx:
  deps:DepsTracker = field(default_factory=DepsTracker)
  opid:itertools.count = field(default_factory=lambda: itertools.count(0))
  last_per_queue:weakref.WeakValueDictionary[tuple[Any, str], UOp] = field(default_factory=weakref.WeakValueDictionary)

def schedule_inner_sync(ctx:DepsCtx, call:UOp) -> UOp:
  refs = [b.buffer for b in get_call_arg_uops(call)]
  write_bufs = ast.arg[1].outs if (ast:=call.src[0]).op is Ops.PROGRAM else (0,)

  # tag carries (queue arg, opid)
  ctx.last_per_queue[call.arg[0]] = (op:=call.src[0].rtag((call.arg, next(ctx.opid))))

  deps = []
  for lane in range(len(refs[0].bufs) if isinstance(refs[0], MultiBuffer) else 1):
    deps += ctx.deps.access_resources([b.bufs[lane] if isinstance(b, MultiBuffer) else b for b in refs], write_bufs, op)
  return op.after(*dps).rtag("deps") if (dps:=dedup(deps)) else op
pm_schedule_inner_sync = PatternMatcher([(UPat(Ops.CALL, tag="hcq_cmd", name="call", allow_any_len=True), schedule_inner_sync)])

# *****************
# 3.2. finalizer

def make_finalizer(queues:list[UOp], nbump:int) -> UOp:
  devs = tuple(dedup([d for q in queues for d in to_tuple(q.tag[0][0])]))
  zero = UOp.const(dtypes.int, 0)
  tl = make_signal_value(devs)

  submit = make_submit(make_signal(devs).store(tl.index(zero) + 1), devs=devs, queue="COMPUTE:0")

  upd = [(tl, 1)] + [(make_signal_value(devs, queue=qn), nbump) for qn in dedup([q.tag[0][1] for q in queues])]
  return UOp.barrier(*[s.after(submit).index(zero, dtype=s.dtype.ptr()).store(s.index(zero) + inc) for s, inc in upd]).sink().call().rtag("hcq")

def add_finalizer(ctx:DepsCtx, linear:UOp) -> UOp:
  parts:dict[str, list[UOp]] = collections.defaultdict(list)
  for d, q in ctx.last_per_queue.items(): parts[d[0].split(':')[0]].append(q)

  nbump = next(ctx.opid)
  return linear.replace(src=linear.src + tuple([make_finalizer(queues, nbump) for queues in parts.values()]))
pm_add_finalizer = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), add_finalizer)])

# *****************
# 3.3. lower loads/stores

def add_loads(ctx:set[int], deps:UOp) -> UOp:
  cur_devs = to_tuple((cur:=deps.src[0]).tag[0][0])

  waits = []
  for (devs, queue), opid in [dq.tag for dq in deps.src[1:]]:
    ctx.add(opid) # mark op to update signal.

    sig = make_mstack([make_signal(d, queue=queue, sentinel=d not in devs) for d in cur_devs])
    val = make_signal_value(cur_devs, queue=queue).index(UOp.const(dtypes.int, 0))
    waits.append(sig.wait(val + opid))
  return UOp(Ops.LINEAR, dtypes.void, (*waits, cur), arg=cur.tag[0])
pm_add_inner_loads = PatternMatcher([(UPat(Ops.AFTER, tag="deps", name="deps"), add_loads)])

def add_stores(ctx:set[int], submit:UOp, q:UOp) -> UOp:
  new_src = []
  for op in q.src:
    new_src.append(op.rtag(None) if op.tag else op)
    if op.tag and op.tag[1] in ctx:
      (devs, queue), opid = op.tag
      new_src.append(make_signal(devs, queue=queue).store(make_signal_value(devs, queue=queue).index(UOp.const(dtypes.int, 0)) + opid))
  return submit.replace(src=(q.replace(src=tuple(new_src)),))
pm_add_inner_stores = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_stores)])

def add_global_sync(ctx:set[tuple[str, ...]], submit:UOp, q:UOp) -> UOp|None:
  if (devs:=q.arg[0]) in ctx: return None
  ctx.add(devs)

  # some devices from a command buffer might be used for the first time this schedule, so we wait for their global timeline epoch.
  wait = make_signal(devs).wait(make_signal_value(devs).index(UOp.const(dtypes.int, 0)) - 1)
  return submit.replace(src=(q.replace(src=(UOp(Ops.BARRIER, dtypes.void), wait, *q.src)),))
pm_add_global_sync = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit", src=(UPat(Ops.LINEAR, name="q"),), name="submit"), add_global_sync)])

# *****************
# 4.1. encode cmdbufs

@functools.cache
def get_pm_lower(name:str) -> PatternMatcher|None:
  try:
    importlib.import_module(f'tinygrad.runtime.ops_{name.lower()}') # TODO: remove that
    return importlib.import_module(f'extra.hcq2.ops_{name.lower()}2').pm_lower
  except ImportError: return None

def encode_cmdbuf(submit:UOp, lin:UOp) -> UOp|None:
  if (pm:=get_pm_lower(to_tuple(lin.arg[0])[0].split(":")[0])) is None: return None
  return pm.rewrite(submit)
pm_encode_cmdbufs = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit", src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), encode_cmdbuf)])

# *****************
# 4.2. lift patches to the command buffer (root)

def lift_patches_to_cmdbuf(cmdbuf:UOp) -> UOp|None:
  if not (patches:=dedup(u for store in cmdbuf.src[1:] for u in store.toposort() if u.op is Ops.AFTER)): return None
  deps = tuple(d for p in patches for d in p.src[1:])
  return cmdbuf.replace(src=cmdbuf.src + deps).substitute({p: p.src[0] for p in patches})
pm_lift_patches_to_cmdbuf = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, tag={"compute", "copy"}),), allow_any_len=True, name="cmdbuf"), lift_patches_to_cmdbuf),
])

# *****************
# 5. bufferize placeholders: replace placeholders with real buffers.

def bufferize_buf(buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  uops = tuple(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=dv), dev) for dev in to_tuple(buf.src[1].arg))
  return make_mstack(uops)
pm_bufferize = PatternMatcher([(UPat(Ops.BUFFER, name="buf"), bufferize_buf)])

# *****************
# 6.1. capture buffers reachable from each hcq call as BIND, so we don't drop their refs

def hold_call_buffers(call:UOp) -> UOp|None:
  if not (bufs:=tuple(dedup(u for u in call.src[0].toposort() if u.op is Ops.BUFFER and u not in call.src))): return None
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=bufs),))
pm_hold_call_buffers = PatternMatcher([(UPat(Ops.CALL, tag="hcq", name="call"), hold_call_buffers)])

# *****************
# 6.2. resolve patches

def push_stack(op, s): return UOp(Ops.STACK, op.dtype.scalar().vec(len(s.src)),
  tuple(op.replace(dtype=op.dtype.scalar(), src=tuple(x if y is s else y for y in op.src)) for x in s.src))

def fold_blob_store(buf:UOp, blob:UOp) -> UOp:
  for b in (buf.src if buf.op is Ops.MSTACK else (buf,)): b.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(buf:UOp, off:UOp, val:UOp) -> UOp:
  for b, v in zip((buf.src if buf.op is Ops.MSTACK else (buf,)), (val.src if val.op is Ops.STACK else (val,))):
    struct.pack_into(f'<{v.dtype.fmt}', b.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B'), off.arg * b.dtype.base.itemsize, v.arg)
  return UOp(Ops.NOOP)

def resolve_getaddr(buf:UOp, g:UOp) -> UOp:
  if isinstance(b:=buf.buffer, Buffer): return UOp.const(dtypes.uint64, b.get_buf(g.src[1].arg).va_addr)
  return UOp(Ops.STACK, dtypes.uint64.vec(len(b.bufs)), tuple(UOp.const(dtypes.uint64, x.ensure_allocated()._buf.va_addr) for x in b.bufs))

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU, src=[UPat(Ops.STACK, name="s"), UPat(Ops.CONST)], name="op"), push_stack),
  (UPat(Ops.CAST, src=(UPat(Ops.STACK, name="s"),), name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(Ops.SLICE, name="bv"), UPat(Ops.DEVICE, name="dev"))), # getaddr(slice(x)) -> offset+getaddr(x)
    lambda bv, dev: UOp(Ops.GETADDR, dtypes.uint64, src=(bv.src[0], dev)) + UOp.const(dtypes.uint64, bv.src[1].arg * bv.src[0].dtype.itemsize)),
  (UPat(Ops.GETADDR, src=(UPat({Ops.BUFFER, Ops.MSTACK, Ops.MSELECT}, name="buf"), UPat(Ops.DEVICE)), name="g"), resolve_getaddr),

  # folders
  (UPat({Ops.BUFFER, Ops.MSTACK}, name="buf").store(UPat(Ops.BINARY, name="blob")), fold_blob_store),
  (UPat({Ops.BUFFER, Ops.MSTACK}, name="buf").index(UPat.cvar("off")).or_casted().store(UPat.any(UPat.cvar("val"), UPat(Ops.STACK, name="val"))),
    fold_const_store),
]) + symbolic_simple

# *****************
# 7. callify hcq programs

def to_param(bufs:list[UOp], ref:UOp) -> UOp:
  bufs.append(ref)
  return UOp.placeholder((ref.buffer.size,), ref.dtype, len(bufs)-1)
pm_to_param = PatternMatcher([(UPat({Ops.MSELECT, Ops.MSTACK, Ops.BUFFER}, name="r"), lambda ctx, r: to_param(ctx, r))])

def parametrize_host_buffers(call:UOp) -> UOp:
  body = graph_rewrite(call.src[0], pm_to_param, ctx=(bufs:=[]), bottom_up=True, name="parametrize host buffers")
  return call.replace(src=(body, *bufs) + call.src[1:], tag="hcq_param")
pm_parametrize_host_buffers = PatternMatcher([(UPat(Ops.CALL, tag="hcq", name="call"), parametrize_host_buffers)])

def callify_hcq(call:UOp) -> UOp:
  sink = UOp.sink(call.src[0], arg=KernelInfo(name="hcq_submit", estimates=Estimates()), tag=1)
  return to_program(sink, Device["CPU"].renderer).call(*call.src[1:])
pm_callify_hcq = PatternMatcher([(UPat(Ops.CALL, tag="hcq_param", name="call"), callify_hcq)])

@track_rewrites(lambda _,ret: f"HCQ Schedule {pluralize('Kernel', len(ret.src))}")
def hcq_schedule(linear:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_insert_copy_staging + pm_flatten_linear, name="insert copy staging")
  linear = graph_rewrite(linear, pm_prep_runtime, name="prepare runtime")

  linear = graph_rewrite(linear, pm_lower_ops, name="lower ops into hcq ir")
  linear = graph_rewrite(linear, pm_schedule_inner_sync, ctx=(deps_ctx:=DepsCtx()), walk=True, name="schedule inner sync", enter_calls=True)
  linear = graph_rewrite(linear, pm_add_finalizer, ctx=deps_ctx, walk=True, name="add finalizer")
  linear = graph_rewrite(linear, pm_add_inner_loads + pm_flatten_linear, ctx=(waited:=set()), walk=True, name="add loads", enter_calls=True)
  linear = graph_rewrite(linear, pm_add_inner_stores, ctx=waited, walk=True, name="add stores", enter_calls=True)
  linear = graph_rewrite(linear, pm_add_global_sync, ctx=set(), walk=True, name="add global sync", enter_calls=True)
  linear = graph_rewrite(linear, pm_encode_cmdbufs, walk=True, name="encode cmdbufs", enter_calls=True)
  linear = graph_rewrite(linear, pm_lift_patches_to_cmdbuf, name="lift patches to cmdbuf", enter_calls=True)

  # realize starts from here
  linear = graph_rewrite(linear, pm_bufferize, bottom_up=True, name="bufferize placeholders", enter_calls=True)
  linear = graph_rewrite(linear, pm_hold_call_buffers, walk=True, name="hold call buffers")
  linear = graph_rewrite(linear, pm_resolve_patches, bottom_up=False, name="simplify patches", enter_calls=True)
  linear = graph_rewrite(linear, pm_parametrize_host_buffers, name="parametrize host buffers")
  linear = graph_rewrite(linear, pm_callify_hcq, name="callify hcq")

  return linear
