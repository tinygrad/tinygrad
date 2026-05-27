from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any, TYPE_CHECKING
import struct, functools, time, collections, importlib, itertools
from dataclasses import replace
if TYPE_CHECKING: from tinygrad.engine.realize import ExecContext
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, mv_address, round_up, DEBUG, dedup
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.uop.symbolic import symbolic_simple, symbolic
from tinygrad.dtype import dtypes, DType
from dataclasses import dataclass, field
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, track_stats, get_call_arg_uops, resolve_params, pm_flatten_linear

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  timestamp_divider: float = 1000.0  # GPU timestamp counter ticks per microsecond; override per device

  def __init__(self, device:str, allocator:'HCQAllocator', compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    super().__init__(device, allocator, compilers, lambda *a, **kw: None, None, arch=arch)

  @functools.cached_property
  def timeline_signal(self) -> Buffer:
    return Buffer(self.device, 0x100, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)

  @functools.cached_property
  def timestamps_buf(self) -> Buffer:
    return Buffer(self.device, 0x100, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  @functools.cached_property
  def timeline_value(self) -> Buffer:
    buf = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    buf.as_memoryview(force_zero_copy=True).cast('Q')[0] = 1
    return buf

  @functools.cached_property
  def pm_bufferize(self) -> PatternMatcher:
    return PatternMatcher([
      (UPat(Ops.BUFFER, tag="timeline_signal"), lambda ctx: ctx.timeline_signal),
      (UPat(Ops.BUFFER, tag="timeline_value"), lambda ctx: ctx.timeline_value),
      (UPat(Ops.BUFFER, name="b"),
        lambda ctx, b: Buffer(ctx.device, b.arg, b.dtype, options=BufferSpec(host=True, uncached=True, cpu_access=True, nolru=True), preallocate=True)),
    ])

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal._buf.cpu_view().mv.cast('Q')
    tl = self.timeline_value.as_memoryview(force_zero_copy=True).cast('Q')
    st = time.perf_counter()
    while sig[0] < tl[0] - 1:
      if time.perf_counter() - st > (timeout or 3000) / 1000: self.on_device_hang()

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def _realloc(self, oldbuf:HCQ2Buffer|None, new_size:int, options:BufferSpec|None=None, force=False) -> tuple[HCQ2Buffer, bool]:
    if oldbuf is not None: self.allocator.free(oldbuf, oldbuf.size, options=options)
    try: buf, realloced = self.allocator.alloc(new_size, options=options), True
    except MemoryError:
      if force: raise
      buf, realloced = self.allocator.alloc(oldbuf.size if oldbuf is not None else new_size, options=options), False
    return buf, realloced

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
# 1.1. prep runtimes: staging copies

HCQ_DEVS = ("AMD",)
HCQ_P2P_DEVS = HCQ_DEVS + ("CPU",)

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  dst_dev, src_dev = dst.device.split(":")[0], src.device.split(":")[0]
  if not ((src_dev in HCQ_DEVS and dst_dev not in HCQ_P2P_DEVS) or (dst_dev in HCQ_DEVS and src_dev not in HCQ_P2P_DEVS)): return None
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
  if (pm:=get_pm_prep_program(dev.split(":")[0])) is None or (lowered:=pm.rewrite(prg)) is None: return None
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
# 2.1. lowering to hcq ir

def lower_program(call:UOp, prg:UOp) -> UOp:
  q = UOp(Ops.LINEAR, dtypes.void, (prg,), arg=(call.src[1].device, "COMPUTE"))
  return call.replace(src=(q,) + call.src[1:]).rtag('hcq')

def lower_copy(call:UOp, copy:UOp) -> UOp|None:
  dst, src = call.src[1], call.src[2]
  if (hcq_dev:=next((b.device for b in (dst, src) if b.device.split(":")[0] in HCQ_DEVS), None)) is None: return None

  q = UOp(Ops.LINEAR, dtypes.void, (UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes),), arg=(hcq_dev, "COPY"))
  return call.replace(src=(q,) + call.src[1:]).rtag('hcq')

pm_lower_ops = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.AFTER), UPat(Ops.AFTER)), name="prg"),), name="call", allow_any_len=True), lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
])

# *****************
# 2.2. queue split

# def split_into_queues(linear:UOp) -> UOp:
#   out = []
#   for k, grp in itertools.groupby(linear.src, lambda c: c.src[0].arg if c.op is Ops.CALL and c.src[0].op is Ops.LINEAR else None):
#     if k is None: out.extend(grp)
#     else:
#       calls = list(grp)
#       items = tuple(x for c in calls for x in c.src[0].src)
#       args = tuple(a for c in calls for a in c.src[1:])
#       out.append(calls[0].replace(src=(UOp(Ops.LINEAR, dtypes.void, items, arg=k),) + args))
#   return linear.replace(src=tuple(out))
# pm_split_into_queues = PatternMatcher([(UPat(Ops.LINEAR, name="linear"), split_into_queues)])

# *****************
# 2.3. barriers / signals / timeline inc

def add_barriers(call:UOp, q:UOp) -> UOp:
  return call.replace(src=(q.replace(src=(UOp(Ops.BARRIER, dtypes.void), *q.src)),) + call.src[1:])
pm_add_barriers = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.LINEAR, name="q"),), name="call", allow_any_len=True), add_barriers)])

def add_signals(call:UOp, q:UOp) -> UOp:
  sig = UOp.new_buffer(q.arg[0], 0x100, dtypes.uint8).rtag("timeline_signal")
  tl = UOp.new_buffer(q.arg[0], 1, dtypes.uint64).rtag("timeline_value").index(UOp.const(dtypes.int, 0))
  return call.replace(src=(q.replace(src=(sig.wait(tl-1), *q.src, sig.store(tl)), arg=q.arg),) + call.src[1:])
pm_add_signals = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.LINEAR, name="q"),), name="call", allow_any_len=True), add_signals)])

# *****************
# 3.1. encode cmdbufs

@functools.cache
def get_pm_lower(name:str) -> PatternMatcher|None:
  try:
    importlib.import_module(f'tinygrad.runtime.ops_{name.lower()}') # TODO: remove that
    return importlib.import_module(f'extra.hcq2.ops_{name.lower()}2').pm_lower
  except ImportError: return None

def encode_cmdbuf(call:UOp, q:UOp) -> UOp|None:
  if (pm:=get_pm_lower(call.src[1].device.split(":")[0])) is None or (encoded:=pm.rewrite(q)) is None: return None
  return call.replace(src=(encoded,) + call.src[1:])
pm_encode_cmdbufs = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.LINEAR, name="q"),), name="call", allow_any_len=True), encode_cmdbuf)])

# *****************
# 3.2. add timeline inc

def add_timeline_inc(call:UOp, s:UOp) -> UOp:
  tl = UOp.new_buffer(s.device, 1, dtypes.uint64).rtag("timeline_value")
  return call.replace(src=(tl.after(s).index(UOp.const(dtypes.int, 0), dtype=tl.dtype.ptr()).store(tl.index(UOp.const(dtypes.int, 0)) + 1),) + call.src[1:])
pm_add_timeline_inc = PatternMatcher([(UPat(Ops.CALL, tag="hcq", src=(UPat(name="s"),), name="call", allow_any_len=True), add_timeline_inc)])

# *****************
# 3.3. lift patches to the command buffer (root)

def lift_patches_to_cmdbuf(cmdbuf:UOp) -> UOp|None:
  if not (patches:=dedup(u for store in cmdbuf.src[1:] for u in store.toposort() if u.op is Ops.AFTER)): return None
  deps = tuple(d for p in patches for d in p.src[1:])
  return cmdbuf.replace(src=cmdbuf.src + deps).substitute({p: p.src[0] for p in patches})
pm_lift_patches_to_cmdbuf = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(Ops.BUFFER, tag={"compute", "copy"}),), allow_any_len=True, name="cmdbuf"), lift_patches_to_cmdbuf),
])

# *****************
# 4. bufferize placeholders: replace placeholders with real debuve buffers.

def bufferize_buf(buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  d = buf.src[1].arg
  uops = tuple(UOp.from_buffer((d:=Device[dev]).pm_bufferize.rewrite(buf, ctx=d), dev) for dev in (d if isinstance(d, tuple) else (d,)))
  return uops[0] if len(uops) == 1 else UOp(Ops.MSTACK, uops[0].dtype, uops)
pm_bufferize = PatternMatcher([(UPat(Ops.BUFFER, name="buf"), bufferize_buf)])

# *****************
# 5.1. capture buffers reachable from each hcq call as BIND, so resolve_patches doesn't drop their refs

def hold_call_buffers(call:UOp) -> UOp|None:
  if not (bufs:=tuple(dedup(u for u in call.src[0].toposort() if u.op is Ops.BUFFER))): return None
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=bufs),))
pm_hold_call_buffers = PatternMatcher([(UPat(Ops.CALL, tag="hcq", name="call"), hold_call_buffers)])

# *****************
# 5.2. resolve patches

def fold_blob_store(buf:UOp, blob:UOp) -> UOp:
  buf.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(buf:UOp, off:UOp, val:UOp) -> UOp:
  struct.pack_into(f'<{val.dtype.fmt}', buf.buffer.ensure_allocated()._buf.cpu_view().mv.cast('B'), off.arg * buf.dtype.base.itemsize, val.arg)
  return UOp(Ops.NOOP)

pm_resolve_patches = symbolic + PatternMatcher([
  (UPat(Ops.GETADDR, src=(UPat(Ops.SLICE, name="bv"), UPat(Ops.DEVICE, name="dev"))), # getaddr(slice(x)) -> offset+getaddr(x)
    lambda bv, dev: UOp(Ops.GETADDR, dtypes.uint64, src=(bv.src[0], dev)) + UOp.const(dtypes.uint64, bv.src[1].arg * bv.src[0].dtype.itemsize)),
  (UPat(Ops.GETADDR, src=(UPat(Ops.BUFFER, name="buf"), UPat(Ops.DEVICE)), name="g"),
    lambda buf, g: UOp.const(dtypes.uint64, buf.buffer.get_buf(g.src[1].arg).va_addr)),
  (UPat(Ops.GETADDR, src=(UPat.cvar("const"), UPat())), lambda const: const),

  (UPat(Ops.BUFFER, name="buf").store(UPat(Ops.BINARY, name="blob")), fold_blob_store),
  (UPat(Ops.BUFFER, name="buf").index(UPat.cvar("off")).or_casted().store(UPat.cvar("val")), fold_const_store),
])

# *****************
# 6. callify hcq programs

pm_fixup = PatternMatcher([ # TODO: this should gone?
  (UPat(Ops.CONST, name="c"), lambda c: c.replace(src=()) if len(c.src) else None),
])

def callify_hcq(call:UOp) -> UOp:
  bufs = dedup([u for u in call.src[0].toposort() if u.op is Ops.BUFFER])
  subs = {b: UOp.placeholder((b.arg,), b.dtype, i) for i, b in enumerate(bufs)}
  sink = UOp.sink(call.src[0].substitute(subs), arg=KernelInfo(name="hcq_submit", estimates=Estimates()), tag=1)
  return to_program(sink, Device["CPU"].renderer).call(*bufs, UOp(Ops.BIND, dtypes.void, src=call.src[1:]))
pm_callify_hcq = PatternMatcher([(UPat(Ops.CALL, tag="hcq", name="call"), callify_hcq)])

@track_rewrites(name=lambda **k: "Schedule HCQ")
def hcq_schedule(linear:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_insert_copy_staging + pm_flatten_linear, name="insert copy staging")
  linear = graph_rewrite(linear, pm_prep_runtime, name="prepare runtime")

  linear = graph_rewrite(linear, pm_lower_ops, name="lower ops into hcq ir")
  # linear = graph_rewrite(linear, pm_split_into_queues, name="split into queues")
  linear = graph_rewrite(linear, pm_add_barriers, walk=True, name="add barriers")
  linear = graph_rewrite(linear, pm_add_signals, walk=True, name="add signals")
  linear = graph_rewrite(linear, pm_encode_cmdbufs, walk=True, name="encode cmdbufs")
  linear = graph_rewrite(linear, pm_add_timeline_inc, walk=True, name="add timeline inc")
  linear = graph_rewrite(linear, pm_lift_patches_to_cmdbuf, name="lift patches to cmdbuf", enter_calls=True)

  # realize starts from here
  linear = graph_rewrite(linear, pm_bufferize, bottom_up=True, name="bufferize placeholders", enter_calls=True)
  linear = graph_rewrite(linear, pm_hold_call_buffers, walk=True, name="hold call buffers")
  linear = graph_rewrite(linear, pm_resolve_patches, bottom_up=False, name="simplify patches", enter_calls=True)
  linear = graph_rewrite(linear, pm_fixup, bottom_up=False, name="fixup", enter_calls=True)
  linear = graph_rewrite(linear, pm_callify_hcq, name="parametrize host buffers")

  return linear
