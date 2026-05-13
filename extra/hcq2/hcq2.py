from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any, TYPE_CHECKING
import struct, functools, time, collections
from dataclasses import replace
if TYPE_CHECKING: from tinygrad.engine.realize import ExecContext
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, wait_cond, mv_address, round_up, DEBUG
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.dtype import dtypes
from dataclasses import dataclass, field
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, track_stats

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  timestamp_divider: float = 1000.0  # GPU timestamp counter ticks per microsecond; override per device

  def __init__(self, device:str, allocator:'HCQAllocator', compilers:list[type[Renderer]], runtime,
               kernargs_size=(16 << 20), can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    from extra.hcq2.graph.hcq import HCQ2Graph
    super().__init__(device, allocator, compilers, lambda *a, **kw: None, HCQ2Graph, arch=arch)

    self.kernargs_size = kernargs_size
    self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(kernargs_size, wrap=True)

  @functools.cached_property
  def kernargs_buf(self) -> Buffer:
    return Buffer(self.device, self.kernargs_size, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

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

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal._buf.cpu_view().mv.cast('Q')
    tl = self.timeline_value.as_memoryview(force_zero_copy=True).cast('Q')
    wait_cond(lambda: sig[0] >= tl[0] - 1, timeout_ms=3000, msg=f"{sig[0]} < {tl[0] - 1}")

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
    run_linear(UOp(Ops.LINEAR, dtypes.void, (su.copy_to_device(dst.device).call(UOp.from_buffer(dst), su),)), jit=True, update_stats=False)

  def _copyin(self, dest:HCQ2Buffer, src:memoryview):
    s = Buffer(self.dev.device, len(src), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    s._buf.cpu_view()[:len(src)] = src
    self._copy(self._wrap(self.dev.device, len(src), dest), s)

  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    d = Buffer(self.dev.device, len(dest), dtypes.uint8, options=BufferSpec(host=True), preallocate=True)
    self._copy(d, self._wrap(self.dev.device, len(dest), src))
    self.dev.synchronize()
    dest[:] = d._buf.cpu_view()[:len(dest)]

  def _as_buffer(self, buf): return buf.cpu_view().mv

# **************** lower context ****************

@dataclass
class HCQ2DeviceCtx:
  device:str                       # device name; resolve to instance via Device[device]
  kernargs_host:UOp                # UOp whose .buffer is dev.kernargs_buf (BUFFER UOp in runtime, PARAM in graph)
  kernargs_gpu:UOp                 # va_addr const of dev.kernargs_buf
  kernargs_allocator:BumpAllocator # runtime: dev's rotating; graph: fresh non-wrap

@dataclass
class HCQ2LowerCtx:
  name:str
  inputs:list[Buffer] = field(default_factory=list)
  holds:list[UOp] = field(default_factory=list)
  devs:dict[str, HCQ2DeviceCtx] = field(default_factory=dict)

  def host_param(self, buf:Buffer) -> UOp:
    if buf not in self.inputs: self.inputs.append(buf)
    return UOp.placeholder((buf.size,), buf.dtype, self.inputs.index(buf))

class HCQEncoder:
  def __init__(self, ctx:HCQ2LowerCtx, dev:HCQ2Compiled): self.ctx, self.dev, self.blob, self.patches, self.deps = ctx, dev, b'', [], set()

  @property
  def src(self) -> tuple[UOp, ...]: return tuple(self.patches + list(self.deps))

  def get_dev_addr(self, uop:UOp) -> sint|UOp:
    # unwrap transient AFTER on the value: deps flow into enc.deps separately, the outer wrapper never reaches the final graph
    while uop.op is Ops.AFTER:
      self.deps.update(uop.src[1:])
      uop = uop.src[0]
    self.deps.add(uop)
    return uop.buffer.get_buf(self.dev.device).va_addr if uop.op in (Ops.BUFFER, Ops.BUFFER_VIEW) else uop.ssimplify()

  def append(self, *data, dtype=dtypes.uint32):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{dtype.fmt}', d)
      elif d.op is Ops.CONST: self.blob += struct.pack(f'<{dtype.fmt}', d.arg)
      else:
        self.patches.append(UOp(Ops.PATCH, dtype, src=(d,), arg=len(self.blob)))
        self.blob += struct.pack(f'<{dtype.fmt}', 0)

  def q(self, *values): self.append(*values)

# **************** prep runtime ****************

pm_prep_runtime = PatternMatcher([
  # device-specific lowering of the program
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(), UPat(), UPat(Ops.BINARY)), name="p"),), name="c", allow_any_len=True),
    lambda ctx, c, p: c.replace(src=(Device[p.src[1].arg].pm_lower.rewrite(p, ctx),) + c.src[1:])),
])

# **************** lower kernargs ****************

def lower_kernargs(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  data, info = prg.arg
  # after amd_build_program, prg.src is (BUFFER_lib_gpu,); the buffer's device names the device
  dctx = ctx.devs[prg.src[0].buffer.device]

  enc = HCQEncoder(ctx, Device[dctx.device])
  for gi in info.globals: enc.append(enc.get_dev_addr(call.src[1+gi]), dtype=dtypes.uint64)
  for v in info.vars: enc.append(v, dtype=dtypes.uint32)

  args_off = dctx.kernargs_allocator.alloc(data.kernargs_alloc_size, 16)
  dctx.kernargs_host.buffer.view(len(enc.blob), dtypes.uint8, args_off).ensure_allocated().as_memoryview(force_zero_copy=True)[:] = enc.blob

  args_uop = (dctx.kernargs_gpu + args_off).after(dctx.kernargs_host.after(*tuple(p.replace(arg=p.arg+args_off) for p in enc.patches)))
  return call.replace(src=(prg.replace(src=prg.src + (args_uop,), arg=(data, info)),) + call.src[1:])

pm_lower_kernargs = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER),), name="prg"),), name="call", allow_any_len=True), lower_kernargs),
])

# **************** lower ops ****************

def lower_program(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  q = UOp(Ops.LINEAR, dtypes.void, (prg,), arg=(prg.src[0].buffer.device, "COMPUTE"))
  return UOp(Ops.LINEAR, dtypes.void, (q,), tag=call.tag)

def lower_copy(ctx:HCQ2LowerCtx, call:UOp, copy:UOp) -> UOp:
  dst, src = call.src[1], call.src[2]
  q = UOp(Ops.LINEAR, dtypes.void, (UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes),), arg=(dst.buffer.device, "COPY"))
  return UOp(Ops.LINEAR, dtypes.void, (q,), tag=call.tag)

pm_lower_ops = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER), UPat()), name="prg"),), name="call", allow_any_len=True), lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
])

# **************** split into queues ****************

def split_into_queues(ctx:HCQ2LowerCtx, outer:UOp) -> UOp:
  groups:dict[tuple, list[UOp]] = collections.defaultdict(list)
  for child in outer.src:
    wrapper = child.src[0] if child.op is Ops.AFTER else child
    for q in wrapper.src: groups[q.arg].extend(q.src)
  return outer.replace(src=tuple(UOp(Ops.LINEAR, dtypes.void, tuple(cmds), arg=k) for k, cmds in groups.items()))
pm_split_into_queues = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR, src=UPat(Ops.LINEAR)).or_after(), name="outer"), split_into_queues)])

# **************** add barriers ****************

def add_barriers(ctx:HCQ2LowerCtx, outer:UOp) -> UOp:
  def maybe_barrier(q:UOp) -> UOp:
    return q.replace(src=(UOp(Ops.BARRIER, dtypes.void), *q.src)) if q.arg[1] == "COMPUTE" else q
  return outer.replace(src=tuple(maybe_barrier(q) for q in outer.src))
pm_add_barriers = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"), add_barriers)])

# **************** add signals (runtime) ****************

def add_signals(ctx:HCQ2LowerCtx, outer:UOp) -> UOp:
  def wrap(q:UOp) -> UOp:
    (dev_name, qname), devs = q.arg, {q.arg[0]} | {u.buffer.device for u in q.toposort() if u.op in (Ops.BUFFER, Ops.BUFFER_VIEW)}
    sigs_tls = [(UOp.from_buffer(Device[d].timeline_signal), ctx.host_param(Device[d].timeline_value)) for d in sorted(devs) if d.startswith("AMD")]
    return q.replace(src=(*(s.wait(t[0]-1) for s,t in sigs_tls), *q.src, *(s.store(t[0]) for s,t in sigs_tls)), arg=qname)
  return outer.replace(src=tuple(wrap(q) for q in outer.src))
pm_add_signals = PatternMatcher([(UPat(Ops.LINEAR, src=UPat(Ops.LINEAR), name="outer"), add_signals)])

# **************** build host program ****************

def resolve_cmdbuf(ctx:HCQ2LowerCtx, blob:UOp) -> UOp:
  inner = blob.src[0] if blob.op is Ops.AFTER else blob
  dev_name, qtype = inner.tag

  # prepare the cmdbuf and make it a param
  bb = Buffer("CPU", len(inner.arg)//4, dtypes.uint32, preallocate=True)
  bb.copyin(memoryview(bytearray(inner.arg)))
  bb_param = ctx.host_param(bb)

  submit_cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(bb_param.after(*(blob.src[1:] if blob.op is Ops.AFTER else ())),),
                  arg=f"submit_{qtype.lower()}", tag=dev_name)

  # increment the timeline value
  tl = ctx.host_param(Device[dev_name].timeline_value)
  return tl.after(UOp(Ops.BARRIER, dtypes.void, src=(submit_cf,))).index(UOp.const(dtypes.int, 0), ptr=True).store(tl[0] + 1)

def resolve_patches(ctx:HCQ2LowerCtx, buf:UOp) -> UOp|None:
  inner = buf.src[0]

  # buffer is accessed from the launcher, so transform it to a host param
  if inner.op is Ops.BUFFER: inner = ctx.host_param(inner.buffer)

  return inner.after(*(inner.index(UOp.const(dtypes.int, p.arg//inner.dtype.base.itemsize), ptr=True).cast(p.dtype.ptr()).store(p.src[0].cast(p.dtype))
                           if p.op is Ops.PATCH else p for p in buf.src[1:]))

def resolve_ref_buffers(ctx:HCQ2LowerCtx, buf:UOp) -> UOp:
  if buf not in ctx.holds: ctx.holds.append(buf)
  return UOp(Ops.NOOP)

def hcq_callify(ctx:HCQ2LowerCtx, sink:UOp) -> UOp:
  call = to_program(sink, Device["CPU"].renderer).call(*[UOp.from_buffer(b, "CPU") if isinstance(b, Buffer) else b for b in ctx.inputs])
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=tuple(ctx.holds)),)) if ctx.holds else call

pm_create_host_sink = PatternMatcher([
  (UPat(Ops.LINEAR, name="l", allow_any_len=True), lambda ctx, l: UOp.sink(*l.src, arg=KernelInfo(name=ctx.name, estimates=Estimates()), tag=1))
])

# lower cmdbuf submits
pm_lower_cmdbufs = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(Ops.BINARY),), name="blob", allow_any_len=True), resolve_cmdbuf),
  (UPat(Ops.BINARY, name="blob"), resolve_cmdbuf),
])

# transform patches attached to buffers and params
pm_resolve_patches = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat((Ops.BUFFER, Ops.PARAM)),), name="buf", allow_any_len=True), resolve_patches)
])

# replace referenced buffers with noops
pm_resolve_ref_buffers = PatternMatcher([(UPat((Ops.BUFFER, Ops.BUFFER_VIEW), name="buf"), resolve_ref_buffers)])

pm_callify = PatternMatcher([(UPat(Ops.SINK, name="sink"), hcq_callify)])

def hcq_build_host_program(ctx:HCQ2LowerCtx, linear:UOp, ast:UOp, dev:HCQ2Compiled) -> UOp:
  sink = graph_rewrite(linear, pm_create_host_sink, ctx=ctx, name="hcq: create host sink", walk=True)
  sink = graph_rewrite(sink, pm_lower_cmdbufs, ctx=ctx, bottom_up=True, name="hcq: lower cmdbufs")
  sink = graph_rewrite(sink, pm_resolve_patches, ctx=ctx, bottom_up=True, name="hcq: resolve patches")
  sink = graph_rewrite(sink, pm_resolve_ref_buffers, ctx=ctx, bottom_up=True, name="hcq: resolve ref buffers")
  sink = graph_rewrite(sink, dev.pm_lower, ctx=ctx, name=f"hcq: device lower {dev.device}", walk=True)
  return graph_rewrite(sink, pm_callify, ctx=ctx, name="hcq: callify")

# **************** schedule ****************

@track_rewrites(name=lambda ctx,linear,ast,dev,**kw: f"hcq schedule {getattr(ast.arg, 'name', ast.op.name.lower())}")
def hcq_schedule(ctx:HCQ2LowerCtx, linear:UOp, ast:UOp, dev:HCQ2Compiled) -> UOp:
  linear = graph_rewrite(linear, pm_prep_runtime, ctx=ctx, name="hcq: prepare runtime")
  linear = graph_rewrite(linear, pm_lower_kernargs + pm_lower_ops, ctx=ctx, name="hcq: lower ops")
  linear = graph_rewrite(linear, pm_split_into_queues, ctx=ctx, name="hcq: split into queues")
  linear = graph_rewrite(linear, pm_add_barriers, ctx=ctx, name="hcq: add barriers", walk=True)
  linear = graph_rewrite(linear, pm_add_signals, ctx=ctx, name="hcq: add signals", walk=True)
  linear = graph_rewrite(linear, dev.pm_lower, ctx=ctx, name=f"hcq: encode cmdbuf {dev.device}", walk=True)
  return hcq_build_host_program(ctx, linear, ast, dev)

def _resolve_call(ctx:ExecContext, call:UOp, ast:UOp) -> UOp:
  from tinygrad.engine.realize import resolve_params
  return call.replace(src=(ast,) + tuple(resolve_params(call, ctx.input_uops)) + tuple(s for s in call.src[1:] if s.op is Ops.BIND))

def _run_host_call(ctx:ExecContext, call:UOp, dev:HCQ2Compiled, host_call:UOp, bufs:list[Buffer]) -> float:
  from tinygrad.engine.realize import run_linear
  with track_stats(ctx, call, dev.device, bufs, ctx.var_vals) as tm:
    st = time.perf_counter() if ctx.wait else 0.0
    run_linear(UOp(Ops.LINEAR, dtypes.void, (host_call,)), var_vals=ctx.var_vals, jit=True, update_stats=DEBUG>=3)
    if ctx.wait:
      dev.synchronize()
      tm[0] = time.perf_counter() - st
  return tm[0] if tm[0] is not None else 0.0

def hcq_exec_program(ctx:ExecContext, call:UOp, ast:UOp) -> float|None:
  if ast.src[1].arg.split(":")[0] != "AMD": return None
  dev, resolved_call = Device[ast.src[1].arg], _resolve_call(ctx, call, ast)
  hcq_ctx = HCQ2LowerCtx(name="submit_program", devs={dev.device: HCQ2DeviceCtx(device=dev.device,
    kernargs_host=UOp.from_buffer(dev.kernargs_buf, dev.device),
    kernargs_gpu=UOp.const(dtypes.uint64, dev.kernargs_buf.get_buf(dev.device).va_addr),
    kernargs_allocator=dev.kernargs_offset_allocator)})
  host_call = hcq_schedule(hcq_ctx, UOp(Ops.LINEAR, dtypes.void, (resolved_call,)), ast, dev)
  prg_bufs = [cast(Buffer, resolved_call.src[1+gi].buffer) for gi in ast.arg.globals]
  return _run_host_call(ctx, call, dev, host_call, prg_bufs)

def hcq_exec_copy(ctx:ExecContext, call:UOp, ast:UOp) -> float|None:
  if ast.src[1].arg.split(":")[0] != "AMD": return None
  dev, resolved_call = Device[ast.src[1].arg], _resolve_call(ctx, call, ast)
  hcq_ctx = HCQ2LowerCtx(name="submit_copy")
  src_buf = resolved_call.src[2].buffer
  try: src_buf.get_buf(dev.device)
  except Exception:
    (cpubuf := Buffer("CPU", src_buf.nbytes, dtypes.uint8, preallocate=True)).copyin(src_buf.ensure_allocated().as_memoryview())
    hcq_ctx.holds.append(buf_uop:=UOp.from_buffer(cpubuf, dev.device))
    resolved_call = resolved_call.replace(src=resolved_call.src[:2] + (buf_uop,) + resolved_call.src[3:])
  host_call = hcq_schedule(hcq_ctx, UOp(Ops.LINEAR, dtypes.void, (resolved_call,)), ast, dev)
  bufs = [cast(Buffer, resolved_call.src[1].buffer), cast(Buffer, resolved_call.src[2].buffer)]
  return _run_host_call(ctx, call, dev, host_call, bufs)

pm_hcq_exec = PatternMatcher([
  # TODO: use upat device=?
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True), hcq_exec_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True), hcq_exec_copy),
])
