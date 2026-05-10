from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any
import struct, functools
from dataclasses import replace
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, wait_cond
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.dtype import dtypes
from dataclasses import dataclass, field
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.renderer import Renderer
from tinygrad.engine.realize import get_runtime, pm_flatten_linear
from tinygrad.codegen import to_program
from tinygrad.renderer import Estimates

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQProgram:
  def __init__(self, dev:HCQ2Compiled, name:str, lib:bytes, *aux, runtimevars=None, prg:UOp|None=None, **kwargs):
    assert prg is not None
    self.dev, self.name, self.prg = dev, name, prg
    n_bufs = max(prg.arg.globals, default=-1) + 1
    ctx = HCQ2LowerCtx(dev=dev, name=f"submit_{name}")
    self.buf_addrs = Buffer("CPU", n_bufs+1, dtypes.uint64, preallocate=True)
    buf_addrs = ctx.host_param(self.buf_addrs)
    ctx.kernargs, self.kernargs_i = ctx.host_param(dev.kernargs_buf, dev_uop=buf_addrs[0]), ctx.inputs.index(dev.kernargs_buf)
    host_prg = _hcq_schedule(dev, ctx, UOp(Ops.LINEAR, dtypes.void, (prg.call(*[buf_addrs[i+1] for i in range(n_bufs)]),), arg="COMPUTE"), prg).src[0]
    self.host_rt, self.host_globals = get_runtime("CPU", host_prg), host_prg.arg.globals
    self.host_inputs, self.kernargs_alloc_size = ctx.inputs, max(ctx.kernargs_allocator.ptr, 1)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int|None,...]=(),
               wait=False, timeout:int|None=None):
    # allocate kernargs and fill buf_addrs with gpu addresses
    self.host_inputs[self.kernargs_i] = self.dev.kernargs_buf.view(self.kernargs_alloc_size, dtypes.uint8,
      self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8)).ensure_allocated()
    addrs = self.buf_addrs.as_memoryview(force_zero_copy=True).cast('Q')
    addrs[0] = self.host_inputs[self.kernargs_i].get_buf(self.dev.device).va_addr
    for j, gi in enumerate(self.prg.arg.globals): addrs[gi+1] = bufs[j].va_addr
    self.host_rt(*[self.host_inputs[i].get_buf("CPU") for i in self.host_globals], vals=vals)
    if wait: self.dev.synchronize(timeout)
    return None

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  signal_pool: list[Buffer] = []

  def __init__(self, device:str, allocator:HCQAllocatorBase, compilers:list[type[Renderer]], runtime,
               kernargs_size=(16 << 20), sigalloc_size=0x1000, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.sigalloc_size, self.kernargs_size = sigalloc_size, kernargs_size
    self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(kernargs_size, wrap=True)

  @functools.cached_property
  def kernargs_buf(self) -> Buffer:
    return Buffer(self.device, self.kernargs_size, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  @functools.cached_property
  def timeline_signal(self) -> Buffer: return self.new_signal()

  @functools.cached_property
  def timeline_value(self) -> Buffer:
    buf = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    buf.as_memoryview(force_zero_copy=True).cast('Q')[0] = 1
    return buf

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal.as_memoryview(force_zero_copy=True).cast('Q')
    tl = self.timeline_value.as_memoryview(force_zero_copy=True).cast('Q')
    wait_cond(lambda: sig[0] >= tl[0] - 1, timeout_ms=3000, msg=f"{sig[0]} < {tl[0] - 1}")

  def new_signal(self) -> Buffer:
    if not HCQ2Compiled.signal_pool:
      alc = Buffer(self.device, self.sigalloc_size, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
      HCQ2Compiled.signal_pool += [alc.view(16, dtypes.uint8, off).ensure_allocated() for off in range(0, self.sigalloc_size, 16)]
    return HCQ2Compiled.signal_pool.pop()

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
  def __init__(self, va_addr:sint, size:int, meta:Any=None, _base:HCQ2Buffer|None=None, view:MMIOInterface|None=None,
               owner:HCQ2Compiled|None=None):
    self.va_addr, self.size, self.meta, self._base, self.view, self.owner = va_addr, size, meta, _base, view, owner

  def offset(self, offset:int=0, size:int|None=None) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def base(self) -> HCQ2Buffer: return self._base or self

class HCQAllocatorBase(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  """
  A base allocator class compatible with the HCQ (Hardware Command Queue) API.

  This class implements basic copy operations following the HCQ API, utilizing both types of `HWQueue`.
  """

  def __init__(self, dev:HCQDeviceType, batch_size:int=(2 << 20), batch_cnt:int=32, copy_bufs=None, max_copyout_size:int|None=None, **kwargs):
    super().__init__(dev, **kwargs)
    self.b = copy_bufs or [self._alloc(batch_size, BufferSpec(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next, self.max_copyout_size = [0] * len(self.b), 0, max_copyout_size

  def _map(self, buf:HCQ2Buffer) -> HCQ2Buffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)

  def _as_buffer(self, buf): return buf.cpu_view().mv

class HCQAllocator(HCQAllocatorBase, Generic[HCQDeviceType]):
  def _copyin(self, dest:HCQ2Buffer, src:memoryview):
    self.dev.synchronize()
    dest.cpu_view().mv[:src.nbytes] = bytes(src)
  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    self.dev.synchronize()
    dest[:] = bytes(src.cpu_view().mv[:dest.nbytes])
  def _transfer(self, dest:HCQ2Buffer, src:HCQ2Buffer, sz:int, src_dev:HCQDeviceType, dest_dev:HCQDeviceType):
    cast(HCQAllocator, src_dev.allocator)._map(dest)
    dest.cpu_view().mv[:sz] = bytes(src.cpu_view().mv[:sz])

# **************** lower context ****************

@dataclass
class HCQ2LowerCtx:
  dev:HCQ2Compiled
  kernargs:UOp|None = None
  kernargs_allocator:BumpAllocator = field(default_factory=lambda: BumpAllocator(0x1000, wrap=False))
  inputs:list[Buffer] = field(default_factory=list)
  holds:list[UOp] = field(default_factory=list)
  param2dev:dict[UOp, UOp] = field(default_factory=dict)
  name:str = "hcq_submit"

  def host_param(self, buf:Buffer, dev_uop:UOp|None=None) -> UOp:
    # dev_uop overrides the encoded device uop
    if buf not in self.inputs: self.inputs.append(buf)
    param = UOp.placeholder((buf.size,), buf.dtype, self.inputs.index(buf))
    if dev_uop is not None: self.param2dev[param] = dev_uop
    return param

  def as_dev_uop(self, uop:UOp) -> UOp|int:
    if uop.op in (Ops.BUFFER, Ops.BUFFER_VIEW): return uop.buffer.get_buf(self.dev.device).va_addr
    return self.param2dev.get(uop, uop)

class HCQEncoder:
  def __init__(self, ctx:HCQ2LowerCtx): self.ctx, self.dev, self.blob, self.patches, self.deps = ctx, ctx.dev, b'', [], set()

  @property
  def src(self) -> tuple[UOp, ...]: return tuple(self.patches + list(self.deps))

  def get_dev_addr(self, uop:UOp) -> sint|UOp:
    self.deps.add(uop) # uops used to encode this blob are referenced as sources
    while uop.op in (Ops.AFTER,): uop = uop.src[0]
    return self.ctx.as_dev_uop(uop)

  def append(self, *data, dtype=dtypes.uint32):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{dtype.fmt}', d)
      elif d.op is Ops.CONST: self.blob += struct.pack(f'<{dtype.fmt}', d.arg)
      else:
        self.patches.append(UOp(Ops.PARAM, dtype, src=(d,), arg=len(self.blob)))
        self.blob += struct.pack(f'<{dtype.fmt}', 0)

  def q(self, *values): self.append(*values)

# **************** prep runtime ****************

def do_init_args(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  data, info = prg.arg

  enc = HCQEncoder(ctx)
  for gi in info.globals: enc.append(call.src[1+gi], dtype=dtypes.uint64)
  for v in info.vars: enc.append(v, dtype=dtypes.uint32)

  args_off = ctx.kernargs_allocator.alloc(data.kernargs_alloc_size, 16)

  # attach kernargs patches
  assert ctx.kernargs is not None
  args_uop = ctx.kernargs.after(*tuple(p.replace(arg=p.arg+args_off) for p in enc.patches))
  return call.replace(src=(prg.replace(src=prg.src + (args_uop,), arg=(data, info, args_off)),) + call.src[1:])

pm_prep_runtime = PatternMatcher([
  # device-specific lowering of the program
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"),),
        name="call", allow_any_len=True), lambda ctx,call,prg: call.replace(src=(ctx.dev.pm_lower.rewrite(prg, ctx),) + call.src[1:])),
  # init args for programs
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER),), name="prg"),), name="call", allow_any_len=True), do_init_args),
])

# **************** lower hcq ****************

def lower_program(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  sig, tl = UOp.from_buffer(ctx.dev.timeline_signal), ctx.host_param(ctx.dev.timeline_value)
  return UOp.linear(sig.wait(tl[0] - 1), UOp(Ops.BARRIER, dtypes.void), prg, sig.store(tl[0]))

def lower_copy(ctx:HCQ2LowerCtx, call:UOp, copy:UOp) -> UOp:
  dst, src, dev = call.src[1], call.src[2], ctx.dev
  src_dev = Device[src.device]
  devs = [dev, src_dev] if src_dev is not dev else [dev]
  sigs_tls = [(UOp.from_buffer(d.timeline_signal), ctx.host_param(d.timeline_value)) for d in devs]
  return UOp.linear(*[s.wait(t[0] - 1) for s,t in sigs_tls], UOp(Ops.BARRIER, dtypes.void),
                    UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes),
                    *[s.store(t[0]) for s,t in sigs_tls])

# lower to hcq-specific commands
pm_hcq_lower = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER), UPat()), name="prg"),), name="call", allow_any_len=True), lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), lower_copy),
])

# **************** build host program ****************

def resolve_cmdbuf(ctx:HCQ2LowerCtx, blob:UOp) -> UOp:
  inner = blob.src[0] if blob.op is Ops.AFTER else blob

  # prepare the cmdbuf and make it a param
  bb = Buffer("CPU", len(inner.arg)//4, dtypes.uint32, preallocate=True)
  bb.copyin(memoryview(bytearray(inner.arg)))
  bb_param = ctx.host_param(bb)

  submit_cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(bb_param.after(*(blob.src[1:] if blob.op is Ops.AFTER else ())),),
                  arg=f"submit_{inner.tag.lower()}")

  # increment the timeline value
  tl = ctx.host_param(ctx.dev.timeline_value)
  return tl.after(UOp(Ops.BARRIER, dtypes.void, src=(submit_cf,))).index(UOp.const(dtypes.int, 0), ptr=True).store(tl[0] + 1)

def resolve_patches(ctx:HCQ2LowerCtx, buf:UOp) -> UOp|None:
  inner = buf.src[0]

  # buffer is accessed from the launcher, so transform it to a param
  if inner.op is Ops.BUFFER: inner = ctx.param(inner)

  return inner.after(*(inner.index(UOp.const(dtypes.int, p.arg//inner.dtype.base.itemsize), ptr=True).cast(p.dtype.ptr()).store(p.src[0].cast(p.dtype))
                           if p.op is Ops.PARAM and len(p.src) == 1 else p for p in buf.src[1:]))

def resolve_ref_buffers(ctx:HCQ2LowerCtx, buf:UOp) -> UOp:
  if buf not in ctx.holds: ctx.holds.append(buf)
  return UOp(Ops.NOOP)

def hcq_callify(ctx:HCQ2LowerCtx, sink:UOp) -> UOp:
  call = to_program(sink, Device["CPU"].renderer).call(*[UOp.from_buffer(b, "CPU") if isinstance(b, Buffer) else b for b in ctx.inputs])
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=tuple(ctx.holds)),)) if ctx.holds else call

pm_create_host_sink = PatternMatcher([
  (UPat(Ops.LINEAR, name="l", allow_any_len=True), lambda ctx, l: UOp.sink(*l.src, arg=KernelInfo(name=ctx.name), tag=1))
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
pm_resolve_ref_buffers = PatternMatcher([(UPat(Ops.BUFFER, name="buf"), resolve_ref_buffers)])

pm_callify = PatternMatcher([(UPat(Ops.SINK, name="sink"), hcq_callify)])

def hcq_build_host_program(ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  sink = graph_rewrite(linear, pm_create_host_sink, ctx=ctx, name="hcq: create host sink", walk=True)
  sink = graph_rewrite(sink, pm_lower_cmdbufs, ctx=ctx, bottom_up=True, name="hcq: lower cmdbufs")
  sink = graph_rewrite(sink, pm_resolve_patches, ctx=ctx, bottom_up=True, name="hcq: resolve patches")
  sink = graph_rewrite(sink, pm_resolve_ref_buffers, ctx=ctx, bottom_up=True, name="hcq: resolve ref buffers")
  sink = graph_rewrite(sink, ctx.dev.pm_lower, ctx=ctx, name="hcq: device lower", walk=True)
  return graph_rewrite(sink, pm_callify, ctx=ctx, name="hcq: callify")

# **************** schedule ****************

@track_rewrites(name=lambda dev,ctx,linear,ast,**kw: f"hcq schedule {getattr(ast.arg, 'name', ast.op.name.lower())}")
def _hcq_schedule(dev:HCQ2Compiled, ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_prep_runtime, ctx=ctx, name="hcq: prepare runtime")
  linear = graph_rewrite(linear, pm_hcq_lower + pm_flatten_linear, ctx=ctx, name="hcq: lower to cmdbuf ops")
  linear = UOp.linear(graph_rewrite(linear, dev.pm_lower, ctx=ctx, name="hcq: encode cmdbuf ops"))
  return hcq_build_host_program(ctx, linear, ast)
