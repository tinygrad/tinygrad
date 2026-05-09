from __future__ import annotations
from typing import cast, Callable, Type, TypeVar, Generic, Any
import contextlib, decimal, statistics, time, ctypes, array, os, struct, collections, functools, itertools
from dataclasses import replace
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import DEV, PROFILE, getenv, to_mv, from_mv, mv_address, cpu_profile, ProfileRangeEvent, select_first_inited, select_by_name, unwrap
from tinygrad.helpers import suppress_finalizing, pluralize, TracingKey, wait_cond
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.uop.ops import Ops, sym_infer, sint, UOp, UPat, PatternMatcher, buffers, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.dtype import dtypes, dtype_for_fmt
from dataclasses import dataclass, field
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import HWQueue, hcq_profile
from tinygrad.engine.realize import unwrap_multi, resolve_params, get_runtime, pm_compile, pm_flatten_linear, estimate_uop
from tinygrad.codegen import to_program
from tinygrad.renderer import Estimates

class MMIOInterface:
  def __init__(self, addr:int, nbytes:int, fmt='B'): self.mv, self.addr, self.nbytes, self.fmt = to_mv(addr, nbytes).cast(fmt), addr, nbytes, fmt
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k): return (self.mv[k] if self.fmt == 'B' else self.mv[k].tolist()) if isinstance(k, slice) else self.mv[k]
  def __setitem__(self, k, v): self.mv[k] = v
  def view(self, offset:int=0, size:int|None=None, fmt=None) -> MMIOInterface:
    return MMIOInterface(self.addr+offset, (self.nbytes - offset) if size is None else size, fmt=fmt or self.fmt)

class FileIOInterface:
  """
  Hardware Abstraction Layer for HCQ devices. The class provides a unified interface for interacting with hardware devices.
  """

  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path:str = path
    self.fd:int = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'): os.close(self.fd)
  def ioctl(self, request, arg): return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, self.fd, offset)
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file: return file.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as file: file.write(content)
  def listdir(self): return os.listdir(self.path)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def _mmap(start, sz, prot, flags, fd, offset):
    x = libc.mmap(start, sz, prot, flags, fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset): return FileIOInterface._mmap(start, sz, prot, flags, -1, offset)
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None): return FileIOInterface(fd=os.eventfd(initval, flags))  # type: ignore[attr-defined]

# **************** for HCQ Compatible Devices ****************

def hcq_filter_visible_devices(devs, device):
  assert (v:=getenv("HCQ_VISIBLE_DEVICES", "")) == "", f"HCQ_VISIBLE_DEVICES={v} is deprecated, use DEV={DEV.target(device, indices=v)} instead"
  if '-' in (idstr:=DEV.target(device).indices): ids = list(range(int(idstr.split('-')[0]), int(idstr.split('-')[1])+1))
  else: ids = [int(x) for x in idstr.split(',') if x.strip()]
  assert all(x < len(devs) for x in ids), f"invalid visibility filter: {ids} ({pluralize('device', len(devs))} available)"
  return [devs[x] for x in ids] if ids else devs

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  signal_pool: list[Buffer] = []
  cpu_devices: list[HCQ2Compiled] = []

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
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fallback to mock ifaces
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # Try to finalize device in any case.
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # If the device has an interface, call its device_fini method to clean up resources.
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
  kernargs_buf:Buffer
  kernargs_allocator:BumpAllocator
  name:str = "hcq_submit"
  estimates:Estimates = field(default_factory=Estimates)
  inputs:list[Buffer] = field(default_factory=list)
  holds:list[UOp] = field(default_factory=list)

  def host_param(self, buf:Buffer) -> UOp:
    if buf not in self.inputs: self.inputs.append(buf)
    return UOp.placeholder((buf.size,), buf.dtype, self.inputs.index(buf))

class HCQEncoder:
  def __init__(self, ctx:HCQ2LowerCtx):
    self.ctx, self.dev, self.blob, self.patches, self.buffers = ctx, ctx.dev, b'', [], []
  @property
  def src(self) -> tuple[UOp, ...]: return tuple(self.patches + self.buffers)
  def unwrap_buf(self, uop:UOp) -> int:
    if uop.op is Ops.AFTER:
      if uop not in self.buffers: self.buffers.append(uop)
    elif uop not in self.ctx.holds: self.ctx.holds.append(uop)
    return uop.buffer.get_buf(self.dev.device).va_addr
  def append(self, *data, fmt='I'):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{fmt}', d)
      elif d.op is Ops.CONST: self.blob += struct.pack(f'<{fmt}', d.arg)
      elif d.op in (Ops.BUFFER, Ops.BUFFER_VIEW): self.blob += struct.pack(f'<{fmt}', self.unwrap_buf(d))
      else:
        self.blob += struct.pack(f'<{fmt}', 0)
        self.patches.append(UOp(Ops.PARAM, dtype_for_fmt(fmt), src=(d,), arg=len(self.blob) - struct.calcsize(fmt)))
  def q(self, *values): self.append(*values)

# **************** prep runtime ****************

def do_init_args(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  data, info = prg.arg
  enc = HCQEncoder(ctx)
  for gi in info.globals: enc.append(call.src[1+gi], fmt='Q')
  for v in info.vars: enc.append(v, fmt='I')

  args_view = ctx.kernargs_buf.view(data.kernargs_alloc_size, dtypes.uint8,
                                    ctx.kernargs_allocator.alloc(data.kernargs_alloc_size, 8)).ensure_allocated()
  args_view._buf.cpu_view().mv[:len(enc.blob)] = enc.blob
  args_uop = UOp.from_buffer(args_view, ctx.dev.device).after(*enc.patches)
  return call.replace(src=(prg.replace(src=prg.src + (args_uop,)),) + call.src[1:])

pm_hcq_prep_runtime = PatternMatcher([
  # device-specific lowering of the program
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"),),
        name="call", allow_any_len=True), lambda ctx,call,prg: call.replace(src=(ctx.dev.pm_lower.rewrite(prg, ctx),) + call.src[1:])),

  # init args for programs
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER),), name="prg"),), name="call", allow_any_len=True), do_init_args),
])

# **************** lower hcq ****************

def do_lower_program(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  sig, tl = UOp.from_buffer(ctx.dev.timeline_signal), ctx.host_param(ctx.dev.timeline_value)
  return UOp.linear(sig.wait(tl[0] - 1), UOp(Ops.BARRIER, dtypes.void), prg, sig.store(tl[0]))

def do_lower_copy(ctx:HCQ2LowerCtx, call:UOp, copy:UOp) -> UOp:
  dst, src, dev = call.src[1], call.src[2], ctx.dev
  src_dev = Device[src.device]
  devs = [dev, src_dev] if src_dev is not dev else [dev]
  sigs_tls = [(UOp.from_buffer(d.timeline_signal), ctx.host_param(d.timeline_value)) for d in devs]
  return UOp.linear(*[s.wait(t[0] - 1) for s,t in sigs_tls], UOp(Ops.BARRIER, dtypes.void),
                    UOp(Ops.COPY, dtypes.void, src=(dst, src), arg=src.buffer.nbytes),
                    *[s.store(t[0]) for s,t in sigs_tls])

pm_hcq_lower = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER), UPat()), name="prg"),), name="call", allow_any_len=True), do_lower_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="copy"),), name="call", allow_any_len=True), do_lower_copy),
])

# **************** build host program ****************

pm_hcq_create_host_sink = PatternMatcher([
  (UPat(Ops.LINEAR, name="l", allow_any_len=True), lambda ctx, l: UOp.sink(*l.src, arg=KernelInfo(name=ctx.name, estimates=ctx.estimates), tag=1))
])

def resolve_cmdbuf(ctx:HCQ2LowerCtx, blob:UOp) -> UOp:
  inner = blob.src[0] if blob.op is Ops.AFTER else blob
  bb = Buffer("CPU", len(inner.arg)//4, dtypes.uint32, preallocate=True)
  bb.copyin(memoryview(bytearray(inner.arg)))
  bb_param = ctx.host_param(bb)
  submit_cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(bb_param.after(*(blob.src[1:] if blob.op is Ops.AFTER else ())),),
                  arg=f"submit_{inner.tag.lower()}")
  tl = ctx.host_param(ctx.dev.timeline_value)
  return tl.after(UOp(Ops.BARRIER, dtypes.void, src=(submit_cf,))).index(0, ptr=True).store(tl[0] + 1)

def resolve_patches(ctx:HCQ2LowerCtx, buf:UOp) -> UOp|None:
  if buf.op is Ops.BUFFER: return ctx.host_param(buf.buffer)
  if buf.op is Ops.PARAM: return None
  bb_param = buf.src[0] if buf.src[0].op is Ops.PARAM else ctx.host_param(buf.src[0].buffer)
  return bb_param.after(*(bb_param.index(p.arg//bb_param.dtype.base.itemsize, ptr=True).cast(p.dtype.ptr()).store(p.src[0].cast(p.dtype))
                          if p.op is Ops.PARAM else p for p in buf.src[1:]))

pm_hcq_resolve = PatternMatcher([
  (UPat(Ops.BINARY).or_after("blob"), resolve_cmdbuf),
  (UPat((Ops.BUFFER, Ops.PARAM)).or_after("buf"), resolve_patches),
])

def hcq_callify(ctx:HCQ2LowerCtx, sink:UOp) -> UOp:
  call = to_program(sink, Device["CPU"].renderer).call(*[UOp.from_buffer(b, "CPU") if isinstance(b, Buffer) else b for b in ctx.inputs])
  return call.replace(src=call.src + (UOp(Ops.BIND, dtypes.void, src=tuple(ctx.holds)),)) if ctx.holds else call
pm_hcq_callify = PatternMatcher([(UPat(Ops.SINK, name="sink"), hcq_callify)])

def hcq_build_host_program(ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  sink = graph_rewrite(linear, pm_hcq_create_host_sink, ctx=ctx, name="hcq: create host sink", walk=True)
  sink = graph_rewrite(sink, pm_hcq_resolve, ctx=ctx, bottom_up=True, name="hcq: resolve blobs")
  sink = graph_rewrite(sink, ctx.dev.pm_lower, ctx=ctx, name="hcq: device lower", walk=True)
  return graph_rewrite(sink, pm_hcq_callify, ctx=ctx, name="hcq: callify")

# **************** schedule ****************

@track_rewrites(name=lambda dev,ctx,linear,ast,**kw: f"hcq schedule {getattr(ast.arg, 'name', ast.op.name.lower())}")
def _hcq_schedule(dev:HCQ2Compiled, ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_hcq_prep_runtime, ctx=ctx, name="hcq: prepare runtime")
  linear = graph_rewrite(linear, pm_hcq_lower + pm_flatten_linear, ctx=ctx, name="hcq: lower to cmdbuf ops")
  linear = UOp.linear(graph_rewrite(linear, dev.pm_lower, ctx=ctx, name="hcq: encode cmdbuf ops"))
  return hcq_build_host_program(ctx, linear, ast)

def hcq_schedule_program(call:UOp, ast:UOp) -> UOp|None: # TODO: clean up
  # TODO unwrap calls
  dev = Device[ast.src[1].arg]
  ctx = HCQ2LowerCtx(dev=dev, kernargs_buf=dev.kernargs_buf, kernargs_allocator=dev.kernargs_offset_allocator,
                     name=f"submit_{ast.arg.name}", estimates=estimate_uop(call))
  return _hcq_schedule(dev, ctx, UOp.linear(call).rtag("COMPUTE"), ast)

def hcq_schedule_copy(call:UOp, ast:UOp) -> UOp|None:
  dev = Device[ast.src[1].arg]
  ctx = HCQ2LowerCtx(dev=dev, kernargs_buf=dev.kernargs_buf, kernargs_allocator=dev.kernargs_offset_allocator,
                     name=ast.op.name.lower(), estimates=estimate_uop(call))
  src_buf = call.src[2].buffer
  try: src_buf.get_buf(dev.device)
  except Exception:
    (cpubuf := Buffer("CPU", src_buf.nbytes, dtypes.uint8, preallocate=True)).copyin(src_buf.as_memoryview())
    ctx.holds.append(buf_uop:=UOp.from_buffer(cpubuf, dev.device))
    call = call.replace(src=call.src[:2] + (buf_uop,) + call.src[3:])
  return _hcq_schedule(dev, ctx, UOp.linear(call).rtag("COPY"), ast)

pm_hcq_schedule = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_copy),
])
