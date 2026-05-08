from __future__ import annotations
from typing import cast, Callable, Type, TypeVar, Generic, Any
import contextlib, decimal, statistics, time, ctypes, array, os, struct, collections, functools, itertools
from dataclasses import replace
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import DEV, PROFILE, getenv, to_mv, from_mv, mv_address, cpu_profile, ProfileRangeEvent, select_first_inited, select_by_name, unwrap
from tinygrad.helpers import suppress_finalizing, pluralize, TracingKey
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.uop.ops import Ops, sym_infer, sint, UOp, UPat, PatternMatcher, buffers, KernelInfo, graph_rewrite, track_rewrites
from tinygrad.dtype import dtypes, dtype_for_fmt
from dataclasses import dataclass, field
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import HWQueue, HCQSignal, hcq_profile
from tinygrad.engine.realize import unwrap_multi, resolve_params, get_runtime, pm_compile, pm_flatten_linear
from tinygrad.codegen import to_program

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

  def __init__(self, device:str, allocator:HCQAllocatorBase, compilers:list[type[Renderer]], runtime, signal_t=None,
               comp_queue_t:Callable[..., HWQueue]|None=None, copy_queue_t:Callable[..., HWQueue]|None=None, kernargs_size=(16 << 20),
               sigalloc_size=0x1000, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.hw_compute_queue_t, self.hw_copy_queue_t = comp_queue_t, copy_queue_t
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
    while sig[0] < tl[0] - 1: pass

  def new_signal(self) -> Buffer:
    if not HCQ2Compiled.signal_pool:
      alc = Buffer(self.device, self.sigalloc_size, dtypes.uint8, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
      HCQ2Compiled.signal_pool += [alc.view(16, dtypes.uint8, off).ensure_allocated() for off in range(0, self.sigalloc_size, 16)]
    return HCQ2Compiled.signal_pool.pop()

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def hw_compute_queues(self) -> list[tuple[str|None, Callable[[], HWQueue]]]:
    return [(None, self.hw_compute_queue_t)] if self.hw_compute_queue_t is not None else []
  def hw_copy_queues(self) -> list[tuple[str, Callable[[], HWQueue]]]:
    return [("SDMA:0", self.hw_copy_queue_t)] if self.hw_copy_queue_t is not None else []

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
    return self._do_map(buf) or buf

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)

  def _as_buffer(self, buf): return buf.cpu_view().mv

def _host_buffer(mv:memoryview) -> Buffer:
  return Buffer("CPU", mv.nbytes, dtypes.uint8, options=BufferSpec(external_ptr=mv_address(mv)), preallocate=True)

@functools.cache
def _hcq_copy_fn():
  n = UOp.variable("n", 1, 1<<31, dtype=dtypes.uint32)
  dst, src = UOp.placeholder((0,), dtypes.uint8, 0), UOp.placeholder((0,), dtypes.uint8, 1)
  i = UOp.range(n.cast(dtypes.int), 0, dtype=dtypes.int)
  sink = UOp.sink(dst[i].store(src[i]).end(i), arg=KernelInfo(name="hcq_copy"))
  prg = to_program(sink, Device["CPU"].renderer)
  return get_runtime("CPU", prg).fxn

class HCQAllocator(HCQAllocatorBase, Generic[HCQDeviceType]):
  def _copy(self, dest:HCQ2Buffer, src:HCQ2Buffer, size:int):
    self.dev.synchronize()
    with cpu_profile(f"copy {size}B", f"{self.dev.device}:COPY"):
      _hcq_copy_fn()(ctypes.c_uint64(dest.va_addr), ctypes.c_uint64(src.va_addr), ctypes.c_int32(size))

  def _copyin(self, dest:HCQ2Buffer, src:memoryview): self._copy(dest, _host_buffer(src)._buf, src.nbytes)
  def _copyout(self, dest:memoryview, src:HCQ2Buffer): self._copy(_host_buffer(dest)._buf, src, dest.nbytes)

  def _transfer(self, dest:HCQ2Buffer, src:HCQ2Buffer, sz:int, src_dev:HCQDeviceType, dest_dev:HCQDeviceType):
    if src_dev.peer_group != dest_dev.peer_group: return src_dev.rdma_dev().allocator._transfer(dest, src, sz, src_dev, dest_dev)
    cast(HCQAllocator, src_dev.allocator)._map(dest)
    src_dev.allocator._copy(dest, src, sz)

# **************** lower context ****************

@dataclass
class HCQ2LowerCtx:
  dev:HCQ2Compiled
  kernargs_buf:Buffer
  kernargs_allocator:BumpAllocator

  # encode-stage state
  blob:bytes = b''
  patches:list[UOp] = field(default_factory=list)

  # launcher-stage state
  inputs:list[Buffer] = field(default_factory=list)

  def addr(self, uop:UOp) -> int: return uop.buffer.get_buf(self.dev.device).va_addr

  def append(self, *data, fmt='I'):
    for d in data:
      if isinstance(d, int): self.blob += struct.pack(f'<{fmt}', d)
      elif d.op is Ops.CONST: self.blob += struct.pack(f'<{fmt}', d.arg)
      else:
        self.blob += struct.pack(f'<{fmt}', 0)
        self.patches.append(UOp(Ops.PARAM, dtype_for_fmt(fmt), src=(d,), arg=len(self.blob) - struct.calcsize(fmt)))

  def param(self, buf:Buffer) -> UOp:
    if buf not in self.inputs: self.inputs.append(buf)
    return UOp.placeholder((buf.size,), buf.dtype, self.inputs.index(buf))

# **************** prep runtime ****************

def do_init_args(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  # TODO: patches!
  data, info = prg.arg
  args_view = ctx.kernargs_buf.view(data.kernargs_alloc_size, dtypes.uint8,
                                    ctx.kernargs_allocator.alloc(data.kernargs_alloc_size, 8)).ensure_allocated()
  bufs = tuple(s.buffer for s in call.src[1:])
  view = args_view._buf.cpu_view()
  for i,gi in enumerate(info.globals): view.view(offset=i*8, size=8, fmt='Q')[0] = bufs[gi].ensure_allocated()._buf.va_addr
  for i,v in enumerate(info.vars):
    if isinstance(v, int): view.view(offset=len(info.globals)*8+i*4, size=4, fmt='I')[0] = v
  return call.replace(src=(prg.replace(src=prg.src + (UOp.from_buffer(args_view, ctx.dev.device),)),) + call.src[1:])

pm_hcq_prep_runtime = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"),),
        name="call", allow_any_len=True), lambda ctx,call,prg: call.replace(src=(ctx.dev.pm_program.rewrite(prg, ctx),) + call.src[1:])),
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER),), name="prg"),), name="call", allow_any_len=True), do_init_args),
])

# **************** lower ****************

def do_lower_program(ctx:HCQ2LowerCtx, call:UOp, prg:UOp) -> UOp:
  sig, tl = UOp.from_buffer(ctx.dev.timeline_signal), ctx.param(ctx.dev.timeline_value)
  return UOp.linear(sig.wait(tl[0] - 1), UOp(Ops.BARRIER, dtypes.void), prg, sig.store(tl[0]))

pm_hcq_lower = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, src=(UPat(Ops.BUFFER), UPat(Ops.BUFFER)), name="prg"),),
        name="call", allow_any_len=True), do_lower_program),
])

# **************** encode ****************

pm_hcq_encode = PatternMatcher([
  (UPat(Ops.LINEAR, name="lin"), lambda ctx, lin: UOp(Ops.BINARY, dtypes.void, arg=ctx.blob, src=tuple(ctx.patches))),
])

# **************** launcher ****************

pm_hcq_create_launcher_sink = PatternMatcher([
  (UPat(Ops.BINARY, name="blob"), lambda blob: UOp.sink(blob, arg=KernelInfo(name="hcq_submit"))),
])

def resolve_blob(ctx:HCQ2LowerCtx, blob:UOp) -> UOp:
  # bufferize blob
  bb = Buffer("CPU", len(blob.arg)//4, dtypes.uint32, preallocate=True)
  bb.copyin(memoryview(bytearray(blob.arg)))

  # patches are related to the new buffer
  bb_param = ctx.param(bb)
  patch_stores = tuple(bb_param[p.arg//4].store(p.src[0]) for p in blob.src)

  submit_cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(bb_param.after(*patch_stores),), arg="submit_compute")

  # bump timeline
  tl = ctx.param(ctx.dev.timeline_value)
  return tl.after(submit_cf)[0].store(tl[0] + 1)

pm_hcq_resolve_blobs = PatternMatcher([
  (UPat(Ops.BINARY, name="blob"), resolve_blob),
])

def hcq_callify(sink:UOp, ctx:HCQ2LowerCtx) -> UOp:
  return to_program(sink, Device["CPU"].renderer).call(*[UOp.from_buffer(b, "CPU") for b in ctx.inputs])

# **************** schedule ****************

@track_rewrites(name=lambda dev,ctx,linear,ast,**kw: f"hcq schedule {ast.arg.name}")
def _hcq_schedule(dev:HCQ2Compiled, ctx:HCQ2LowerCtx, linear:UOp, ast:UOp) -> UOp:
  linear = graph_rewrite(linear, pm_hcq_prep_runtime, ctx=ctx, name="hcq: prepare runtime")
  linear = graph_rewrite(linear, pm_hcq_lower + pm_flatten_linear, ctx=ctx, name="hcq: lower (runtime)")
  linear = graph_rewrite(linear, dev.pm_encode + pm_hcq_encode, ctx=ctx, name="hcq: encode")
  linear = graph_rewrite(linear, pm_hcq_create_launcher_sink, name="hcq: create launcher sink", walk=True)
  linear = graph_rewrite(linear, pm_hcq_resolve_blobs + dev.pm_submit, ctx=ctx, name="hcq: resolve blobs")
  return hcq_callify(linear, ctx)

def hcq_schedule_program(call:UOp, ast:UOp) -> UOp|None:
  # TODO unwrap calls
  dev = Device[ast.src[1].arg]
  ctx = HCQ2LowerCtx(dev=dev, kernargs_buf=dev.kernargs_buf, kernargs_allocator=dev.kernargs_offset_allocator)
  return _hcq_schedule(dev, ctx, UOp.linear(call).rtag((ast.src[1].arg, "COMPUTE")), ast)

pm_hcq_schedule = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_program),
  # (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_copy),
])
