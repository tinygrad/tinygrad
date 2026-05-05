from __future__ import annotations
from typing import cast, Callable, Type, TypeVar, Generic, Any
import contextlib, decimal, statistics, time, ctypes, array, os, struct, collections, functools, itertools
from dataclasses import replace
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import DEV, PROFILE, getenv, to_mv, from_mv, cpu_profile, ProfileRangeEvent, select_first_inited, select_by_name, unwrap
from tinygrad.helpers import suppress_finalizing, pluralize, TracingKey
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.uop.ops import Ops, sym_infer, sint, UOp, UPat, PatternMatcher, buffers, KernelInfo
from tinygrad.dtype import dtypes
from dataclasses import dataclass, field
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import HWQueue, HCQSignal, hcq_profile

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

SignalType = TypeVar('SignalType', bound='HCQSignal')
HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')
ProgramType = TypeVar('ProgramType', bound='HCQProgram')
ArgsStateType = TypeVar('ArgsStateType', bound='HCQArgsState')

class HCQArgsState(Generic[ProgramType]):
  def __init__(self, buf:Buffer, prg:ProgramType, bufs:tuple[HCQ2Buffer, ...], vals:tuple[sint|None, ...]=()):
    self.buf, self.prg, self.bufs, self.vals = buf, prg, bufs, vals

  def bind_sints_to_buf(self, *vals:sint, buf:Buffer, fmt, offset=0):
    mv = buf._buf.cpu_view().view(offset=offset, size=len(vals)*struct.calcsize(fmt), fmt=fmt)
    for i, v in enumerate(vals):
      if isinstance(v, int): mv[i] = v

class CLikeArgsState(HCQArgsState[ProgramType]):
  def __init__(self, buf:Buffer, prg:ProgramType, bufs:tuple[HCQ2Buffer, ...], vals:tuple[sint|None, ...]=(), prefix:list[int]|None=None):
    super().__init__(buf, prg, bufs, vals=vals)

    if prefix is not None: self.buf._buf.cpu_view().view(size=len(prefix) * 4, fmt='I')[:] = array.array('I', prefix)

    self.bind_sints_to_buf(*[b.va_addr for b in bufs], buf=self.buf, fmt='Q', offset=len(prefix or []) * 4)
    assert None not in vals
    self.bind_sints_to_buf(*cast(tuple[sint, ...], vals), buf=self.buf, fmt='I', offset=len(prefix or []) * 4 + len(bufs) * 8)

class HCQProgram(Generic[HCQDeviceType]):
  def __init__(self, args_state_t:Type[HCQArgsState], dev:HCQDeviceType, name:str, kernargs_alloc_size:int, lib:bytes|None=None, base:int|None=None):
    self.args_state_t, self.dev, self.name, self.kernargs_alloc_size = args_state_t, dev, name, kernargs_alloc_size
    # self.prof_prg_counter = next(self.dev.prof_prg_counter)
    # if PROFILE: Compiled.profile_events += [ProfileProgramEvent(dev.device, name, lib, base, self.prof_prg_counter)]

  @staticmethod
  def _fini(dev, buf, spec): dev.allocator.free(buf, buf.size, spec)

  def fill_kernargs(self, bufs:tuple[HCQ2Buffer, ...], vals:tuple[int|None, ...]=(), kernargs:Buffer|None=None) -> HCQArgsState:
    argsbuf = kernargs or self.dev.kernargs_buf.view(self.kernargs_alloc_size, dtypes.uint8,
                                                     self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8)).ensure_allocated()
    return self.args_state_t(argsbuf, self, bufs, vals=vals)

  def __call__(self, *bufs:HCQ2Buffer, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    assert False
    
    """
    Enqueues the program for execution with the given arguments and dimensions.

    Args:
      bufs: Buffer arguments to execute the kernel with.
      global_size: Specifies the global work size for kernel execution (equivalent to CUDA's grid size).
      local_size: Specifies the local work size for kernel execution (equivalent to CUDA's block size).
      vals: Value arguments to execute the kernel with.
      wait: If True, waits for the kernel to complete execution.

    Returns:
      Execution time of the kernel if 'wait' is True, otherwise None.
    """

    # kernargs = self.fill_kernargs(bufs, vals)
    # q = unwrap(self.dev.hw_compute_queue_t)().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()

    # q.exec(self, kernargs, global_size, local_size)
    # q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
    # if wait: self.dev.synchronize(timeout=timeout)

# **************** encode ****************

@dataclass
class HCQEncodeCtx:
  dev: HCQ2Compiled
  blob: bytes = b''
  patches: list[tuple[int, UOp]] = field(default_factory=list)

def hcq_blob_append(ctx:HCQEncodeCtx, data:list[int|UOp], fmt='I'):
  for d in data:
    if isinstance(d, int): ctx.blob += struct.pack(f'<{fmt}', d)
    elif isinstance(d, UOp):
      ctx.blob += struct.pack(f'<{fmt}', d.arg if d.op is Ops.CONST else 0)
      if d.op is not Ops.CONST: ctx.patches.append((len(ctx.blob) - struct.calcsize(fmt), d))

def hcq_encode(dev:HCQ2Compiled, linear:UOp) -> tuple[UOp, list[tuple[int, UOp]]]:
  ctx = HCQEncodeCtx(dev)
  for op in linear.src: ctx.dev.pm_encode.rewrite(op, ctx=ctx)
  return UOp(Ops.BINARY, dtypes.void, arg=ctx.blob), ctx.patches

# **************** routines ****************

@dataclass
class HCQRoutineCtx:
  device:str
  uops:list[UOp] = field(default_factory=list)
  inputs:list[Buffer] = field(default_factory=list)
  def param(self, buf:Buffer, dtype=dtypes.uint32.ptr()) -> UOp:
    return self.inputs.append(buf) or UOp(Ops.PARAM, dtype, arg=len(self.inputs)-1)

# **************** schedule ****************

# def hcq_schedule_copy(call:UOp, ast:UOp) -> UOp|None:
#   from tinygrad.engine.realize import unwrap_multi, resolve_params
#   ops = []
#   for bufs, _ in unwrap_multi(call, resolved:=resolve_params(call, ())):
#     if Device.canonicalize(bufs[0].device) != Device.canonicalize(bufs[1].device): return None
#     dev = Device[bufs[0].device]
#     if dev.hw_copy_queue_t is None: return None
#     blob = hcq_encode(HCQEncodeCtx(dev, dev.hw_copy_queue_t()), UOp.linear(
#       UOp(Ops.WAIT,  dtypes.void, src=(dev.timeline_uop, dev.timeline_var - 1)),
#       UOp(Ops.COPY,  dtypes.void, src=tuple(resolved)),
#       UOp(Ops.STORE, dtypes.void, src=(dev.timeline_uop, dev.timeline_var)))).replace(tag=ast)
#     ops.append(UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(blob,), arg="invoke", tag=dev.sdma_submit).call(*resolved))
#   return UOp.linear(*ops)

def hcq_schedule_program(call:UOp, ast:UOp) -> UOp|None:
  from tinygrad.engine.realize import unwrap_multi, resolve_params, get_runtime
  from tinygrad.codegen import to_program
  ops = []
  for bufs, _ in unwrap_multi(call, resolve_params(call, ())):
    dev = Device[ast.src[1].arg]
    rt = get_runtime(dev.device, ast)
    args_state = rt.fill_kernargs(tuple(bufs[i].ensure_allocated()._buf for i in ast.arg.globals),
                                  tuple(None if v.expr in ast.arg.runtimevars else v for v in ast.arg.vars))
    blob_uop, patches = hcq_encode(dev, UOp.linear(
      UOp(Ops.WAIT, dtypes.void, src=(UOp.from_buffer(dev.timeline_signal), UOp.const(dtypes.uint64, dev.timeline_value - 1))),
      UOp(Ops.BARRIER, dtypes.void),
      UOp(Ops.PROGRAM, dtypes.void, src=(UOp.from_buffer(rt.lib_gpu, ast.src[1].arg), UOp.from_buffer(args_state.buf, ast.src[1].arg)),
                                    arg=(ast.arg.global_size, ast.arg.local_size or (1, 1, 1), rt, args_state)),
      UOp(Ops.STORE, dtypes.void, src=(UOp.from_buffer(dev.timeline_signal), UOp.const(dtypes.uint64, dev.next_timeline())))))

    (blob_buf:=Buffer("CPU", len(blob_uop.arg), dtypes.uint8, preallocate=True)).copyin(memoryview(bytearray(blob_uop.arg)))

    ctx = HCQRoutineCtx(dev.device)
    blob, size = ctx.param(blob_buf), UOp.const(dtypes.uint32, len(blob_uop.arg)//4)
    ctx.uops += [blob.index(UOp.const(dtypes.int, o//4)).store(v) for o,v in patches]
    dev.pm4_submit_routine(ctx, blob, size)

    sink = UOp.sink(*ctx.uops, arg=KernelInfo(name="submit"))
    prg = to_program(UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="CPU"))), Device["CPU"].renderer)
    ops.append(prg.call(*[UOp.from_buffer(b, "CPU") for b in ctx.inputs]))
  return UOp.linear(*ops)

pm_hcq_schedule = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_program),
  # (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_copy),
])

class HCQ2Compiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  signal_pool: list[Buffer] = []
  cpu_devices: list[HCQ2Compiled] = []

  def __init__(self, device:str, allocator:HCQAllocatorBase, compilers:list[type[Renderer]], runtime, signal_t:Type[SignalType]|None=None,
               comp_queue_t:Callable[..., HWQueue]|None=None, copy_queue_t:Callable[..., HWQueue]|None=None, kernargs_size=(16 << 20),
               sigalloc_size=0x1000, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.hw_compute_queue_t, self.hw_copy_queue_t = comp_queue_t, copy_queue_t
    self.sigalloc_size, self.kernargs_size = sigalloc_size, kernargs_size
    self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(kernargs_size, wrap=True)
    self.timeline_value:int = 1

  @functools.cached_property
  def kernargs_buf(self) -> Buffer:
    return Buffer(self.device, self.kernargs_size, dtypes.uint8, options=BufferSpec(cpu_access=True), preallocate=True)

  @functools.cached_property
  def timeline_signal(self) -> Buffer: return self.new_signal()

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def synchronize(self, timeout:int|None=None):
    if not hasattr(self, 'iface'): return
    sig = self.timeline_signal._buf.view.view(fmt='Q')
    
    import time
    time.sleep(0.2)
    print(sig[0], self.timeline_value - 1)
    # while sig[0] < self.timeline_value - 1: pass
    # if self.error_state is not None: raise self.error_state
    # if not hasattr(self, 'timeline_signal'): return

    # # If we have any work on CPU devices, need to synchronize them. This is just an optimization to release GIL allowing to finish faster.
    # if not self._is_cpu():
    #   for dev in HCQ2Compiled.cpu_devices: dev.synchronize()

    # try: self.timeline_signal.wait(self.timeline_value - 1, timeout=timeout if timeout is not None and self.can_recover else None)
    # except RuntimeError as e:
    #   self.error_state = e
    #   if hasattr(self, 'on_device_hang'): self.on_device_hang()
    #   raise e

    # if self.timeline_value > (1 << 31): self._wrap_timeline_signal()

  def next_timeline(self):
    self.timeline_value += 1
    return self.timeline_value - 1

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

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self.timeline_signal.value = 0
    cast(HCQAllocatorBase, self.allocator).b_timeline = [0] * len(cast(HCQAllocatorBase, self.allocator).b)

  def _realloc(self, oldbuf:HCQ2Buffer|None, new_size:int, options:BufferSpec|None=None, force=False) -> tuple[HCQ2Buffer, bool]:
    if oldbuf is not None: self.allocator.free(oldbuf, oldbuf.size, options=options)
    try: buf, realloced = self.allocator.alloc(new_size, options=options), True
    except MemoryError:
      if force: raise
      buf, realloced = self.allocator.alloc(oldbuf.size if oldbuf is not None else new_size, options=options), False
    return buf, realloced

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

  def _as_buffer(self, buf): return buf.cpu_view()

class HCQAllocator(HCQAllocatorBase, Generic[HCQDeviceType]):
  def _copyin(self, dest:HCQ2Buffer, src:memoryview):
    self.dev.synchronize()
    src_addr = ctypes.addressof(src) if isinstance(src, ctypes.Array) else from_mv(src)
    src_len = len(src) if isinstance(src, ctypes.Array) else src.nbytes
    print(f"  copyin va={hex(dest.va_addr)} src_addr={hex(src_addr) if isinstance(src_addr,int) else 'arr'} type={type(src).__name__} len={src_len} first_src_bytes={bytes(to_mv(src_addr if isinstance(src_addr,int) else ctypes.addressof(src_addr), 16)).hex()}", flush=True)
    with cpu_profile(f'TINY -> {self.dev.device}', f"{self.dev.device}:COPY"): ctypes.memmove(int(dest.va_addr), from_mv(src) if not isinstance(src, ctypes.Array) else src, src_len)
    print(f"  after  va={hex(dest.va_addr)} dst_bytes={bytes(to_mv(dest.va_addr, 16)).hex()}", flush=True)

  def copy_from_disk(self, dest:HCQ2Buffer, src, size):
    def _get_temp_buf():
      # Check if the next buffer is safe to be used (its signal has passed) and reserve it.
      if self.b_timeline[(self.b_next + 1) % len(self.b)] <= self.dev.timeline_signal.value:
        self.b_timeline[(self.b_next + 1) % len(self.b)], self.b_next = (1 << 64), (self.b_next + 1) % len(self.b)
        return (self.b[self.b_next].cpu_view(), self.b_next)
      return None

    assert self.dev.hw_copy_queue_t is not None
    for (batch_info, dst_off, src_off, copy_size) in src.device.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=self.b[0].size,
                                                                                           use_ioring=type(self.b[0].cpu_view()) is MMIOInterface):
      self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                .copy(dest.offset(dst_off), self.b[batch_info[1]].offset(src_off), copy_size) \
                                .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
      self.b_timeline[batch_info[1]] = self.dev.timeline_value - 1

  def _copyout(self, dest:memoryview, src:HCQ2Buffer):
    self.dev.synchronize()
    with cpu_profile(f'{self.dev.device} -> TINY', f"{self.dev.device}:COPY"): ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))

  def _transfer(self, dest:HCQ2Buffer, src:HCQ2Buffer, sz:int, src_dev:HCQDeviceType, dest_dev:HCQDeviceType):
    if src_dev.peer_group != dest_dev.peer_group: return src_dev.rdma_dev().allocator._transfer(dest, src, sz, src_dev, dest_dev)

    cast(HCQAllocator, src_dev.allocator)._map(dest)

    assert src_dev.hw_copy_queue_t is not None
    src_dev.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                             .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                             .copy(dest, src, sz) \
                             .signal(src_dev.timeline_signal, src_dev.next_timeline()).submit(src_dev)

    if src_dev != dest_dev:
      unwrap(dest_dev.hw_compute_queue_t)().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                           .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                           .signal(dest_dev.timeline_signal, dest_dev.next_timeline()).submit(dest_dev)
