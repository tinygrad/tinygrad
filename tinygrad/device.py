from __future__ import annotations
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Any, cast
import importlib, inspect, functools, pathlib, os, ctypes
from tinygrad.helpers import getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, from_mv
from tinygrad.dtype import DType, ImageDType
from tinygrad.renderer import Renderer

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._devices: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]  # noqa: E501
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "")   # noqa: E501
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT
  def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    assert ((cpn:=multiprocessing.current_process().name) == "MainProcess") or ix.split(":")[0] in ["DISK", "NPY"], \
      f"can only open device {ix} from parent, not {cpn}"
    x = ix.split(":")[0].upper()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    return ret
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._devices, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "AMD", "CUDA", "GPU", "CLANG", "LLVM"]:
      try:
        if self[device]:
          os.environ[device] = "1"   # we set this in environment for spawned children
          return device
      except Exception: pass
    raise RuntimeError("no usable devices")
Device = _Device()

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None,
               initial_value:Optional[bytes]=None, lb_refcount=0, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.options, self.offset = device, size, dtype, options, offset
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._lb_refcount = lb_refcount
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def lb_refcount(self): return self.base._lb_refcount
  def ref(self, cnt): self.base._lb_refcount += cnt
  def is_allocated(self) -> bool: return hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not hasattr(self, '_buf') else self
  def allocate(self, opaque=None) -> Buffer:
    assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"
    self.allocator = Device[self.device].allocator
    if self._base is not None:
      self._base.ensure_allocated()
      assert hasattr(self.allocator, "offset"), "offset function required for view"
      self._buf: Any = self.allocator.offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
    return self
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, hasattr(self, '_buf'))
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.lb_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.lb_refcount)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf'): return
    if self._base is None:
      if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
      self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self):
    return f"<buf real:{hasattr(self, '_buf')} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if hasattr(self, "base") else "") + \
           (">" if self.options is None else f" {self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyin to unallocated buffer"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyout unallocated buffer"
    self.allocator.copyout(mv, self._buf)
    return mv
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    if self._base is not None: return Buffer(self.device, size, dtype, base=self._base, offset=self.offset+offset)
    return Buffer(self.device, size, dtype, base=self, offset=offset)

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    assert not isinstance(size, int) or size > 0, f"alloc size must be positve, getting {size}"
    return self._alloc(size, options if options is not None else BufferOptions())
  def _alloc(self, size:int, options:BufferOptions): raise NotImplementedError("need alloc")
  def free(self, opaque, size:int, options:Optional[BufferOptions]=None):
    self._free(opaque, options if options is not None else BufferOptions())
  def _free(self, opaque, options:BufferOptions): pass  # if opaque is a Python object, you don't need a free
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):  # pylint: disable=abstract-method
  def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):
    if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferOptions): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
  def offset(self, buf, size:int, offset:int): return from_mv(self.as_buffer(buf)[offset:offset+size])

MallocAllocator = _MallocAllocator()

# **************** for Compiled Devices ****************

class CompileError(Exception): pass

class Compiler:
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def compile(self, src:str) -> bytes: raise NotImplementedError("need a compile function")
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib

class Compiled:
  def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):
    self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler if compiler else Compiler(), runtime, graph
    self.renderer = renderer if renderer else Renderer()
  def synchronize(self): pass  # override this in your device

# **************** for HCQ Compatible Devices ****************

class HCQCompatCompiled(Compiled):
  def __init__(self, device:str, allocator:Allocator, renderer:Renderer, compiler:Compiler, runtime, comp_queue_t, copy_queue_t, timeline_signals):
    self.hw_compute_queue_t, self.hw_copy_queue_t = comp_queue_t, copy_queue_t
    self.timeline_value: int = 1
    self.timeline_signal, self._shadow_timeline_signal = timeline_signals

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, allocator, renderer, compiler, runtime, HCQGraph)

  @classmethod
  def _read_signal(self, sig): raise NotImplementedError("need _read_signal") # reads a value for a signal

  @classmethod
  def _set_signal(self, sig, value): raise NotImplementedError("need _set_signal") # sets a value for a signal

  @classmethod
  def _get_signal(self, value=0, **kwargs): raise NotImplementedError("need _get_signal") # allocates a new signal

  @classmethod
  def _wait_signal(self, signal, value=0, timeout=10000): raise NotImplementedError("need _wait_signal") # waits for a signal value

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self._set_signal(self.timeline_signal, 0)
    cast(HCQCompatAllocator, self.allocator).b_timeline = [0] * len(cast(HCQCompatAllocator, self.allocator).b)

class HCQCompatAllocator(LRUAllocator): # pylint: disable=abstract-method
  def __init__(self, device, batch_size=(2 << 20), batch_cnt=32):
    self.device = device
    self.b = [self._alloc(batch_size, BufferOptions(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next = [0] * len(self.b), 0
    super().__init__()

  def copyin(self, dest, src: memoryview):
    for i in range(0, src.nbytes, self.b[0].size):
      self.b_next = (self.b_next + 1) % len(self.b)
      self.device._wait_signal(self.device.timeline_signal, self.b_timeline[self.b_next])
      ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].size, src.nbytes-i))
      self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                   .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.b_timeline[self.b_next] = self.device.timeline_value
      self.device.timeline_value += 1

  def copy_from_disk(self, dest, src, size):
    def _get_temp_buf():
      # Check if the next buffer is safe to be used (its signal has passed) and reserve it.
      if self.b_timeline[(self.b_next + 1) % len(self.b)] <= self.device._read_signal(self.device.timeline_signal):
        self.b_timeline[(self.b_next + 1) % len(self.b)], self.b_next = (1 << 64), (self.b_next + 1) % len(self.b)
        return (self.b[self.b_next].va_addr, self.b_next)
      return None

    for (batch_info, dst_off, src_off, copy_size) in src.device.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=self.b[0].size):
      self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                   .copy(dest.va_addr + dst_off, batch_info[0] + src_off, copy_size) \
                                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.b_timeline[batch_info[1]] = self.device.timeline_value
      self.device.timeline_value += 1

  def copyout(self, dest:memoryview, src):
    self.device.synchronize()
    for i in range(0, dest.nbytes, self.b[0].size):
      self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                   .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i)) \
                                   .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
      self.device._wait_signal(self.device.timeline_signal, self.device.timeline_value)
      self.device.timeline_value += 1

      ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest, src, sz: int, src_dev, dest_dev):
    src_dev._gpu_map(dest)
    self.device.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                 .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                 .copy(dest.va_addr, src.va_addr, sz) \
                                 .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
    self.device.hw_compute_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value).submit(dest_dev)
    src_dev.timeline_value += 1

  def offset(self, buf, size:int, offset:int): return type(buf)(base=buf.base + offset, va_addr=buf.va_addr + offset, length=size, size=size)
