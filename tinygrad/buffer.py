from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from weakref import WeakSet
from tinygrad.helpers import flat_mv
from tinygrad.dtype import DType, ImageDType
from tinygrad.ops import GlobalCounters

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  host: bool = False
  nolru: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None, initial_value:Optional[bytes]=None,
               base:Optional[Buffer]=None, offset:int=0, cow:bool=False):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.options, self.base, self.offset, self.cow = device, size, dtype, options, base, offset, cow
    self.views: WeakSet[Buffer] = WeakSet()
    if self.base is not None and self.cow: self.base.views.add(self)
    if opaque is not None: self.allocate(opaque)
    if initial_value is not None:
      self.allocate()
      self.copyin(memoryview(initial_value))
  def allocate(self, opaque=None) -> Buffer:
    assert not hasattr(self, '_buf') or self._buf is None, "can't allocate already allocated buffer"
    from tinygrad.device import Device
    self.allocator = Device[self.device].allocator
    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
    if not self.device.startswith("DISK") and self.base is None: GlobalCounters.mem_used += self.nbytes
    return self
  def view(self, offset:int, size:int, dtype:Optional[DType]=None, cow=True) -> Buffer:
    if not hasattr(self.allocator, "offset"): raise RuntimeError("device doesn't support views")
    if not cow: self.writable()
    dtype, base = self.dtype if dtype is None else dtype, self.base if self.base is not None else self
    assert self.nbytes >= offset + size and size % dtype.itemsize == 0, "wrong size for dtype"
    return Buffer(self.device, size//dtype.itemsize, dtype, self.allocator.offset(base._buf, self.offset+offset, size), self.options, base=base,
                  offset=self.offset+offset, cow=cow)
  def writable(self):
    if self.base is None: # we are the base buffer, create new base buffer and transfer CoW views into it if we have any
      if len(self.views) == 0: return self
      if not hasattr(self.allocator, "offset"): raise RuntimeError("device doesn't support views")
      newbase = Buffer(self.device, self.size, self.dtype, options=self.options, initial_value=self.as_buffer(allow_zero_copy=True))
      newbase.views, self.views = self.views, WeakSet()
      for v in newbase.views: v.base, v._buf = newbase, self.allocator.offset(newbase._buf, v.offset, v.size*v.dtype.itemsize)
    elif self.cow: # we are the CoW view buffer, detach ourselfs from base buffer and convert into base
      self.base.views.remove(self)
      oldbuf, self.base, self._buf, self.views, self.offset, self.cow = self.as_buffer(allow_zero_copy=True), None, None, WeakSet(), 0, False
      self.allocate()
      self.copyin(oldbuf)
    else: self.base.writable()
    return self
  def __reduce__(self):
    buf = None
    if hasattr(self, '_buf'):
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf') or self.base is not None: return
    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
    assert len(self.views) == 0, "attempted to free base that has views"
    self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self):
    return f"<buf real:{hasattr(self, '_buf')} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (">" if self.options is None else f"{self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.writable()
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyout(mv, self._buf)
    return mv