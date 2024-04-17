from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from tinygrad.helpers import GlobalCounters, flat_mv
from tinygrad.dtype import DType, ImageDType

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  host: bool = False
  nolru: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None,
               initial_value:Optional[bytes]=None, lb_refcount=0):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.options, self.lb_refcount = device, size, dtype, options, lb_refcount
    if opaque is not None: self.allocate(opaque)
    if initial_value is not None:
      self.allocate()
      self.copyin(memoryview(initial_value))
  def is_allocated(self) -> bool: return hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not hasattr(self, '_buf') else self
  def allocate(self, opaque=None) -> Buffer:
    assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"
    from tinygrad.device import Device
    self.allocator = Device[self.device].allocator
    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
    if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
    return self
  def __reduce__(self):
    buf = None
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.lb_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.lb_refcount)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf'): return
    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
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
    assert self.is_allocated(), "can't copyin to unallocated buffer"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyout unallocated buffer"
    self.allocator.copyout(mv, self._buf)
    return mv
