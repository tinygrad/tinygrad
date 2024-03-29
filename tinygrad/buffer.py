from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
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
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None, initial_value:Optional[bytes]=None):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.options = device, size, dtype, options
    if opaque is not None: self.allocate(opaque)
    if initial_value is not None:
      self.allocate()
      self.copyin(memoryview(initial_value))
  def allocate(self, opaque=None) -> Buffer:
    assert not hasattr(self, '_buf'), "can't alloc"
    from tinygrad.device import Device
    self.allocator = Device[self.device].allocator
    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
    if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
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
    if not hasattr(self, '_buf'): return
    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
    self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self): return f"<buf device:{self.device} size:{self.size} dtype:{self.dtype}" + (">" if self.options is None else f"{self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyout(mv, self._buf)
    return mv