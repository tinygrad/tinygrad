import unittest
from typing import Any, Union, Tuple, Optional
from tinygrad.buffer import Buffer, flat_mv
from tinygrad.dtype import DType, dtypes
from random import randbytes

# Simplest thing that works to check behaviour of sophisticated Buffer with zero copy, CoW and stuff agaisnt
class ReferenceBuffer:
  def __init__(self, size: int, dtype: DType, opaque: Optional[memoryview] = None, initial_value: Optional[bytes] = None):
    self.size, self.dtype = size, dtype
    if opaque is not None: self.allocate(opaque)
    if initial_value is not None:
      self.allocate()
      self.copyin(memoryview(initial_value))
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def view(self, offset:int, size:int, dtype:Optional[DType]=None, cow=True):
    dtype = dtype if dtype is not None else self.dtype
    assert offset+size <= self.nbytes and size % dtype.itemsize == 0
    if cow:
      return ReferenceBuffer(size//dtype.itemsize, dtype, initial_value=self.as_buffer()[offset:offset+size])
    else:
      return ReferenceBuffer(size//dtype.itemsize, dtype, opaque=self.as_buffer()[offset:offset+size])
  def allocate(self, opaque: Optional[memoryview]=None):
    assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"
    self._buf = opaque if opaque is not None else memoryview(bytearray([0] * self.nbytes))
    return self
  def as_buffer(self) -> memoryview:
    return self._buf
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self._buf[:] = mv
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    mv[:] = self._buf
    return mv

# This thing wraps both ReferenceBuffer and Buffer from tinygrad, does same operations on both and checks if results match
class DoubleBuffer:
  def __init__(self, size:int, dtype:DType, initial_value:Optional[Union[bytes,Tuple]]=None, bufs:Optional[Tuple[Buffer,ReferenceBuffer]]=None):
    if bufs is not None:
      self.buf = bufs[0]
      self.refbuf = bufs[1]
      assert initial_value is None, size == self.refbuf.size and dtype == self.refbuf.dtype
      self.check()
    else:
      if isinstance(initial_value, tuple): initial_value = randbytes(size*dtype.itemsize)
      if not isinstance(initial_value, memoryview) and initial_value is not None: initial_value = memoryview(bytearray(initial_value))
      self.buf = Buffer("CLANG", size, dtype, initial_value=initial_value)
      self.refbuf = ReferenceBuffer(size, dtype, initial_value=initial_value)
      if initial_value is not None: self.check(initial_value)
  def check(self, expected: Optional[bytes]=None):
    assert self.refbuf.dtype == self.buf.dtype, "dtype mismatch"
    assert self.refbuf.size == self.buf.size, "size mismatch"
    bufmem = self.buf.as_buffer()
    refbufmem = self.refbuf.as_buffer()
    assert bufmem == refbufmem, "bufmem != refbufmem"
    if expected is not None:
      if not isinstance(expected, memoryview): expected = memoryview(bytearray(expected))
      assert refbufmem == expected, "mem != expected"
  def view(self, offset:int, size:int, dtype:Optional[DType]=None, cow=True):
    rb = self.refbuf.view(offset, size, dtype, cow)
    b = self.buf.view(offset, size, dtype, cow)
    return DoubleBuffer(rb.size, rb.dtype, bufs=(b, rb))
  def allocate(self):
    self.buf.allocate()
    self.refbuf.allocate()
    self.check()
    return self
  def copyin(self, mv:Union[memoryview, Tuple]):
    if isinstance(mv, tuple): mv = randbytes(self.refbuf.nbytes)
    if not isinstance(mv, memoryview): mv = memoryview(bytearray(mv))
    self.buf.copyin(mv)
    self.refbuf.copyin(mv)
    self.check()
    return self

class TestOffsetBuffer(unittest.TestCase):
  def test_create(self):
    # 0..16
    d1 = DoubleBuffer(16, dtypes.uint8, initial_value=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    d1.check([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    # tuple = random
    DoubleBuffer(16, dtypes.uint8, initial_value=())
    # empty/unallocated
    DoubleBuffer(16, dtypes.uint8)
    # copyout on empty thing
    with self.assertRaises(Exception):
      DoubleBuffer(16, dtypes.uint8).check()
  def test_simple_onelevel(self):
    # --- create ---
    d1 = DoubleBuffer(16, dtype=dtypes.uint8, initial_value=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    d1.check([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # should copy

    # --- test cow=True ---
    d2 = d1.view(4,4)

    d2.check([4,5,6,7]) # should offset
    d2.copyin([255,254,253,252])
    d2.check([255,254,253,252]) # d2 should change

    # side effects
    d1.check([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # d1 shouldn't change because cow=True

    # --- test cow=False ---
    d3 = d1.view(4,4, cow=False)

    d3.check([4,5,6,7]) # should offset
    d3.copyin([125,126,127,128]) # copy new stuff to d3
    d3.check([125,126,127,128]) # d3 should change

    # side effects
    d1.check([0,1,2,3,125,126,127,128,8,9,10,11,12,13,14,15]) # part of d1 should change to d3 because cow=False
    d2.check([255,254,253,252]) # d2 shouldn't change because it was forked before copyin to cow=False view

if __name__ == "__main__":
  unittest.main()
