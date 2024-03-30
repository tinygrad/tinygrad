import unittest
from typing import Union, Tuple, Optional
from tinygrad.buffer import Buffer, flat_mv
from tinygrad.dtype import DType, dtypes
from random import randbytes, randint, choice, seed

def mvs(mv: memoryview) -> str:
  return str(list(mv))

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
  def view(self, offset:int, size:int, dtype:Optional[DType]=None):
    dtype = dtype if dtype is not None else self.dtype
    assert offset+size <= self.nbytes and size % dtype.itemsize == 0
    return ReferenceBuffer(size//dtype.itemsize, dtype, opaque=self._buf[offset:offset+size])
  def allocate(self, opaque: Optional[memoryview]=None):
    assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"
    self._buf = opaque if opaque is not None else memoryview(bytearray([0] * self.nbytes))
    return self
  def as_buffer(self) -> memoryview:
    return self.copyout(memoryview(bytearray(self.nbytes)))
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
    assert refbufmem == bufmem, f"{mvs(refbufmem)} != {mvs(bufmem)}"
    if expected is not None:
      if not isinstance(expected, memoryview): expected = memoryview(bytearray(expected))
      assert expected == bufmem, f"{mvs(expected)} != {mvs(bufmem)}"
  def view(self, offset:int, size:int, dtype:Optional[DType]=None):
    rb = self.refbuf.view(offset, size, dtype)
    b = self.buf.view(offset, size, dtype)
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
    self.check(mv)
    return self

class TestOffsetBuffer(unittest.TestCase):
  def test_create(self):
    # 0..16
    d1 = DoubleBuffer(16, dtypes.uint8, initial_value=range(16))
    d1.check([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    # tuple = random
    DoubleBuffer(16, dtypes.uint8, initial_value=()).check()
    # empty/unallocated
    DoubleBuffer(16, dtypes.uint8)
    # copyout on empty thing
    with self.assertRaises(Exception):
      DoubleBuffer(16, dtypes.uint8).check()

  def test_simple_onelevel(self):
    # --- create ---
    d1 = DoubleBuffer(16, dtype=dtypes.uint8, initial_value=range(16))

    d3 = d1.view(4,4)

    d3.check([4,5,6,7]) # should offset
    d3.copyin([125,126,127,128]) # copy new stuff to d3
    d3.check([125,126,127,128]) # d3 should change

    # side effects
    d1.check([0,1,2,3,125,126,127,128,8,9,10,11,12,13,14,15]) # part of d1 should change to d3

  def test_base_assign(self):
    d1 = DoubleBuffer(16, dtype=dtypes.uint8, initial_value=range(16))

    d2 = d1.view(4,4)

    d2.check([4,5,6,7]) # should offset
    d1.copyin([255-x for x in range(16)])
    d2.check([251,250,249,248]) # should change because it's cow=False

  def test_changing_dtype(self):
    base = DoubleBuffer(16, dtype=dtypes.uint8, initial_value=range(16))
    i32 = base.view(4,4, dtype=dtypes.int32)
    i32.check([4,5,6,7])
    assert i32.refbuf.size == 1 and i32.refbuf.nbytes == 4
    u16 = i32.view(0,4,dtype=dtypes.uint16)
    u16.check([4,5,6,7])
    assert u16.refbuf.size == 2 and i32.refbuf.nbytes == 4
    with self.assertRaises(AssertionError):
      base.view(0,6).view(0,6,dtype=dtypes.int32)

  def test_fuzz(self):
    seed(1337)
    for _ in range(300):
      try:
        syms = {"base": randint(16,1024)}
        prg = [f"base = DoubleBuffer({syms['base']}, dtype=dtypes.uint8, initial_value=())"]
        def dt(sz:int) -> str:
          avail = ["dtypes.uint8", "dtypes.int8"]
          if sz % 2 == 0: avail.extend(["dtypes.uint16", "dtypes.int16", "dtypes.float16"]*12)
          if sz % 4 == 0: avail.extend(["dtypes.uint32", "dtypes.int32", "dtypes.float32"]*24)
          return choice(avail)
        for _ in range(randint(4, 64)):
          action = randint(0,100)
          if action < 50:
            src,srcsz = list(syms.items())[randint(0, len(syms)-1)]
            if srcsz < 4:
              action = randint(60,100)
              continue
            newof = randint(0,srcsz-1)
            newsz = randint(1,srcsz-newof)
            prg.append(f"b{len(syms)} = {src}.view({newof}, {newsz}, dtype={dt(newsz)})")
            syms[f"b{len(syms)}"] = newsz
          elif action < 90:
            dst = list(syms.keys())[randint(0, len(syms)-1)]
            prg.append(f"{dst}.copyin(())")
          else:
            for _ in range(randint(1, 5)):
              dst = list(syms.keys())[randint(0, len(syms)-1)]
              prg.append(f"{dst}.check()")

        for dst in syms.keys(): prg.append(f"{dst}.check()")

        s = "\n".join(prg)
        #print(f"-----\n{s}\n-----")
        exec(s)
      except:
        print(s)
        raise


if __name__ == "__main__":
  unittest.main()