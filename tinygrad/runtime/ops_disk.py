import os, mmap
from typing import Optional
from typing import Callable, Dict
from tinygrad.helpers import prod, DType
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, MovementOps, UnaryOps

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype:DType, device:Optional[str]=None, buf=None, shape=None, offset=0):  # pylint: disable=super-init-not-called
    self.shape = (size, ) if shape is None else shape
    self.offset = offset  # this is an offset in bytes
    assert device is not None or buf is not None, "disk tensor needs a path or a buf"
    if device is not None:
      with open(device, "a+b") as f:
        if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
        buf = mmap.mmap(f.fileno(), size * dtype.itemsize)
        buf.madvise(mmap.MADV_SEQUENTIAL)
    # NOTE: we don't call super since disk tensors don't use RAM
    self.size, self.dtype, self._buf = size, dtype, buf
  def cast(self, new_dtype:DType): return RawDiskBuffer(self.size, new_dtype, buf=self._buf, shape=self.shape, offset=self.offset)
  def reshape(self, arg): return RawDiskBuffer(self.size, self.dtype, buf=self._buf, shape=arg, offset=self.offset)
  def shrink(self, arg):
    assert arg[1:] == tuple([(0,x) for x in self.shape[1:]]), f"can only slice the first dim of disk tensor {arg}"
    offset = arg[0][0]*prod(self.shape[1:])*self.dtype.itemsize
    size = (arg[0][1]-arg[0][0]) * prod(self.shape[1:])
    return RawDiskBuffer(size, self.dtype, buf=self._buf, offset=self.offset+offset, shape=(arg[0][1]-arg[0][0],)+self.shape[1:])
  def _buffer(self): return memoryview(self._buf)[self.offset:self.offset+self.size*self.dtype.itemsize]

disk_fxn_for_op: Dict[Op, Callable] = { UnaryOps.NOOP: lambda x: x, UnaryOps.CAST: RawDiskBuffer.cast, MovementOps.RESHAPE: RawDiskBuffer.reshape, MovementOps.SHRINK: RawDiskBuffer.shrink }

DiskBuffer = Interpreted(RawDiskBuffer, disk_fxn_for_op, to_underlying=lambda x:x, from_underlying=lambda x:x)