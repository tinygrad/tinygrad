import os, mmap
from typing import Optional
from typing import Callable, Dict
from tinygrad.helpers import prod
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, MovementOps, UnaryOps

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype, device:Optional[str]=None, buf=None, shape=None):
    self.shape = (size, ) if shape is None else shape
    assert device is not None or buf is not None, "disk tensor needs a path or a buf"
    if device is not None:
      with open(device, "a+b") as f:
        if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
        buf = memoryview(mmap.mmap(f.fileno(), size * dtype.itemsize))
    super().__init__(size, dtype, buf)
  def reshape(self, arg): return RawDiskBuffer(self.size, self.dtype, buf=self._buffer(), shape=arg)
  def shrink(self, arg):
    assert arg[1:] == tuple([(0,x) for x in self.shape[1:]]), f"can only slice the first dim of disk tensor {arg}"
    return RawDiskBuffer(arg[0][1]-arg[0][0], self.dtype, buf=self._buffer()[arg[0][0]*prod(self.shape[1:])*self.dtype.itemsize:arg[0][1]*prod(self.shape[1:])*self.dtype.itemsize])
  def _buffer(self): return self._buf

disk_fxn_for_op: Dict[Op, Callable] = { UnaryOps.NOOP: lambda x: x, MovementOps.RESHAPE: RawDiskBuffer.reshape, MovementOps.SHRINK: RawDiskBuffer.shrink }

DiskBuffer = Interpreted(RawDiskBuffer, disk_fxn_for_op, to_underlying=lambda x:x, from_underlying=lambda x:x)