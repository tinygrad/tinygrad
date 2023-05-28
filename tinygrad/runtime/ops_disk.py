import os, mmap
from typing import Optional
from typing import Callable, Dict
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Interpreted, Op, MovementOps, UnaryOps

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype, device:Optional[str]=None, buf=None):
    assert device is not None or buf is not None, "disk tensor needs a path or a buf"
    if device is not None:
      with open(device, "a+b") as f:
        if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
        buf = memoryview(mmap.mmap(f.fileno(), size * dtype.itemsize))
    super().__init__(size, dtype, buf)
  def _buffer(self): return self._buf

def disk_shrink(x, arg):
  assert len(arg) == 1, "can't slice multidimensional disk tensor"
  return RawDiskBuffer(arg[0][1]-arg[0][0], x.dtype, buf=x._buffer()[arg[0][0]*x.dtype.itemsize:arg[0][1]*x.dtype.itemsize])

disk_fxn_for_op: Dict[Op, Callable] = { UnaryOps.NOOP: lambda x: x, MovementOps.RESHAPE: lambda x, arg: x, MovementOps.SHRINK: disk_shrink }

DiskBuffer = Interpreted(RawDiskBuffer, disk_fxn_for_op, to_underlying=lambda x:x, from_underlying=lambda x:x)