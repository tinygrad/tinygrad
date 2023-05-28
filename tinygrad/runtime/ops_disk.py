import os, mmap
from typing import Optional
from tinygrad.runtime.lib import RawBufferMapped
from tinygrad.ops import Compiled

class RawDiskBuffer(RawBufferMapped):
  def __init__(self, size, dtype, device:Optional[str]=None):
    assert device is not None, "disk tensor needs a path"
    f = open(device, "a+b")
    if os.path.getsize(device) < size * dtype.itemsize: os.ftruncate(f.fileno(), size * dtype.itemsize)
    buf = mmap.mmap(f.fileno(), size * dtype.itemsize)
    f.close()  # TODO: is this really okay?
    super().__init__(size, dtype, buf)
  def _buffer(self): return memoryview(self._buf)

DiskBuffer = Compiled(RawDiskBuffer, None, None, None)