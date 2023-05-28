from typing import Optional
from tinygrad.runtime.lib import RawBuffer
from tinygrad.ops import Compiled

class RawDiskBuffer(RawBuffer):
  def __init__(self, size, dtype, device:Optional[str]=None):
    super().__init__(size, dtype, buf)
  @classmethod
  def fromCPU(cls, x, device:Optional[str]=None):
    pass
  def toCPU(self):
    pass

DiskBuffer = Compiled(RawDiskBuffer, None, None, None)