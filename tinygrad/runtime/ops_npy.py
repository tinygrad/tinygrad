import numpy as np
from tinygrad.helpers import flat_mv
from tinygrad.device import Compiled, Allocator

class NpyAllocator(Allocator):
  def copyout(self, dest:memoryview, src:np.ndarray, src_offset:int):
    dest[:] = flat_mv(np.require(src, requirements='C').data)[src_offset:src_offset+len(dest)]

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NpyAllocator(), None, None)
