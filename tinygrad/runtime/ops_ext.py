from typing import Tuple, Any
from tinygrad.device import Compiled, Allocator

# the Any is an arbitrary object that's kept in scope with the memoryview
class ExtAllocator(Allocator):
  # NOTE: this doesn't work with allow_zero_copy, it's read only somehow
  #def as_buffer(self, src:Tuple[memoryview, Any]) -> memoryview: return src[0]
  def copyin(self, dest:Tuple[memoryview, Any], src:memoryview): dest[0][:] = src
  def copyout(self, dest:memoryview, src:Tuple[memoryview, Any]): dest[:] = src[0]

class ExtDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, ExtAllocator(), None, None)
